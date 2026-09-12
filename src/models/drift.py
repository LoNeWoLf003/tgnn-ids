"""Concept-drift detection + adaptation strategies (Days 10-11 / 10-14).

Two halves, per IMPLEMENTATION.md §6-7:
  1. `EmbeddingDriftDetector` -- an ADWIN wrapper over a scalar statistic derived from the
     model's per-window embedding Z_t, plus an optional Jensen-Shannon secondary signal.
  2. Three adaptation strategies behind one shape (`no_adapt` / `full_retrain` /
     `incremental_adapt`) and `run_drift_adaptation_loop`, which wires a detector + a chosen
     strategy together over a chronological graph sequence -- this is what eval.py's
     Experiments B/C/D call.

Drift phases are the CIC-IDS2017 day-files themselves (PLAN.md's "native drift, not
fabricated" decision) -- Mon = benign baseline, Tue = brute-force introduced, Wed = DoS,
Thu = web-attack/infiltration (held out of training as "unseen"), Fri = botnet/portscan/DDoS.

This module is imported by eval.py (which lives alongside train.py in src/), never run
standalone -- `import train` below resolves because the *entry point* script (eval.py) sits in
src/, putting src/ on sys.path[0], the same mechanism train.py itself relies on for
`from models.gat_lstm import TemporalGAT`.
"""

import copy
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from river.drift import ADWIN
from scipy.spatial.distance import jensenshannon

from train import (
    CHECKPOINT_DIR,
    DEVICE,
    HIDDEN,
    K_LOOKBACK,
    LSTM_HIDDEN,
    RESULTS_DIR,
    GraphWindowDataset,
    train as train_loop,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 6.1 Mon-Fri day-file -> drift-phase mapping (locked in per PLAN.md)
# ---------------------------------------------------------------------------

DAY_ORDER = ["mon", "tue", "wed", "thu_am", "thu_pm", "fri_am", "fri_pm_scan", "fri_pm_ddos"]

PHASE_MAP = {
    "mon": 1,       # benign baseline
    "tue": 2,       # brute-force introduced
    "wed": 3,       # DoS
    "thu_am": 4, "thu_pm": 4,   # web attack / infiltration -- held out of training as "unseen"
    "fri_am": 5, "fri_pm_scan": 5, "fri_pm_ddos": 5,   # botnet/portscan/DDoS
}

# Ground truth for degradation plots -- day cutovers, not injected/synthetic drift points.
PHASE_BOUNDARIES = ["mon->tue", "tue->wed", "wed->thu_am", "thu_pm->fri_am"]


def phase_boundary_window_ids(graphs: list) -> dict:
    """First `window_id` seen for each day in `graphs` (already sorted chronologically) --
    x-axis positions for the PHASE_BOUNDARIES vertical lines in eval.py's degradation plot."""
    first_seen = {}
    for g in graphs:
        first_seen.setdefault(g.day, g.window_id)
    return {day: first_seen[day] for day in DAY_ORDER if day in first_seen}


# ---------------------------------------------------------------------------
# 6.2 ADWIN wrapper over the embedding stream
# ---------------------------------------------------------------------------

class EmbeddingDriftDetector:
    """ADWIN wrapper over a scalar drift statistic derived from the model's per-window
    embedding Z_t.

    river>=0.15's `ADWIN.update(x)` mutates internal state and returns `None`; whether a drift
    was just detected is read off the `.drift_detected` attribute afterwards -- confirmed
    against the installed river==0.23.0 here, which differs from the tuple-return
    `in_drift, _ = adwin.update(x)` API IMPLEMENTATION.md's pseudocode shows (an older river
    release's signature).
    """

    def __init__(self, delta: float = 0.002):
        self.adwin = ADWIN(delta=delta)
        self.drift_points: list = []

    def update(self, z_stat: float, window_id) -> bool:
        self.adwin.update(z_stat)
        in_drift = bool(self.adwin.drift_detected)
        if in_drift:
            self.drift_points.append(window_id)
        return in_drift


def embedding_l2_norm(model, graph_window: list) -> float:
    """Cheap, no-extra-model scalar drift statistic: ||Z_t||_2 (IMPLEMENTATION.md §6.2)."""
    z_t = model.embed(graph_window)
    return float(torch.linalg.norm(z_t).item())


def js_divergence_drift(z_window_a: np.ndarray, z_window_b: np.ndarray, bins: int = 20) -> float:
    """Secondary drift signal (optional -- PLAN.md cut-line #2, drop first if behind schedule).
    Jensen-Shannon divergence between two embedding-statistic distributions. Not wired into
    `run_drift_adaptation_loop` by default; call directly to corroborate an ADWIN firing."""
    hist_a, edges = np.histogram(z_window_a, bins=bins, density=True)
    hist_b, _ = np.histogram(z_window_b, bins=edges, density=True)
    return float(jensenshannon(hist_a + 1e-12, hist_b + 1e-12))


# ---------------------------------------------------------------------------
# 7. Adaptation strategies
# ---------------------------------------------------------------------------

def no_adapt(model, *_args, **_kwargs):
    """Unchanged -- baseline for comparison (§7)."""
    return model


def full_retrain(model_class, model_kwargs: dict, all_graphs_so_far: list, class_to_idx: dict,
                  k: int = K_LOOKBACK, lr: float = 1e-3, num_epochs: int = 10):
    """From scratch on every window seen so far, per §1.11's naive approach. Reuses train.py's
    own checkpointed loop (so a crash mid-retrain is still resumable), writing to a scratch
    checkpoint that's intentionally overwritten by every retrain event -- it's a means to an
    end (producing the retrained model), not an artifact worth keeping per-event."""
    model = model_class(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = GraphWindowDataset(all_graphs_so_far, class_to_idx, k=k)
    val_ds = GraphWindowDataset([], class_to_idx, k=k)  # no held-out slice during a drift retrain
    if len(train_ds) == 0:
        log.warning("full_retrain called with < k+1 graphs -- returning an untrained model")
        return model
    train_loop(
        model, train_ds, val_ds, optimizer,
        checkpoint_path=CHECKPOINT_DIR / "gat_lstm_retrain_scratch.pt",
        num_epochs=num_epochs, log_path=RESULTS_DIR / "train_log_retrain_events.csv",
    )
    return model


def incremental_adapt(model, recent_graphs: list, class_to_idx: dict, k: int = K_LOOKBACK,
                       layers_to_unfreeze=("lstm", "anomaly_head"), lr: float = 1e-4,
                       num_epochs: int = 3):
    """Drift-triggered fine-tune of only `layers_to_unfreeze`, a few epochs, only on recent
    data (§7). Mutates `model` in place and returns it; restores full trainability afterwards
    so a later `full_retrain` (or another `incremental_adapt` call) isn't left partially frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    ds = GraphWindowDataset(recent_graphs, class_to_idx, k=k)
    if len(ds) == 0:
        log.warning("incremental_adapt called with < k+1 recent graphs -- skipping this event")
        for param in model.parameters():
            param.requires_grad = True
        return model

    criterion_anom = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(num_epochs):
        for idx in range(len(ds)):
            window, y_anom, y_cls = ds[idx]
            y_anom, y_cls = y_anom.to(DEVICE), y_cls.to(DEVICE)
            optimizer.zero_grad()
            score, logits = model(window)
            loss = criterion_anom(score, y_anom) + 0.5 * criterion_cls(logits, y_cls)
            loss.backward()
            optimizer.step()

    for param in model.parameters():
        param.requires_grad = True
    return model


def state_dict_checksum(model) -> str:
    """SHA256 over concatenated flattened parameter tensors -- used by the no_adapt /
    full_retrain / incremental verification drill in IMPLEMENTATION.md §7 (confirm `no_adapt`
    never changes weights; confirm the other two strategies do)."""
    h = hashlib.sha256()
    for _, tensor in sorted(model.state_dict().items()):
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Trigger loop: detector + strategy wired over a chronological graph sequence
# ---------------------------------------------------------------------------

def run_drift_adaptation_loop(model, graphs: list, class_to_idx: dict, strategy: str,
                               k: int = K_LOOKBACK, delta: float = 0.002,
                               recent_span: int = 50, num_epochs_incremental: int = 3,
                               num_epochs_retrain: int = 10, lr_incremental: float = 1e-4,
                               lr_retrain: float = 1e-3):
    """Score `graphs` window-by-window in chronological order, applying `strategy` whenever
    the ADWIN detector fires on the ||Z_t|| stream (§7's trigger logic). Metrics are always
    logged *before* any adaptation for that window, matching the guide's ordering, so an
    adaptation event never retroactively improves the very score that triggered it.

    `model` is deep-copied up front -- the caller's already-trained (stationary-split) model is
    never mutated, so the same base checkpoint can be reused across strategies/experiments.

    Returns:
      records: list[dict] with keys {window_id, day, y_true, y_score}, one per scored window.
      events: list[dict] with keys {window_id, day, wall_clock_s, peak_gpu_mem_bytes,
        n_samples}, one per adaptation event (empty for strategy="no_adapt").
    """
    if strategy not in ("no_adapt", "full_retrain", "incremental"):
        raise ValueError(f"unknown strategy {strategy!r}")

    model = copy.deepcopy(model)
    detector = EmbeddingDriftDetector(delta=delta)
    ds = GraphWindowDataset(graphs, class_to_idx, k=k)
    in_dim = graphs[0].x.shape[1]
    num_classes = model.class_head.out_features

    records, events = [], []
    for i in range(len(ds)):
        window, y_anom, _ = ds[i]

        model.eval()
        with torch.no_grad():
            score, _logits = model(window)
        target = window[-1]
        records.append({
            "window_id": target.window_id, "day": target.day,
            "y_true": y_anom.item(), "y_score": score.item(),
        })

        z_stat = embedding_l2_norm(model, window)
        drifted = detector.update(z_stat, target.window_id)
        if not drifted or strategy == "no_adapt":
            continue

        log.info("drift detected at window %s (strategy=%s)", target.window_id, strategy)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()

        graph_pos = i + k  # index into `graphs` of the current (last) window in this lookback
        if strategy == "incremental":
            recent = graphs[max(graph_pos - recent_span, 0): graph_pos + 1]
            model = incremental_adapt(model, recent, class_to_idx, k=k,
                                       num_epochs=num_epochs_incremental, lr=lr_incremental)
            n_samples = len(recent)
        else:  # full_retrain
            seen_so_far = graphs[: graph_pos + 1]
            model = full_retrain(
                type(model), dict(in_dim=in_dim, hidden=HIDDEN, lstm_hidden=LSTM_HIDDEN,
                                   num_classes=num_classes),
                seen_so_far, class_to_idx, k=k, num_epochs=num_epochs_retrain, lr=lr_retrain,
            )
            n_samples = len(seen_so_far)

        wall = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) if torch.cuda.is_available() else 0
        events.append({
            "window_id": target.window_id, "day": target.day,
            "wall_clock_s": wall, "peak_gpu_mem_bytes": peak_mem, "n_samples": n_samples,
        })
        log.info("adaptation event @ window %s (%s): %.2fs, peak_mem=%d bytes, n=%d",
                  target.window_id, strategy, wall, peak_mem, n_samples)

    return records, events
