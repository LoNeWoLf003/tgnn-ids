"""Experiments A-D + ablation (Days 12-14) -- IMPLEMENTATION.md §8.

Single source of truth: every per-window row this module produces is appended to
`results/experiment_log.csv` (columns `[experiment, model, day, window_id, f1, pr_auc, roc_auc,
fpr, phase]`); every experiment-specific CSV and figure is generated *from* that data, never
hand-typed, per PLAN.md's done-criterion.

- **A (stationary):** Mon+Tue, time-based split -- every baseline from `models/baselines.py`
  plus the proposed model, loaded from its Days 7-9 checkpoint (never retrained here).
- **B (temporal contribution):** static-ML (RandomForest) / static-GCN / GAT+LSTM, each trained
  once on the Mon+Tue split and then only ever *scored* (never updated) across the full
  Mon->Fri sequence -- the `no_adapt` degradation baseline that C/D build on.
- **C (drift degradation):** reuses B's GAT+LSTM run; plots rolling F1-over-time with the
  Mon/Tue/Wed/Thu/Fri phase boundaries marked.
- **D (adaptation):** `no_adapt` (reused from B) vs `full_retrain` vs `incremental`, via
  `models.drift.run_drift_adaptation_loop`, across the full Mon->Fri sequence.
- **Ablation:** pure aggregation of rows already produced by A/D -- no new training.

A single window's binary label makes an *instantaneous* per-window F1 degenerate (0 or 1 only),
so B/C/D's per-window metrics are a trailing rolling aggregate (`ROLLING_WINDOW` windows) rather
than a literal single-window score -- a documented design choice IMPLEMENTATION.md §8 doesn't
fully specify.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from models import baselines as bl
from models import drift
from models.gat_lstm import TemporalGAT
from train import (
    CHECKPOINT_DIR,
    DEVICE,
    GRAPH_SEQUENCE_PATH,
    HIDDEN,
    K_LOOKBACK,
    LABELS_PATH,
    LSTM_HIDDEN,
    RESULTS_DIR,
    GraphWindowDataset,
    build_class_list,
    compute_metrics,
    evaluate,
    load_split_graphs,
)

PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "cicids2017_features.parquet"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.csv"

ROLLING_WINDOW = 20   # trailing windows aggregated into each per-window-position metric point

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_trained_model(checkpoint_path, in_dim: int, num_classes: int):
    """Load the Days 7-9 checkpoint produced by `python src/train.py`. Never trains here --
    every experiment below evaluates/adapts what train.py already produced."""
    model = TemporalGAT(in_dim=in_dim, hidden=HIDDEN, lstm_hidden=LSTM_HIDDEN,
                         num_classes=num_classes).to(DEVICE)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- run `python src/train.py` first to produce it "
            "(every experiment in eval.py depends on it)."
        )
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    log.info("Loaded %s (trained through epoch %d)", checkpoint_path, ckpt["epoch"])
    return model


def log_experiment_row(experiment: str, model: str, day: str, window_id, metrics: dict,
                        phase, path: Path = EXPERIMENT_LOG_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment": experiment, "model": model, "day": day, "window_id": window_id,
        "f1": metrics.get("f1"), "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"), "fpr": metrics.get("fpr"), "phase": phase,
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def rolling_window_metrics(records: list, window: int = ROLLING_WINDOW) -> list:
    """Turn a raw per-window (y_true, y_score) stream into a trailing-window rolling-metric
    stream. See module docstring for why this isn't a literal single-window F1."""
    out = []
    y_true_buf, y_score_buf = [], []
    for r in records:
        y_true_buf.append(r["y_true"])
        y_score_buf.append(r["y_score"])
        if len(y_true_buf) > window:
            y_true_buf.pop(0)
            y_score_buf.pop(0)
        m = compute_metrics(y_true_buf, y_score_buf)
        out.append({**r, **m})
    return out


def score_full_sequence_rf(rf_model, feat_cols: list) -> list:
    """Score every flow across the full Mon->Fri feature table, aggregated to window level via
    max flow-score (a window is flagged anomalous if any flow in it looks anomalous)."""
    features = pd.read_parquet(FEATURES_PATH)
    scores = rf_model.predict_proba(features[feat_cols].to_numpy())[:, 1]
    features = features.assign(_score=scores)
    per_window = (
        features.groupby("window_id")
        .agg(y_score=("_score", "max"), day=("day", "first"))
        .reset_index()
    )
    labels = pd.read_parquet(LABELS_PATH).set_index("window_id")["y_anomaly"]
    per_window["y_true"] = per_window["window_id"].map(labels)
    per_window = per_window.sort_values("window_id").reset_index(drop=True)
    return per_window[["window_id", "day", "y_true", "y_score"]].to_dict("records")


def score_full_sequence_static_gcn(model, graphs: list) -> list:
    """Score every window graph independently (no temporal state) across the full sequence."""
    model.eval()
    records = []
    with torch.no_grad():
        for g in graphs:
            score = model(g.x.to(DEVICE), g.edge_index.to(DEVICE)).item()
            records.append({
                "window_id": g.window_id, "day": g.day, "y_true": g.y.item(), "y_score": score,
            })
    return records


# ---------------------------------------------------------------------------
# Experiment A -- stationary performance
# ---------------------------------------------------------------------------

def experiment_a(model, class_to_idx: dict) -> pd.DataFrame:
    mon_tue_graphs = load_split_graphs(["mon", "tue"])
    n_train = int(len(mon_tue_graphs) * 0.8)
    test_graphs_with_lookback = mon_tue_graphs[max(n_train - K_LOOKBACK, 0):]
    test_ds = GraphWindowDataset(test_graphs_with_lookback, class_to_idx)
    gat_metrics = evaluate(model, test_ds)
    gat_metrics.update({"model": "GAT+LSTM", "scope": "window", "split": "mon_tue_test"})

    train_df, test_df, feat_cols, train_windows, test_windows, labels_df = bl.load_mon_tue_split()
    graphs_all = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
    graphs_by_window = {
        g.window_id: g for g in graphs_all
        if g.window_id in train_windows or g.window_id in test_windows
    }
    in_dim = len(feat_cols)

    rows = [
        bl.run_tabular_baseline("RandomForest", bl.fit_rf, train_df, test_df, feat_cols),
        bl.run_tabular_baseline("XGBoost", bl.fit_xgb, train_df, test_df, feat_cols),
        bl.run_tabular_baseline("MLP", bl.fit_mlp, train_df, test_df, feat_cols),
        bl.run_autoencoder_baseline(train_df, test_df, feat_cols),
        bl.run_sequence_lstm_baseline(train_df, test_df, feat_cols, train_windows,
                                       test_windows, labels_df),
        bl.run_static_graph_baseline("StaticGCN", bl.StaticGCN, graphs_by_window,
                                      train_windows, test_windows, in_dim),
        bl.run_static_graph_baseline("StaticGAT", bl.StaticGAT, graphs_by_window,
                                      train_windows, test_windows, in_dim),
        gat_metrics,
    ]
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "experiment_a.csv", index=False)
    log.info("Wrote experiment_a.csv:\n%s",
              out[["model", "scope", "f1", "pr_auc", "roc_auc", "fpr", "n"]])

    best = out.loc[out["f1"].idxmax()]
    if best["model"] != "GAT+LSTM":
        log.warning(
            "Proposed model is NOT the best on Experiment A (best=%s, f1=%.3f) -- per "
            "PLAN.md's done-criterion this is a bug to fix, not a result to report.",
            best["model"], best["f1"],
        )
    return out


# ---------------------------------------------------------------------------
# Experiment B -- temporal contribution (no adaptation, full Mon->Fri)
# ---------------------------------------------------------------------------

def experiment_b(gat_lstm_model, class_to_idx: dict, in_dim: int) -> dict:
    train_df, test_df, feat_cols, train_windows, test_windows, labels_df = bl.load_mon_tue_split()
    train_sub = bl._subsample(train_df, bl.TRAIN_SUBSAMPLE_MAX)
    rf_model = bl.fit_rf(train_sub[feat_cols].to_numpy(), train_sub["y_flow"].to_numpy())

    all_graphs = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
    all_graphs.sort(key=lambda d: d.window_id)

    train_graphs_gcn = [g for g in all_graphs if g.window_id in train_windows]
    gcn_model = bl.StaticGCN(in_dim=in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    gcn_model.train()
    for _epoch in range(15):
        for g in train_graphs_gcn:
            y = torch.tensor([[g.y.item()]], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = gcn_model(g.x.to(DEVICE), g.edge_index.to(DEVICE)).unsqueeze(0)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    records_by_model = {
        "RandomForest": score_full_sequence_rf(rf_model, feat_cols),
        "StaticGCN": score_full_sequence_static_gcn(gcn_model, all_graphs),
    }
    gat_records, _events = drift.run_drift_adaptation_loop(
        gat_lstm_model, all_graphs, class_to_idx, strategy="no_adapt"
    )
    records_by_model["GAT+LSTM"] = gat_records

    rows = []
    for model_name, records in records_by_model.items():
        for r in rolling_window_metrics(records):
            phase = drift.PHASE_MAP.get(r["day"])
            log_experiment_row("B", model_name, r["day"], r["window_id"], r, phase)
            rows.append({"experiment": "B", "model": model_name, "phase": phase, **r})

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "experiment_b.csv", index=False)
    log.info("Wrote experiment_b.csv (%d rows across %d models)", len(out), len(records_by_model))
    return records_by_model


# ---------------------------------------------------------------------------
# Experiment C -- drift degradation plot (reuses B's run)
# ---------------------------------------------------------------------------

def experiment_c(records_by_model: dict, all_graphs: list, model_name: str = "GAT+LSTM") -> Path:
    records = records_by_model[model_name]
    df = pd.DataFrame(rolling_window_metrics(records))
    boundaries = drift.phase_boundary_window_ids(all_graphs)
    first_day = all_graphs[0].day

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["window_id"], df["f1"], label=f"{model_name} F1 (rolling, w={ROLLING_WINDOW})")
    for day, window_id in boundaries.items():
        if day == first_day:
            continue  # sequence start, not a boundary
        ax.axvline(window_id, color="gray", linestyle="--", alpha=0.6)
        ax.text(window_id, 1.02, day, rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel("window_id (chronological)")
    ax.set_ylabel("F1 (rolling)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"Drift degradation -- {model_name}, no adaptation")
    ax.legend(loc="lower left")
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fig_drift_degradation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Experiment D -- adaptation comparison
# ---------------------------------------------------------------------------

def experiment_d(gat_lstm_model, class_to_idx: dict, all_graphs: list,
                  no_adapt_records: list = None):
    strategies = ["no_adapt", "full_retrain", "incremental"]
    all_rows, all_events = [], []

    for strategy in strategies:
        if strategy == "no_adapt" and no_adapt_records is not None:
            records, events = no_adapt_records, []
        else:
            records, events = drift.run_drift_adaptation_loop(
                gat_lstm_model, all_graphs, class_to_idx, strategy=strategy
            )

        model_label = f"GAT+LSTM_{strategy}"
        for r in rolling_window_metrics(records):
            phase = drift.PHASE_MAP.get(r["day"])
            log_experiment_row("D", model_label, r["day"], r["window_id"], r, phase)
            all_rows.append({"experiment": "D", "model": model_label, "phase": phase, **r})
        for e in events:
            all_events.append({"strategy": strategy, **e})
        log.info("Experiment D / %s: %d windows scored, %d adaptation events",
                  strategy, len(records), len(events))

    out = pd.DataFrame(all_rows)
    events_df = pd.DataFrame(all_events)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "experiment_d.csv", index=False)
    events_df.to_csv(RESULTS_DIR / "adaptation_events.csv", index=False)
    log.info("Wrote experiment_d.csv (%d rows) and adaptation_events.csv (%d events)",
              len(out), len(events_df))
    return out, events_df


# ---------------------------------------------------------------------------
# Ablation -- pure aggregation of A/D, no new training
# ---------------------------------------------------------------------------

def _ablation_row(label: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"model": label, "f1": float("nan"), "pr_auc": float("nan"),
                "roc_auc": float("nan"), "fpr": float("nan"), "n_evaluations": 0}
    agg = df[["f1", "pr_auc", "roc_auc", "fpr"]].mean(numeric_only=True)
    return {
        "model": label, "f1": agg["f1"], "pr_auc": agg["pr_auc"],
        "roc_auc": agg["roc_auc"], "fpr": agg["fpr"], "n_evaluations": len(df),
    }


def ablation(experiment_a_df: pd.DataFrame, experiment_d_df: pd.DataFrame) -> pd.DataFrame:
    """Model1 (flow-only MLP) / Model2 (static GCN) come from Experiment A's single held-out
    evaluation; Model3/4 (GAT+LSTM no-adapt / incremental) are averaged over every rolling
    per-window evaluation point across the full Mon->Fri run in Experiment D -- `n_evaluations`
    is therefore not directly comparable in scale across rows, only the metric columns are."""
    model1 = experiment_a_df[experiment_a_df["model"] == "MLP"]
    model2 = experiment_a_df[experiment_a_df["model"] == "StaticGCN"]
    model3 = experiment_d_df[experiment_d_df["model"] == "GAT+LSTM_no_adapt"]
    model4 = experiment_d_df[experiment_d_df["model"] == "GAT+LSTM_incremental"]

    out = pd.DataFrame([
        _ablation_row("Model1_flow_only_MLP", model1),
        _ablation_row("Model2_static_GCN", model2),
        _ablation_row("Model3_GAT_LSTM_no_adapt", model3),
        _ablation_row("Model4_GAT_LSTM_incremental", model4),
    ])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    log.info("Wrote ablation.csv:\n%s", out)
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "gat_lstm.pt"))
    parser.add_argument("--experiments", nargs="+", default=["a", "b", "c", "d", "ablation"],
                         choices=["a", "b", "c", "d", "ablation"])
    args = parser.parse_args()

    classes = build_class_list()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    probe_graphs = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
    in_dim = probe_graphs[0].x.shape[1]

    model = load_trained_model(args.checkpoint, in_dim, len(classes))

    a_df, d_df, records_by_model, all_graphs = None, None, None, None

    if "a" in args.experiments:
        a_df = experiment_a(model, class_to_idx)

    if {"b", "c", "d"} & set(args.experiments):
        all_graphs = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
        all_graphs.sort(key=lambda d: d.window_id)

    if "b" in args.experiments:
        records_by_model = experiment_b(model, class_to_idx, in_dim)

    if "c" in args.experiments:
        if records_by_model is None:
            records_by_model = experiment_b(model, class_to_idx, in_dim)
        experiment_c(records_by_model, all_graphs)

    if "d" in args.experiments:
        no_adapt_records = records_by_model["GAT+LSTM"] if records_by_model else None
        d_df, _events_df = experiment_d(model, class_to_idx, all_graphs, no_adapt_records)

    if "ablation" in args.experiments:
        if a_df is None or d_df is None:
            raise RuntimeError(
                "ablation needs both experiment_a and experiment_d output -- "
                "include 'a' and 'd' in --experiments"
            )
        ablation(a_df, d_df)
