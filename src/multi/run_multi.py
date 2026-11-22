"""Run the experiment suite on NSL-KDD, UNSW-NB15 and TON-IoT.

    python src/multi/run_multi.py --datasets nsl-kdd unsw-nb15 ton-iot --seeds 3

Per dataset / seed:
  1. Split phase-0 windows 80/10/10 into train / val (early stopping) / stationary hold-out.
  2. Flow-level baselines (RandomForest, XGBoost, sklearn-MLP, Autoencoder) trained on the train
     windows' flows.
  3. Window-graph models (MLP-node, GCN, GAT, GAT+LSTM) trained on the same windows.
  4. Every model is scored over the chronological stream (phases 1..) with no adaptation;
     GAT+LSTM is additionally run with ADWIN-triggered `incremental` and `full_retrain`.
Outputs (results/multi/): metrics.csv (long), summary.csv (mean/std over seeds), stream_log.csv
(per-window confusion counts), adaptation_events.csv, fig_stream_<dataset>.png.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import ids_data as D  # noqa: E402
import ids_nets as N  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "multi"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("multi")


def metrics(y, score, pred=None):
    y = np.asarray(y).astype(int)
    pred = (score >= 0.5).astype(int) if pred is None else pred.astype(int)
    one_class = len(np.unique(y)) < 2
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": float("nan") if one_class else roc_auc_score(y, score),
        "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "n": len(y),
    }


class AE(nn.Module):
    def __init__(self, d, h=32, b=8):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, b))
        self.d = nn.Sequential(nn.Linear(b, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.d(self.e(x))


def flat(store, idx):
    idx = np.asarray(idx)
    return (store.X[idx].reshape(-1, store.X.shape[-1]).cpu().numpy(),
            store.y[idx].reshape(-1).cpu().numpy().astype(int))


def eval_sets(store, stationary_idx, stream_idx):
    """name -> positions (into the respective index array)."""
    sets = {"stationary": ("stat", np.arange(len(stationary_idx)))}
    sp = store.phase[np.asarray(stream_idx)]
    for p in sorted(set(sp)):
        sets[f"phase{p}"] = ("stream", np.where(sp == p)[0])
    sets["stream_all"] = ("stream", np.arange(len(stream_idx)))
    return sets


def record(rows, slog, ds, seed, model, strat, store, prob_stat, prob_stream, y_stat, y_stream,
           stream_idx, pred_stat=None, pred_stream=None):
    """prob_*: (n_flows,) arrays flattened window-major. Appends metric rows + stream counts."""
    W = store.X.shape[1]
    stationary_idx = np.arange(len(y_stat) // W)
    sp = store.phase[np.asarray(stream_idx)]
    sets = {"stationary": (y_stat, prob_stat, pred_stat)}
    for p in sorted(set(sp)):
        m = np.repeat(sp == p, W)
        sets[f"phase{p}"] = (y_stream[m], prob_stream[m], None if pred_stream is None else pred_stream[m])
    sets["stream_all"] = (y_stream, prob_stream, pred_stream)
    for name, (y, s, pr) in sets.items():
        rows.append({"dataset": ds, "seed": seed, "model": model, "strategy": strat,
                     "eval_set": name, **metrics(y, s, pr)})
    pred = (prob_stream >= 0.5) if pred_stream is None else pred_stream.astype(bool)
    yb = y_stream.astype(bool)
    pw, yw = pred.reshape(-1, W), yb.reshape(-1, W)
    for j in range(len(pw)):
        slog.append({"dataset": ds, "seed": seed, "model": model, "strategy": strat, "stream_idx": j,
                     "phase": int(sp[j]), "tp": int((pw[j] & yw[j]).sum()),
                     "fp": int((pw[j] & ~yw[j]).sum()), "fn": int((~pw[j] & yw[j]).sum()),
                     "tn": int((~pw[j] & ~yw[j]).sum())})


def adaptive_flow_stream(make_fn, Xtr, ytr, Xval_w, Xsm, ysm, W, mode, seed, delta=0.002,
                         cooldown=10, recent_span=50, cap=200_000):
    """Drift-triggered flow-level baseline for a like-for-like comparison with the adaptive
    GAT+LSTM. Same protocol as `ids_nets.adaptive_stream`: every window is scored BEFORE any update
    on it, labels of scored windows arrive afterwards, ADWIN (same delta / cooldown / z-score->[0,1]
    squash) watches the window's mean binary predictive entropy, warm-started on held-out validation
    windows (training windows are unusable: trees fit them almost perfectly).
    mode 'full_retrain': refit on train + every stream window seen; 'replay': refit on the last
    `recent_span` windows + an equal-size random sample of training windows.
    Returns (probs (n_flows,), events)."""
    from river.drift import ADWIN
    rng = np.random.default_rng(seed)

    def fit_on(X, y):
        if len(X) > cap:
            sel = rng.choice(len(X), cap, replace=False)
            X, y = X[sel], y[sel]
        m = make_fn()
        m.fit(X, y)
        return m

    def window_ent(m, X):
        p = np.clip(m.predict_proba(X)[:, 1].astype(np.float64), 1e-9, 1 - 1e-9)
        return p, float((-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean())

    model = fit_on(Xtr, ytr)
    v = np.array([window_ent(model, w)[1] for w in Xval_w])
    mu, sd = v.mean(), v.std() + 1e-3
    squash = lambda e: float(np.clip(0.5 + (e - mu) / (8 * sd), 0.0, 1.0))
    adwin = ADWIN(delta=delta)
    for e in v:
        adwin.update(squash(e))
    nW = len(Xsm) // W
    probs, events, last_fire = [], [], -10**9
    ntr_w = len(Xtr) // W
    for j in range(nW):
        sl = slice(j * W, (j + 1) * W)
        p, e = window_ent(model, Xsm[sl])
        probs.append(p)
        adwin.update(squash(e))
        if not adwin.drift_detected or j - last_fire < cooldown:
            continue
        last_fire = j
        t0 = time.perf_counter()
        if mode == "full_retrain":
            X = np.concatenate([Xtr, Xsm[: (j + 1) * W]]); y = np.concatenate([ytr, ysm[: (j + 1) * W]])
        else:
            lo = max(0, j - recent_span + 1)
            old_w = rng.choice(ntr_w, min(j + 1 - lo, ntr_w), replace=False)
            Xo = np.concatenate([Xtr[w * W:(w + 1) * W] for w in old_w])
            yo = np.concatenate([ytr[w * W:(w + 1) * W] for w in old_w])
            X = np.concatenate([Xsm[lo * W:(j + 1) * W], Xo]); y = np.concatenate([ysm[lo * W:(j + 1) * W], yo])
        model = fit_on(X, y)
        events.append({"stream_idx": j, "n_flows": len(X), "wall_clock_s": time.perf_counter() - t0})
    return np.concatenate(probs), events


def run_dataset(ds, seed, args, rows, slog, events_out):
    t_ds = time.time()
    prep = D.prepare(D.load_spec(ds), window=args.window)
    store = N.build_store(prep, k=args.knn)
    K = args.K
    p0 = np.where(store.phase == 0)[0]
    n = len(p0)
    train_t, val_t, stat_t = p0[: int(.8 * n)], p0[int(.8 * n): int(.9 * n)], p0[int(.9 * n):]
    stream_t = np.where(store.phase >= 1)[0]
    log.info("[%s seed %d] windows: train %d val %d stationary %d stream %d (W=%d, d=%d)", ds, seed,
             len(train_t), len(val_t), len(stat_t), len(stream_t), args.window, prep.n_features)

    Xtr, ytr = flat(store, train_t)
    Xst, yst = flat(store, stat_t)
    Xsm, ysm = flat(store, stream_t)

    # ---- flow-level baselines
    Xval, _ = flat(store, val_t)
    Xval_w = list(Xval.reshape(-1, args.window, Xval.shape[-1]))

    def sk(name, model, make_fn):
        t0 = time.time()
        if len(Xtr) > 200_000:
            sel = np.random.default_rng(seed).choice(len(Xtr), 200_000, replace=False)
            model.fit(Xtr[sel], ytr[sel])
        else:
            model.fit(Xtr, ytr)
        record(rows, slog, ds, seed, name, "static", store, model.predict_proba(Xst)[:, 1],
               model.predict_proba(Xsm)[:, 1], yst, ysm, stream_t)
        log.info("  %s done (%.0fs)", name, time.time() - t0)
        for mode in ["full_retrain", "replay"]:
            t1 = time.time()
            pr, ev = adaptive_flow_stream(make_fn, Xtr, ytr, Xval_w, Xsm, ysm, args.window, mode, seed)
            record(rows, slog, ds, seed, name, mode, store, model.predict_proba(Xst)[:, 1], pr,
                   yst, ysm, stream_t)
            for e in ev:
                events_out.append({"dataset": ds, "seed": seed, "strategy": mode, "model": name, **e})
            log.info("  %s[%s]: %d adaptation events, %.0fs", name, mode, len(ev), time.time() - t1)

    mk_rf = lambda: RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1,
                                           class_weight="balanced", random_state=seed)
    mk_xgb = lambda: XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist",
                                   random_state=seed, eval_metric="logloss")
    sk("RandomForest", mk_rf(), mk_rf)
    sk("XGBoost", mk_xgb(), mk_xgb)
    sk_mlp = MLPClassifier((64,), max_iter=60, early_stopping=True, random_state=seed)
    sk("MLP(sklearn)", sk_mlp, lambda: MLPClassifier((64,), max_iter=60, early_stopping=True,
                                                     random_state=seed))

    torch.manual_seed(seed)
    ae = AE(Xtr.shape[1]).to(N.DEVICE)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    ben = torch.tensor(Xtr[ytr == 0], device=N.DEVICE)
    for _ in range(20):
        perm = torch.randperm(len(ben), device=N.DEVICE)
        for i in range(0, len(ben), 1024):
            b = ben[perm[i:i + 1024]]
            loss = ((ae(b) - b) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()

    def ae_err(X):
        with torch.no_grad():
            x = torch.tensor(X, device=N.DEVICE)
            return ((ae(x) - x) ** 2).mean(1).cpu().numpy()
    thr = np.percentile(ae_err(Xtr[ytr == 0]), 95)  # threshold set on training benign only
    e_st, e_sm = ae_err(Xst), ae_err(Xsm)
    mx = max(e_st.max(), e_sm.max())
    record(rows, slog, ds, seed, "Autoencoder", "static", store, e_st / mx, e_sm / mx, yst, ysm,
           stream_t, pred_stat=e_st > thr, pred_stream=e_sm > thr)

    # ---- window-graph models
    base_gl = None
    for variant in ["MLP", "GCN", "GAT", "GAT+LSTM"]:
        t0 = time.time()
        torch.manual_seed(seed)
        model = N.make_model(variant, prep.n_features)
        N.fit(model, store, train_t, args.epochs, 1e-3, K, val_targets=val_t, seed=seed)
        tr_time = time.time() - t0
        ps, _ = N.predict(model, store, stat_t, K)
        pm, _ = N.predict(model, store, stream_t, K)
        record(rows, slog, ds, seed, variant, "no_adapt", store, ps.reshape(-1).cpu().numpy(),
               pm.reshape(-1).cpu().numpy(), yst, ysm, stream_t)
        log.info("  %s trained+scored (%.0fs, train %.0fs)", variant, time.time() - t0, tr_time)
        if variant == "GAT+LSTM":
            base_gl = model

    for strat in ["incremental", "full_retrain", "incr_replay", "hybrid"]:
        t0 = time.time()
        pm, ev = N.adaptive_stream(base_gl, "GAT+LSTM", store, train_t, stream_t, strat, K, seed)
        ps, _ = N.predict(base_gl, store, stat_t, K)  # stationary hold-out: pre-adaptation model
        record(rows, slog, ds, seed, "GAT+LSTM", strat, store, ps.reshape(-1).cpu().numpy(),
               pm.reshape(-1).cpu().numpy(), yst, ysm, stream_t)
        for e in ev:
            events_out.append({"dataset": ds, "seed": seed, "strategy": strat, **e})
        log.info("  GAT+LSTM[%s]: %d adaptation events, %.0fs", strat, len(ev), time.time() - t0)
    log.info("[%s seed %d] total %.0fs", ds, seed, time.time() - t_ds)


def plot_stream(ds, slog_df, events_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [("XGBoost", "static", "#7f7f7f", "--"), ("GAT", "no_adapt", "#1f77b4", "-"),
           ("GAT+LSTM", "no_adapt", "#ff7f0e", "-"), ("GAT+LSTM", "incremental", "#2ca02c", "-"),
           ("GAT+LSTM", "full_retrain", "#d62728", "-")]
    d = slog_df[slog_df.dataset == ds]
    fig, ax = plt.subplots(figsize=(10, 4))
    for model, strat, col, ls in sel:
        g = d[(d.model == model) & (d.strategy == strat)].groupby("stream_idx")[
            ["tp", "fp", "fn"]].sum()  # pooled over seeds
        if g.empty:
            continue
        roll = g.rolling(10, min_periods=3).sum()
        f1 = 2 * roll.tp / (2 * roll.tp + roll.fp + roll.fn).clip(lower=1)
        ax.plot(f1.index, f1.values, color=col, ls=ls, lw=1.6, label=f"{model} [{strat}]")
    ph = d[d.model == "XGBoost"].groupby("stream_idx").phase.first()
    for b in np.where(np.diff(ph.values) != 0)[0] + 1:
        ax.axvline(b - 0.5, color="k", lw=0.8, ls=":")
    ev = events_df[(events_df.dataset == ds) & (events_df.seed == events_df.seed.min())] \
        if len(events_df) else events_df
    for s, c in (("incremental", "#2ca02c"), ("full_retrain", "#d62728")):
        e = ev[(ev.strategy == s) & (ev.model == "GAT+LSTM")] if len(ev) else ev
        if len(e):
            ax.scatter(e.stream_idx, np.full(len(e), 0.02), marker="^", color=c, s=28, zorder=5)
    ax.set_xlabel("stream window index (dotted = phase boundary, triangles = adaptation, seed 0)")
    ax.set_ylabel("rolling F1 (10 windows)"); ax.set_ylim(0, 1.02); ax.set_title(ds)
    ax.legend(fontsize=8, loc="lower right", ncol=2); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir / f"fig_stream_{ds}.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=D.DATASETS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--K", type=int, default=6, help="trailing windows fed to the LSTM")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        rows, slog, events = [], [], []
        for seed in range(args.seeds):
            run_dataset(ds, seed, args, rows, slog, events)
            pd.DataFrame(rows).to_csv(out / f"metrics_{ds}.csv", index=False)
            pd.DataFrame(slog).to_csv(out / f"stream_log_{ds}.csv", index=False)
            pd.DataFrame(events).to_csv(out / f"adaptation_events_{ds}.csv", index=False)

    parts = lambda pre: [pd.read_csv(out / f"{pre}_{d}.csv") for d in args.datasets
                         if (out / f"{pre}_{d}.csv").exists() and (out / f"{pre}_{d}.csv").stat().st_size > 5]
    m = pd.concat(parts("metrics"))
    m.to_csv(out / "metrics.csv", index=False)
    sl = pd.concat(parts("stream_log")); ev = pd.concat(parts("adaptation_events")) if parts("adaptation_events") else pd.DataFrame()
    ev.to_csv(out / "adaptation_events.csv", index=False)
    g = m.groupby(["dataset", "model", "strategy", "eval_set"])[["f1", "pr_auc", "roc_auc", "fpr"]]
    summ = g.mean().join(g.std(), rsuffix="_std").reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    for ds in args.datasets:
        plot_stream(ds, sl, ev, out)
    log.info("wrote results to %s", out)


if __name__ == "__main__":
    main()
