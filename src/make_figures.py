"""Regenerate the paper's result figures (larger, readable) from saved results.

    .venv\\Scripts\\python.exe src\\make_figures.py

Writes paper/figures/fig_drift_degradation.png and fig_stream_all.png. The attention figures are
produced by src/explain.py (see plot_attention there).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
MULTI = ROOT / "results" / "multi_v2"

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})

# colour-blind-safe (Okabe-Ito) palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "grey": "#555555"}


def drift_figure():
    d = pd.read_csv(ROOT / "results" / "experiment_log.csv")
    g = d[(d.experiment == "B") & (d.model == "GAT+LSTM")].sort_values("window_id")
    starts = g.groupby("day", sort=False).window_id.min()
    ends = g.groupby("day", sort=False).window_id.max()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bounds = list(starts.values) + [g.window_id.max()]
    for i, (day, s) in enumerate(starts.items()):
        e = bounds[i + 1]
        if i % 2 == 0:
            ax.axvspan(s, e, color="#000000", alpha=0.06, lw=0)
        ax.text((s + e) / 2, 1.04, day.replace("_", " ").replace("fri pm ", "Fri "),
                ha="center", va="bottom", fontsize=11, rotation=30)
    ax.plot(g.window_id, g.f1, color=C["blue"], lw=2.2, label="GAT+LSTM, no adaptation")
    ax.set_xlabel("window index (chronological, Mon $\\rightarrow$ Fri)")
    ax.set_ylabel("rolling F1 (20 windows)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(g.window_id.min(), g.window_id.max())
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center", frameon=False, bbox_to_anchor=(0.55, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_drift_degradation.png", dpi=200)
    plt.close(fig)


def stream_figure():
    sl = pd.concat([pd.read_csv(MULTI / f"stream_log_{d}.csv") for d in
                    ["nsl-kdd", "unsw-nb15", "ton-iot"]])
    ev = pd.read_csv(MULTI / "adaptation_events.csv")
    series = [("XGBoost", "static", "XGBoost, unadapted", C["grey"], "--"),
              ("GAT+LSTM", "no_adapt", "GAT+LSTM, unadapted", C["blue"], "-"),
              ("GAT+LSTM", "incremental", "GAT+LSTM, incremental", C["green"], "-"),
              ("GAT+LSTM", "full_retrain", "GAT+LSTM, full retrain", C["red"], "-"),
              ("RandomForest", "replay", "Random Forest, adapted (replay)", C["orange"], "-")]
    names = {"nsl-kdd": "NSL-KDD", "unsw-nb15": "UNSW-NB15", "ton-iot": "TON_IoT"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    handles = {}
    for k, (ax, ds) in enumerate(zip(axes.flat, names)):
        d = sl[sl.dataset == ds]
        lo = 1.0
        for model, strat, label, col, ls in series:
            g = d[(d.model == model) & (d.strategy == strat)].groupby("stream_idx")[
                ["tp", "fp", "fn"]].sum()
            roll = g.rolling(10, min_periods=3).sum()
            f1 = 2 * roll.tp / (2 * roll.tp + roll.fp + roll.fn).clip(lower=1)
            lo = min(lo, np.nanmin(f1.values))
            (h,) = ax.plot(f1.index, f1.values, color=col, ls=ls, lw=2.3, label=label)
            handles[label] = h
        ph = d[(d.model == "XGBoost") & (d.strategy == "static")].groupby("stream_idx").phase.first()
        for b in np.where(np.diff(ph.values) != 0)[0] + 1:
            ax.axvline(b - 0.5, color="k", lw=1.0, ls=":")
        e = ev[(ev.dataset == ds) & (ev.seed == 0) & (ev.model == "GAT+LSTM")
               & (ev.strategy == "full_retrain")]
        ylo = max(0.0, lo - 0.08)
        ax.set_ylim(ylo, 1.01)
        if len(e):
            m = ax.scatter(e.stream_idx, np.full(len(e), ylo + 0.012), marker="^", s=90,
                           color=C["red"], edgecolor="k", lw=0.6, zorder=5)
            handles["GAT+LSTM full-retrain event (seed 0)"] = m
        ax.set_title(f"({chr(97 + k)}) {names[ds]}")
        ax.set_xlabel("stream window index")
        ax.set_ylabel("rolling F1 (10 windows)")
        ax.grid(alpha=0.25)
    leg = axes.flat[3]
    leg.axis("off")
    h, l = list(handles.values()), list(handles.keys())
    leg.legend(h, l, loc="center", frameon=False, fontsize=13.5, title="Dotted vertical lines: phase boundaries\nRolling F1 pooled over 3 seeds",
               title_fontsize=12.5, alignment="left")
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.savefig(OUT / "fig_stream_all.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    drift_figure()
    stream_figure()
    print("wrote figures to", OUT)
