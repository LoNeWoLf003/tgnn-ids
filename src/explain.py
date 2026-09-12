"""Attention-based explainability (Days 12-14, parallel) -- IMPLEMENTATION.md §9.

Reframed for the flow-graph redesign (PLAN.md's data-constraint note): attention highlights
which *neighboring flows* -- not hosts -- most influenced a given flow's contribution to its
window's anomaly signal ("primary contributing flow(s) / suspicious feature-similarity
relationships" rather than the original host-IP framing). Each node is labeled by its
Destination Port + truncated Label, the only identifying fields left in this dataset copy,
instead of an IP.

Depends on a trained checkpoint from `python src/train.py` -- run that first.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from models.gat_lstm import TemporalGAT
from train import (
    CHECKPOINT_DIR,
    DEVICE,
    GRAPH_SEQUENCE_PATH,
    HIDDEN,
    K_LOOKBACK,
    LSTM_HIDDEN,
    GraphWindowDataset,
    build_class_list,
    load_checkpoint,
)

RESULTS_DIR = Path("results")
TOP_K_EDGES = 10
SCORE_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def flow_node_labels(graph) -> list:
    """One label per node: f"port={port} label={label}" -- the only identifying fields left."""
    return [f"port={p} label={str(l)[:12]}" for p, l in zip(graph.dest_port, graph.flow_label)]


def plot_attention(graph, edge_index, alpha, flow_labels: list, top_k: int = TOP_K_EDGES,
                    out_path: Path = RESULTS_DIR / "fig_attention.png") -> Path:
    """`alpha`: (E, heads) last-layer GAT attention weights for `graph`'s edges."""
    alpha = alpha.mean(dim=1).detach().cpu().numpy()  # average over heads
    top_edges = alpha.argsort()[-top_k:]

    G = nx.DiGraph()
    for i in top_edges:
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(flow_labels[src], flow_labels[dst], weight=float(alpha[i]))

    pos = nx.spring_layout(G, seed=0)
    weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, ax=ax, with_labels=True, width=weights, node_size=800, font_size=7)
    ax.set_title(f"Top-{top_k} attention edges -- window {graph.window_id} ({graph.day})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def find_true_positive_windows(model, graphs: list, class_to_idx: dict, k: int = K_LOOKBACK,
                                score_threshold: float = SCORE_THRESHOLD, max_windows: int = 3):
    """Correctly-flagged anomaly windows (y_true=1, y_score>=threshold), highest-scoring
    first -- these are the windows worth visualizing attention for (§9's "pick 2-3 true-positive
    anomaly windows")."""
    ds = GraphWindowDataset(graphs, class_to_idx, k=k)
    model.eval()
    candidates = []
    with torch.no_grad():
        for i in range(len(ds)):
            window, y_anom, _ = ds[i]
            score, _logits = model(window)
            if y_anom.item() == 1.0 and score.item() >= score_threshold:
                candidates.append((score.item(), window))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[:max_windows]


def main(checkpoint_path, days=tuple(
             ["mon", "tue", "wed", "thu_am", "thu_pm", "fri_am", "fri_pm_scan", "fri_pm_ddos"]
         ), max_windows: int = 3, score_threshold: float = SCORE_THRESHOLD):
    classes = build_class_list()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_graphs = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
    graphs = [g for g in all_graphs if g.day in days]
    graphs.sort(key=lambda d: d.window_id)

    in_dim = graphs[0].x.shape[1]
    model = TemporalGAT(in_dim=in_dim, hidden=HIDDEN, lstm_hidden=LSTM_HIDDEN,
                         num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())  # unused after load; load_checkpoint needs one
    epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if epoch == 0:
        raise FileNotFoundError(f"{checkpoint_path} not found -- run `python src/train.py` first.")

    candidates = find_true_positive_windows(
        model, graphs, class_to_idx, max_windows=max_windows, score_threshold=score_threshold
    )
    if not candidates:
        log.warning(
            "No correctly-flagged anomaly windows found above threshold=%.2f -- nothing to "
            "visualize. Try --score-threshold lower, or check the trained model.",
            score_threshold,
        )
        return []

    out_paths = []
    for rank, (score, window) in enumerate(candidates):
        target = window[-1]
        with torch.no_grad():
            _score, _logits, (ei, alpha) = model(window, return_attention=True)
        labels = flow_node_labels(target)
        out_path = RESULTS_DIR / f"fig_attention_{rank}_w{target.window_id}.png"
        plot_attention(target, ei.cpu(), alpha.cpu(), labels, out_path=out_path)
        log.info("window %s (%s): score=%.3f majority_label=%s",
                  target.window_id, target.day, score, target.y_class)
        out_paths.append(out_path)
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "gat_lstm.pt"))
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args()
    main(args.checkpoint, max_windows=args.max_windows, score_threshold=args.score_threshold)
