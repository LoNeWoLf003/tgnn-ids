"""Build one torch_geometric.data.Data graph per window.

Data constraint (see PLAN.md / ingest.py / features.py): no host identity exists in this dataset
copy, so nodes are **flows** (not hosts) and edges are a **k-NN similarity graph** in scaled
CICFlowMeter feature space, built independently within each window. Node count varies per window
(all flows assigned to that window_id) -- no zero-padding needed, PyG handles variable-sized
graphs natively.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "cicids2017_features.parquet"
LABELS_PATH = PROCESSED_DIR / "window_labels.parquet"
GRAPH_SEQUENCE_PATH = PROCESSED_DIR / "graph_sequence.pt"

K_NEIGHBORS = 8  # hyperparameter -- documented, not exhaustively tuned per PLAN.md

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _scaled_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("scaled_")]
    if not cols:
        raise ValueError("No scaled_* columns found -- run features.py first")
    return cols


def build_window_graph(g: pd.DataFrame, feature_cols: list[str], k: int = K_NEIGHBORS) -> Data:
    g = g.reset_index(drop=True)
    x = torch.tensor(g[feature_cols].to_numpy(dtype=np.float32))

    n = len(g)
    k_eff = min(k, n - 1) if n > 1 else 0
    if k_eff > 0:
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(x.numpy())  # +1: includes self
        _, idx = nn.kneighbors(x.numpy())
        src = np.repeat(np.arange(n), k_eff)
        dst = idx[:, 1:].reshape(-1)  # drop self-neighbor (col 0)
        # symmetrize: undirected k-NN graph
        edge_index = torch.tensor(
            np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1),
            dtype=torch.long,
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # single-flow window: no edges

    y_anomaly = torch.tensor(float((g["Label"] != "BENIGN").any()))
    y_class = g["Label"].mode().iloc[0]
    window_id = int(g["window_id"].iloc[0])
    day = g["day"].iloc[0]
    dest_ports = g["Destination Port"].to_numpy()
    labels = g["Label"].to_numpy()

    return Data(
        x=x, edge_index=edge_index,
        y=y_anomaly, y_class=y_class, window_id=window_id, day=day,
        dest_port=dest_ports, flow_label=labels,
    )


def build_graph_sequence(
    features_path: Path = FEATURES_PATH, k: int = K_NEIGHBORS
) -> list[Data]:
    df = pd.read_parquet(features_path)
    feature_cols = _scaled_feature_cols(df)

    graphs = []
    for window_id, g in df.groupby("window_id"):
        graphs.append(build_window_graph(g, feature_cols, k=k))

    graphs.sort(key=lambda d: d.window_id)
    log.info("Built %d window graphs (k=%d neighbors, %d feature dims)",
              len(graphs), k, len(feature_cols))
    return graphs


if __name__ == "__main__":
    graphs = build_graph_sequence()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, GRAPH_SEQUENCE_PATH)
    log.info("Saved %s (%d graphs)", GRAPH_SEQUENCE_PATH, len(graphs))

    # Quick spot-checks
    assert len(graphs) > 0
    assert all(g.x.shape[0] >= 1 for g in graphs)
    for g in graphs[:3] + graphs[-3:]:
        n_edges = g.edge_index.shape[1]
        max_node_ref = g.edge_index.max().item() if n_edges > 0 else -1
        print(f"window_id={g.window_id} day={g.day} nodes={g.x.shape[0]} edges={n_edges} "
              f"max_node_ref={max_node_ref} y={g.y.item()} y_class={g.y_class}")
        assert max_node_ref < g.x.shape[0], "edge_index references a node out of range"
