"""Baseline detectors (Days 5-6) behind one common interface.

Every baseline exposes: fit(...) -> self, score(...) -> np.ndarray in [0,1], predict(...) -> {0,1}.
The exact shape of "data" passed to fit/score depends on the model family (documented per class):
  - tabular (RF/XGBoost/MLP) and the Autoencoder: flow-level, X = scaled CICFlowMeter features.
  - SequenceLSTM / StaticGCN / StaticGAT: window-level, one prediction per window.

Split: Experiment A's definition -- Mon+Tue only, **time-based** split (never random), first 80%
of windows (by window_id, which is already chronologically ordered) as train, last 20% as test.
Flow-level rows inherit their window's split assignment, so no flow-level leakage across the
train/test boundary.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch_geometric.nn import GATConv, GCNConv
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FEATURES_PATH = PROCESSED_DIR / "cicids2017_features.parquet"
LABELS_PATH = PROCESSED_DIR / "window_labels.parquet"
GRAPH_SEQUENCE_PATH = PROCESSED_DIR / "graph_sequence.pt"

TRAIN_SUBSAMPLE_MAX = 200_000  # cap flow-level training rows for RF/XGB/MLP/AE -- laptop-scale
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": auc(recall, precision),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "fpr": ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# Data loading + time-based split (Experiment A: Mon+Tue only)
# ---------------------------------------------------------------------------

def load_mon_tue_split(train_frac: float = 0.8):
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    mon_tue_windows = sorted(labels.loc[labels["day"].isin(["mon", "tue"]), "window_id"])
    n_train = int(len(mon_tue_windows) * train_frac)
    train_windows = set(mon_tue_windows[:n_train])
    test_windows = set(mon_tue_windows[n_train:])
    log.info("Mon+Tue split: %d train windows, %d test windows (time-based)",
             len(train_windows), len(test_windows))

    feat_cols = [c for c in features.columns if c.startswith("scaled_")]
    features = features[features["window_id"].isin(train_windows | test_windows)].copy()
    features["y_flow"] = (features["Label"] != "BENIGN").astype(int)

    train_df = features[features["window_id"].isin(train_windows)]
    test_df = features[features["window_id"].isin(test_windows)]
    return train_df, test_df, feat_cols, train_windows, test_windows, labels


def _subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=0)


# ---------------------------------------------------------------------------
# 1. Flow-level tabular baselines (Random Forest, XGBoost, MLP)
# ---------------------------------------------------------------------------

def fit_rf(X, y):
    m = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, class_weight="balanced",
                                random_state=0)
    m.fit(X, y)
    return m


def fit_xgb(X, y):
    pos = max((y == 1).sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist",
                       scale_pos_weight=(y == 0).sum() / pos, random_state=0,
                       eval_metric="logloss")
    m.fit(X, y)
    return m


def fit_mlp(X, y):
    m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50, early_stopping=True,
                       random_state=0)
    m.fit(X, y)
    return m


def run_tabular_baseline(name, fit_fn, train_df, test_df, feat_cols):
    train_sub = _subsample(train_df, TRAIN_SUBSAMPLE_MAX)
    X_train, y_train = train_sub[feat_cols].to_numpy(), train_sub["y_flow"].to_numpy()
    X_test, y_test = test_df[feat_cols].to_numpy(), test_df["y_flow"].to_numpy()

    log.info("Fitting %s on %d flows (test: %d flows)", name, len(X_train), len(X_test))
    model = fit_fn(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_score)
    metrics.update({"model": name, "scope": "flow", "split": "mon_tue_test"})
    log.info("%s: %s", name, metrics)
    return metrics


# ---------------------------------------------------------------------------
# 2. Autoencoder (unsupervised, trained on BENIGN-only flows)
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden=32, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder_baseline(train_df, test_df, feat_cols, epochs=15, batch_size=1024):
    benign_train = _subsample(train_df[train_df["y_flow"] == 0], TRAIN_SUBSAMPLE_MAX)
    X_train = torch.tensor(benign_train[feat_cols].to_numpy(dtype=np.float32), device=DEVICE)
    X_test = torch.tensor(test_df[feat_cols].to_numpy(dtype=np.float32), device=DEVICE)
    y_test = test_df["y_flow"].to_numpy()

    model = Autoencoder(in_dim=len(feat_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    log.info("Training Autoencoder on %d BENIGN flows", len(X_train))
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i:i + batch_size]]
            optimizer.zero_grad()
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        log.info("AE epoch %d loss %.4f", epoch, total_loss / len(X_train))

    model.eval()
    with torch.no_grad():
        recon = model(X_test)
        recon_error = ((recon - X_test) ** 2).mean(dim=1).cpu().numpy()
    # normalize to [0,1] via min-max over the test scores for a usable "score"
    score = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-12)
    metrics = compute_metrics(y_test, score)
    metrics.update({"model": "Autoencoder", "scope": "flow", "split": "mon_tue_test"})
    log.info("Autoencoder: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 3. Sequence baseline (LSTM over each window's ordered flows)
# ---------------------------------------------------------------------------

class SequenceLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq):
        # seq: (1, T, in_dim)
        z, _ = self.lstm(seq)
        return torch.sigmoid(self.head(z[:, -1, :]))


def _window_sequences(df: pd.DataFrame, feat_cols, window_ids):
    seqs = {}
    for window_id, g in df[df["window_id"].isin(window_ids)].groupby("window_id"):
        seqs[window_id] = torch.tensor(g.sort_values("seq")[feat_cols].to_numpy(dtype=np.float32))
    return seqs


def run_sequence_lstm_baseline(train_df, test_df, feat_cols, train_windows, test_windows,
                                labels_df, epochs=5):
    train_seqs = _window_sequences(train_df, feat_cols, train_windows)
    test_seqs = _window_sequences(test_df, feat_cols, test_windows)
    y_by_window = labels_df.set_index("window_id")["y_anomaly"]

    model = SequenceLSTM(in_dim=len(feat_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    log.info("Training SequenceLSTM on %d windows", len(train_seqs))
    model.train()
    train_items = list(train_seqs.items())
    for epoch in range(epochs):
        total_loss = 0.0
        for window_id, seq in train_items:
            y = torch.tensor([[y_by_window[window_id]]], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = model(seq.unsqueeze(0).to(DEVICE))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.info("SequenceLSTM epoch %d loss %.4f", epoch, total_loss / max(len(train_items), 1))

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for window_id, seq in test_seqs.items():
            pred = model(seq.unsqueeze(0).to(DEVICE)).item()
            y_true.append(y_by_window[window_id])
            y_score.append(pred)

    metrics = compute_metrics(np.array(y_true), np.array(y_score))
    metrics.update({"model": "SequenceLSTM", "scope": "window", "split": "mon_tue_test"})
    log.info("SequenceLSTM: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 4. Static graph baselines (GCN, GAT -- no temporal component)
# ---------------------------------------------------------------------------

class StaticGCN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        graph_emb = h.mean(dim=0)
        return torch.sigmoid(self.head(graph_emb))


class StaticGAT(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        graph_emb = h.mean(dim=0)
        return torch.sigmoid(self.head(graph_emb))


def run_static_graph_baseline(name, model_cls, graphs_by_window, train_windows, test_windows,
                               in_dim, epochs=15):
    train_graphs = [graphs_by_window[w] for w in sorted(train_windows) if w in graphs_by_window]
    test_graphs = [graphs_by_window[w] for w in sorted(test_windows) if w in graphs_by_window]

    model = model_cls(in_dim=in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    log.info("Training %s on %d graphs", name, len(train_graphs))
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for g in train_graphs:
            y = torch.tensor([[g.y.item()]], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            pred = model(g.x.to(DEVICE), g.edge_index.to(DEVICE)).unsqueeze(0)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.info("%s epoch %d loss %.4f", name, epoch, total_loss / max(len(train_graphs), 1))

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for g in test_graphs:
            pred = model(g.x.to(DEVICE), g.edge_index.to(DEVICE)).item()
            y_true.append(g.y.item())
            y_score.append(pred)

    metrics = compute_metrics(np.array(y_true), np.array(y_score))
    metrics.update({"model": name, "scope": "window", "split": "mon_tue_test"})
    log.info("%s: %s", name, metrics)
    return metrics


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df, test_df, feat_cols, train_windows, test_windows, labels_df = load_mon_tue_split()

    graphs = torch.load(GRAPH_SEQUENCE_PATH, weights_only=False)
    graphs_by_window = {g.window_id: g for g in graphs
                        if g.window_id in train_windows or g.window_id in test_windows}
    in_dim = len(feat_cols)

    results = []
    results.append(run_tabular_baseline("RandomForest", fit_rf, train_df, test_df, feat_cols))
    results.append(run_tabular_baseline("XGBoost", fit_xgb, train_df, test_df, feat_cols))
    results.append(run_tabular_baseline("MLP", fit_mlp, train_df, test_df, feat_cols))
    results.append(run_autoencoder_baseline(train_df, test_df, feat_cols))
    results.append(run_sequence_lstm_baseline(train_df, test_df, feat_cols,
                                               train_windows, test_windows, labels_df))
    results.append(run_static_graph_baseline("StaticGCN", StaticGCN, graphs_by_window,
                                              train_windows, test_windows, in_dim))
    results.append(run_static_graph_baseline("StaticGAT", StaticGAT, graphs_by_window,
                                              train_windows, test_windows, in_dim))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baselines.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    print(pd.DataFrame(results)[["model", "scope", "f1", "pr_auc", "roc_auc", "fpr", "n"]])
