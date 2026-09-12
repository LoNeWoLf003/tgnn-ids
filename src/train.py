"""Resumable, checkpointed training loop for TemporalGAT (Day 9).

Default scope: Mon+Tue only, time-based split (matches Experiment A / baselines.py) -- this is
the "does the temporal model at least beat the static-GNN baseline on stationary data" checkpoint
per PLAN.md. Later days (Experiments B/C/D) reuse this same loop over Mon->Fri with the
adaptation strategies in drift.py.

Checkpoints every epoch (model state, optimizer state, epoch, RNG state) to
checkpoints/gat_lstm.pt -- a killed/interrupted run resumes with `--resume` and continues the
same loss curve, not a fresh one.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc as sk_auc, f1_score, roc_auc_score

from models.gat_lstm import TemporalGAT

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
GRAPH_SEQUENCE_PATH = PROCESSED_DIR / "graph_sequence.pt"
LABELS_PATH = PROCESSED_DIR / "window_labels.parquet"
CLASS_LIST_PATH = PROCESSED_DIR / "class_list.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_LOOKBACK = 6
HIDDEN = 64
LSTM_HIDDEN = 64
LR = 1e-3
DEFAULT_EPOCHS = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_class_list() -> list[str]:
    if CLASS_LIST_PATH.exists():
        return json.loads(CLASS_LIST_PATH.read_text())
    labels_df = pd.read_parquet(LABELS_PATH)
    classes = sorted(labels_df["y_class"].unique())
    CLASS_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_LIST_PATH.write_text(json.dumps(classes))
    return classes


def load_split_graphs(days, path=GRAPH_SEQUENCE_PATH):
    graphs = torch.load(path, weights_only=False)
    graphs = [g for g in graphs if g.day in days]
    graphs.sort(key=lambda d: d.window_id)  # positional adjacency == chronological adjacency
    return graphs


class GraphWindowDataset(torch.utils.data.Dataset):
    """Yields (graph_window, y_anomaly, y_class_idx) for each valid lookback position."""

    def __init__(self, graphs, class_to_idx, k=K_LOOKBACK):
        self.graphs = graphs
        self.k = k
        self.class_to_idx = class_to_idx

    def __len__(self):
        return max(len(self.graphs) - self.k, 0)

    def __getitem__(self, idx):
        i = idx + self.k
        window = self.graphs[i - self.k: i + 1]
        y_anom = torch.tensor([[window[-1].y.item()]], dtype=torch.float32)
        y_cls = torch.tensor([self.class_to_idx[window[-1].y_class]], dtype=torch.long)
        return window, y_anom, y_cls


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": sk_auc(recall, precision),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "fpr": ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1),
        "n": len(y_true),
    }


def load_checkpoint(model, optimizer, checkpoint_path):
    if not Path(checkpoint_path).exists():
        return 0
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    torch.set_rng_state(ckpt["rng_state"])
    log.info("Resumed from %s at epoch %d", checkpoint_path, ckpt["epoch"])
    return ckpt["epoch"] + 1


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }, checkpoint_path)


def evaluate(model, dataset):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            window, y_anom, _ = dataset[i]
            score, _ = model(window)
            y_true.append(y_anom.item())
            y_score.append(score.item())
    return compute_metrics(y_true, y_score)


def train(model, train_ds, val_ds, optimizer, checkpoint_path, start_epoch=0,
          num_epochs=DEFAULT_EPOCHS, log_path=RESULTS_DIR / "train_log.csv",
          cls_loss_weight=0.5):
    criterion_anom = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(len(train_ds)).tolist()
        for idx in perm:
            window, y_anom, y_cls = train_ds[idx]
            y_anom, y_cls = y_anom.to(DEVICE), y_cls.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                score, logits = model(window)
            # BCELoss/CrossEntropyLoss are unsafe under autocast (the guide's sigmoid-then-BCE
            # combo needs full precision) -- cast back to fp32 and compute the loss outside the
            # autocast context, per PyTorch's own autocast-op-list guidance.
            loss = (criterion_anom(score.float(), y_anom)
                    + cls_loss_weight * criterion_cls(logits.float(), y_cls))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_ds) if len(val_ds) else {}
        save_checkpoint(model, optimizer, epoch, checkpoint_path)  # every epoch, no exceptions
        row = {"epoch": epoch, "train_loss": total_loss / max(len(train_ds), 1), **val_metrics}
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not Path(log_path).exists(),
                                    index=False)
        log.info("epoch %d loss %.4f val=%s -- checkpoint saved", epoch, row["train_loss"],
                  val_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "gat_lstm.pt"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--days", nargs="+", default=["mon", "tue"])
    args = parser.parse_args()

    classes = build_class_list()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    graphs = load_split_graphs(args.days)
    n_train = int(len(graphs) * 0.8)
    train_graphs, val_graphs = graphs[:n_train], graphs[n_train:]
    # keep the k lookback continuous across the split boundary
    val_graphs_with_lookback = graphs[max(n_train - K_LOOKBACK, 0):]

    in_dim = graphs[0].x.shape[1]
    model = TemporalGAT(in_dim=in_dim, hidden=HIDDEN, lstm_hidden=LSTM_HIDDEN,
                         num_classes=len(classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = GraphWindowDataset(train_graphs, class_to_idx)
    val_ds = GraphWindowDataset(val_graphs_with_lookback, class_to_idx)

    start_epoch = load_checkpoint(model, optimizer, args.checkpoint) if args.resume else 0
    train(model, train_ds, val_ds, optimizer, args.checkpoint,
          start_epoch=start_epoch, num_epochs=args.epochs)
