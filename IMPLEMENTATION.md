# Implementation Guide — Adaptive Temporal Graph Learning for Network Anomaly Detection

> Companion to [`base.md`](./base.md) (research proposal) and [`PLAN.md`](./PLAN.md) (scoped
> day-by-day execution plan). Those answer *what* and *when*. This document answers *how*:
> concrete commands, file skeletons, function signatures, and hyperparameters, in build order.
>
> Follow this top-to-bottom. Each section corresponds to one or more days in `PLAN.md` and ends
> with a **Verify** step you must pass before moving on — don't build on top of something unverified.

---

## 0. Environment setup

This targets **local execution on your own machine**: Windows 11, NVIDIA RTX 2000 Ada Generation
Laptop GPU (8GB VRAM), driver 596.08 / CUDA 13.2. No Colab, no Drive — everything runs and
persists on local disk.

### 0.1 Local Python environment

Use a dedicated virtual environment so this project's pinned versions don't collide with anything
else on the machine. Run in PowerShell:

```powershell
# Pick a project root OUTSIDE the Perforce depot workspace, so multi-GB datasets/checkpoints
# never get swept into Perforce operations (add/edit/revert) by accident.
$PROJECT_ROOT = "$HOME\tgnn-ids"
New-Item -ItemType Directory -Force -Path $PROJECT_ROOT
Set-Location $PROJECT_ROOT

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Keep this project's code (`src/`, `paper/`, `IMPLEMENTATION.md`, `PLAN.md`, `base.md`) wherever
you like relative to the Perforce workspace — e.g. symlink or just copy `base.md`/`PLAN.md`/
`IMPLEMENTATION.md` into `$PROJECT_ROOT` alongside `src/`, and treat `$PROJECT_ROOT` as the
working directory for everything below. Add a `.gitignore`/`.p4ignore`-style exclusion for
`data/`, `checkpoints/`, and `results/*.png` if this ever gets checked in anywhere — they're
regenerable and large.

### 0.2 Install dependencies (CUDA-enabled PyTorch)

Your driver (CUDA 13.2) is newer than any current PyTorch CUDA build — that's fine, drivers are
backward-compatible with older CUDA toolkits. Install the current stable PyTorch CUDA wheel
(check https://pytorch.org/get-started/locally/ for the latest `cuXXX` tag if this one has aged out):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install river scikit-learn xgboost lightgbm pandas matplotlib pyarrow networkx
```

**Verify GPU is actually visible to PyTorch** (do this before writing any other code):

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: `True` and `NVIDIA RTX 2000 Ada Generation Laptop GPU`. If `False`, the wheel you
installed doesn't match your CUDA setup — reinstall with a different `cuXXX` index URL rather
than proceeding on CPU.

Pin versions once this works: `pip freeze > requirements.txt` — commit/save this file so the
environment is reproducible later.

### 0.3 Repo skeleton

Create this exact structure under `$PROJECT_ROOT` (matches `PLAN.md`'s repo structure):

```powershell
New-Item -ItemType Directory -Force -Path data/raw, data/interim, data/processed
New-Item -ItemType Directory -Force -Path src/models
New-Item -ItemType Directory -Force -Path checkpoints, results, paper
New-Item -ItemType File -Force -Path src/__init__.py, src/models/__init__.py
```

Files you will create, in build order (each has its own section below):

```
src/ingest.py
src/features.py
src/graph_build.py
src/models/baselines.py
src/models/gat_lstm.py
src/models/drift.py
src/train.py
src/eval.py
```

**Verify:** `import torch, torch_geometric, river, sklearn, xgboost` runs with no errors (see §0.2 for the CUDA check).

---

## 1. Data acquisition + cleaning (`src/ingest.py`) — Days 1-2

### 1.1 Dataset actually in hand — data constraint

`data/raw/MachineLearningCVE/*.csv` (already present, ~2.83M flows across 8 files) is the common
redistributed CIC-IDS2017 mirror. **Its columns are `Destination Port` + ~70 CICFlowMeter
flow-statistic columns + `Label` only** — no `Flow ID`, `Source IP`, `Destination IP`,
`Source Port`, `Protocol`, or `Timestamp`. The official UNB `GeneratedLabelledFlows.zip` (which has
those columns) is registration-gated and was not pursued — see `PLAN.md`'s data-constraint note.
**Everything below is redesigned around this**, not the original host/timestamp assumption.

Files (unchanged from the original plan — 8 files, one per day/segment):
```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### 1.2 `src/ingest.py`

No timestamp exists, so chronological order is **assumed to be row order within each day-file**
(CICFlowMeter emits flows in capture order) — an explicit, documented limitation, not fabricated.
A synthetic `seq` column (0..n-1 per day, preserved through concat) stands in for a timestamp
everywhere downstream needs "chronological position."

```python
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/MachineLearningCVE")
# Ordered dict: concat order == assumed chronological day order (Mon -> Fri)
DAY_MAP = {
    "Monday-WorkingHours.pcap_ISCX.csv": "mon",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "tue",
    "Wednesday-workingHours.pcap_ISCX.csv": "wed",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "thu_am",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "thu_pm",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "fri_am",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "fri_pm_scan",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "fri_pm_ddos",
}

def load_and_clean_day(filename: str, day_tag: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename, low_memory=False)
    df.columns = df.columns.str.strip()               # known issue: leading spaces in headers
    df["Label"] = df["Label"].str.strip()
    df["Label"] = df["Label"].replace({                 # known label typos (community errata)
        "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
        "Web Attack \x96 XSS": "Web Attack - XSS",
        "Web Attack \x96 Sql Injection": "Web Attack - SQL Injection",
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    df["day"] = day_tag
    df["seq"] = np.arange(len(df))          # per-day chronological-order proxy (row order)
    return df

def build_dataset() -> pd.DataFrame:
    frames = [load_and_clean_day(f, tag) for f, tag in DAY_MAP.items()]
    full = pd.concat(frames, ignore_index=True)   # day order preserved by dict iteration order
    full.to_parquet("data/interim/cicids2017_clean.parquet", index=False)
    return full

if __name__ == "__main__":
    df = build_dataset()
    print(df.shape, df["day"].value_counts(), df["Label"].value_counts())
```

**Verify:** output parquet loads back with `pd.read_parquet`, row count roughly matches published
dataset size (~2.8M flows total before cleaning, fewer after dropping NaN/Inf), `day` column has
all 8 tags, `Label` has no stray encoding artifacts (`\x96` etc.) — grep the unique labels manually.

---

## 2. Feature engineering — per-flow scaling + windowing (`src/features.py`) — Days 3-4

**(Redesigned — no host identity in the data.)** Two jobs: (1) scale the per-flow numeric feature
vector that becomes each graph node's `x`, and (2) assign each flow a `window_id` — a fixed-size
chunk of consecutive rows *within its day* (never spanning a day boundary, so the Mon-Fri phase
mapping in §6 stays clean) — standing in for the original 5-minute wall-clock window.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

WINDOW_SIZE = 1000   # flows per window — hyperparameter, documented not exhaustively tuned

NUMERIC_FEATURE_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

def assign_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    df = df.copy()
    df["local_window"] = df["seq"] // window_size            # window index within its day
    day_order = {d: i for i, d in enumerate(df["day"].unique())}
    # global window_id keeps windows ordered Mon->Fri, never merging across a day boundary
    df["window_id"] = df["day"].map(day_order) * (df["local_window"].max() + 1) + df["local_window"]
    return df

def fit_scaler(df: pd.DataFrame, fit_days=("mon", "tue")) -> StandardScaler:
    """Fit on Mon+Tue only (Experiment A's stationary split) to avoid leakage from later days."""
    scaler = StandardScaler()
    scaler.fit(df.loc[df["day"].isin(fit_days), NUMERIC_FEATURE_COLS])
    joblib.dump(scaler, "data/processed/scaler.joblib")
    return scaler

def add_window_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-window destination-port entropy — partial substitute for the dropped host-level
    port_entropy signal (§1.22 lateral-movement folding is no longer possible, see PLAN.md)."""
    def _entropy(s):
        counts = s.value_counts(normalize=True)
        return float(-(counts * np.log2(counts + 1e-12)).sum())
    port_entropy = df.groupby("window_id")["Destination Port"].apply(_entropy)
    df = df.merge(port_entropy.rename("window_port_entropy"), on="window_id", how="left")
    return df

def window_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per window_id: majority Label (classification head) + any-attack flag (anomaly head)."""
    y_anom = df.groupby("window_id")["Label"].apply(lambda s: float((s != "BENIGN").any()))
    y_class = df.groupby("window_id")["Label"].agg(lambda s: s.mode().iloc[0])
    day = df.groupby("window_id")["day"].first()
    return pd.DataFrame({"y_anomaly": y_anom, "y_class": y_class, "day": day}).reset_index()
```

**Verify:** spot-check 2-3 `window_id`s by hand — row count per window matches `WINDOW_SIZE` (except
the last, partial window per day), `window_id` never mixes rows from two different `day` values,
and rescaling one flow manually with the fitted `scaler` matches `scaler.transform` output.

---

## 3. Graph construction (`src/graph_build.py`) — Days 3-4

**(Redesigned — no host identity, see PLAN.md's data-constraint note.)** Build one
`torch_geometric.data.Data` object per window: **node = flow**, **edge = k-NN in scaled feature
space**. Unlike the host design, node count varies per window (all flows in that window) — no
zero-padding needed, PyG handles variable-sized graphs natively.

```python
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import pandas as pd

K_NEIGHBORS = 8   # hyperparameter — documented, not exhaustively tuned per PLAN.md

def build_graph_sequence(flow_df: pd.DataFrame, feature_cols: list[str], k: int = K_NEIGHBORS):
    """
    Returns: list[Data], one per window_id, sorted chronologically (window_id already encodes
    Mon->Fri order from features.py's assign_windows).
    """
    graphs = []
    for window_id, g in flow_df.groupby("window_id"):
        g = g.reset_index(drop=True)
        x = torch.tensor(g[feature_cols].values.astype(np.float32))

        n = len(g)
        k_eff = min(k, n - 1) if n > 1 else 0
        if k_eff > 0:
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(x.numpy())  # +1: includes self
            _, idx = nn.kneighbors(x.numpy())
            src = np.repeat(np.arange(n), k_eff)
            dst = idx[:, 1:].reshape(-1)                      # drop self-neighbor (col 0)
            # symmetrize: undirected k-NN graph
            edge_index = torch.tensor(
                np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1),
                dtype=torch.long,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)  # single-flow window: no edges

        y_anomaly = torch.tensor(float((g["Label"] != "BENIGN").any()))
        y_class = g["Label"].mode()[0]

        graphs.append(Data(
            x=x, edge_index=edge_index,
            y=y_anomaly, y_class=y_class, window_id=window_id, day=g["day"].iloc[0],
        ))

    graphs.sort(key=lambda d: d.window_id)
    return graphs
```

Save the sequence with `torch.save(graphs, "data/processed/graph_sequence.pt")` — this is the
**single most-depended-on artifact** in the whole project (per `PLAN.md`'s critical-files list).

**Verify:**
```python
graphs = build_graph_sequence(flow_df, feature_cols)
assert len(graphs) > 0
assert all(g.x.shape[0] >= 1 for g in graphs)
print(graphs[0])
```
Also manually cross-check one window's node count against `flow_df["window_id"].value_counts()`,
and confirm `edge_index.max() < x.shape[0]` for a few sampled windows (no out-of-range node refs).

---

## 4. Baselines (`src/models/baselines.py`) — Days 5-6

Common interface every model (baseline and proposed) implements, so `eval.py` can treat them
uniformly:

```python
class DetectorBase:
    def fit(self, X_train, y_train): ...
    def score(self, X_test) -> "anomaly scores in [0,1]": ...
    def predict(self, X_test, threshold=0.5) -> "binary labels": ...
```

### 4.1 Flow-level tabular baselines (Random Forest, XGBoost, MLP)

Use the flow-level feature matrix directly (not the graph) — flatten `flow_df` numeric columns,
scale with `StandardScaler`, time-based train/test split (never random — per `base.md` §1.17).

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def fit_rf(X, y):
    m = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, class_weight="balanced")
    m.fit(X, y); return m

def fit_xgb(X, y):
    m = XGBClassifier(n_estimators=300, max_depth=6, tree_method="hist",
                       scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    m.fit(X, y); return m
```

### 4.2 Sequence baseline (LSTM on flow sequences) + Autoencoder

- LSTM: sort flows chronologically per host, feed fixed-length rolling sequences of flow feature
  vectors, binary output head.
- Autoencoder: train on **BENIGN-only** flows, reconstruction error = anomaly score (this is the
  unsupervised baseline referenced in `base.md` §1.15).

### 4.3 Static graph baselines (GCN, GAT — no LSTM)

Operate on individual `Data` graphs from `graph_build.py`, one window at a time, no temporal
memory — this isolates the "graph" contribution before adding "temporal" (feeds directly into the
ablation study, §1.16).

```python
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn

class StaticGCN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        graph_emb = h.mean(dim=0)          # simple readout
        return torch.sigmoid(self.head(graph_emb))
```

`StaticGAT` is identical with `GATConv(in_dim, hidden, heads=4, concat=False)` in place of
`GCNConv` — needed later for attention-based explainability (§7).

**Verify:** each baseline trains without NaN loss, produces a score in `[0,1]`, and
`results/baselines.csv` has one row per (model, dataset-split, metric).

---

## 5. Core model — GAT + LSTM (`src/models/gat_lstm.py`) — Days 7-9

### 5.1 Day 7 — spatial encoder alone

```python
import torch, torch.nn as nn
from torch_geometric.nn import GATConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.gat2 = GATConv(hidden, hidden, heads=heads, concat=False)

    def forward(self, x, edge_index, return_attention=False):
        h, (ei1, alpha1) = self.gat1(x, edge_index, return_attention_weights=True)
        h = torch.relu(h)
        h, (ei2, alpha2) = self.gat2(h, edge_index, return_attention_weights=True)
        if return_attention:
            return h, (ei2, alpha2)
        return h
```

Probe-train it alone with a linear head (`nn.Linear(hidden, 1)` on `h.mean(0)`) against
per-window anomaly labels, on Mon+Tue data only, to confirm the spatial encoder learns *something*
before adding temporal complexity. **Don't skip this step** — it isolates bugs.

### 5.2 Day 8 — add LSTM temporal encoder + heads

```python
class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hidden=64, lstm_hidden=64, k=6, num_classes=8):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=lstm_hidden, batch_first=True)
        self.anomaly_head = nn.Linear(lstm_hidden, 1)
        self.class_head = nn.Linear(lstm_hidden, num_classes)
        self.k = k   # window lookback, §1.6

    def forward(self, graph_window, return_attention=False):
        """
        graph_window: list of k+1 Data objects, [G_{t-k}, ..., G_t]
        """
        embeddings = []
        attn = None
        for i, g in enumerate(graph_window):
            if return_attention and i == len(graph_window) - 1:
                h, attn = self.encoder(g.x, g.edge_index, return_attention=True)
            else:
                h = self.encoder(g.x, g.edge_index)
            embeddings.append(h.mean(dim=0))   # graph-level readout per timestep

        seq = torch.stack(embeddings).unsqueeze(0)      # (1, k+1, hidden)
        z, _ = self.lstm(seq)
        z_t = z[:, -1, :]                                 # Z_t, per base.md §1.8

        anomaly_score = torch.sigmoid(self.anomaly_head(z_t))
        class_logits = self.class_head(z_t)
        if return_attention:
            return anomaly_score, class_logits, attn
        return anomaly_score, class_logits
```

### 5.3 Day 9 — training loop with checkpoint/resume (`src/train.py`)

On a local machine this isn't protecting against a session timeout, but it's still worth doing:
a laptop can still crash, sleep, lose power, or you may just want to `Ctrl+C` a run and resume it
later without losing hours of training. Every experiment depends on this being correct before you
run anything long.

```python
import torch, argparse, json, os

def train(model, dataloader, optimizer, criterion_anom, criterion_cls,
          checkpoint_path, start_epoch=0, num_epochs=20, log_path="results/train_log.csv"):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for graph_window, y_anom, y_cls in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                score, logits = model(graph_window)
                loss = criterion_anom(score, y_anom) + 0.5 * criterion_cls(logits, y_cls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # checkpoint EVERY epoch — a disconnect mid-run must not lose progress
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }, checkpoint_path)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{total_loss:.4f}\n")   # plain CSV, no MLflow/W&B — per PLAN.md
        print(f"epoch {epoch} loss {total_loss:.4f} — checkpoint saved")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    torch.set_rng_state(ckpt["rng_state"])
    return ckpt["epoch"] + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/gat_lstm.pt")
    args = parser.parse_args()
    # ... build model, optimizer, dataloader ...
    start_epoch = load_checkpoint(model, optimizer, args.checkpoint) if args.resume else 0
    train(model, dataloader, optimizer, criterion_anom, criterion_cls,
          args.checkpoint, start_epoch=start_epoch)
```

Hyperparameters to start with (don't sweep — `PLAN.md` explicitly caps tuning effort):

| Param | Value |
|---|---|
| hidden dim | 64 |
| GAT heads | 4 |
| LSTM hidden | 64 |
| window lookback `k` | 6 (i.e. 30 min of history at 5-min windows) |
| learning rate | 1e-3 (Adam) |
| batch | 1 graph-window sequence at a time (small graphs — no need to batch further) |
| epochs | 20, early-stop on validation F1 |
| mixed precision | on (`torch.cuda.amp`) |

Your 8GB VRAM is less than a Colab T4's 16GB, but these graphs and hidden dims are small enough
that it shouldn't matter — watch `nvidia-smi` (or `nvidia-smi -l 2` in a second terminal) during
the first training run. If you hit `CUDA out of memory`, drop `hidden`/`lstm_hidden` to 32 before
reducing anything else.

**Verify (crash/resume drill — do this before trusting any later result):**
1. Start training, let it checkpoint at least once, then kill the process (`Ctrl+C`, or End Task
   in Task Manager) mid-epoch.
2. Restart with `python src/train.py --resume`.
3. Confirm the loss curve continues smoothly from where it left off (no jump/reset), and
   `epoch` in the log picks up at `last_checkpoint_epoch + 1`.

---

## 6. Drift detection (`src/models/drift.py`) — Days 10-11

### 6.1 Map CIC-IDS2017 days to drift phases

Per `PLAN.md`'s locked-in decision — don't fabricate synthetic drift, use the dataset's native
day-to-day composition shift:

```python
PHASE_MAP = {
    "mon": 1,       # benign baseline
    "tue": 2,       # brute-force introduced
    "wed": 3,       # DoS
    "thu_am": 4, "thu_pm": 4,   # web attack / infiltration — held out of training as "unseen"
    "fri_am": 5, "fri_pm_scan": 5, "fri_pm_ddos": 5,   # botnet/portscan/DDoS
}
PHASE_BOUNDARIES = ["mon->tue", "tue->wed", "wed->thu_am", "thu_pm->fri_am"]  # ground truth for plots
```

### 6.2 ADWIN wrapper over embedding stream

```python
from river.drift import ADWIN

class EmbeddingDriftDetector:
    def __init__(self):
        self.adwin = ADWIN()
        self.drift_points = []

    def update(self, z_t_scalar_stat: float, window_id):
        in_drift, _ = self.adwin.update(z_t_scalar_stat)
        if in_drift:
            self.drift_points.append(window_id)
        return in_drift
```

`z_t_scalar_stat`: start with the **L2 norm of `Z_t`** (cheap, no extra model needed) or the
anomaly-head's reconstruction/prediction error trend. Feed one scalar per window, in chronological
order, across the full Mon→Fri sequence.

Optional secondary signal (cut first if behind schedule per `PLAN.md`'s cut-line #2):
```python
from scipy.spatial.distance import jensenshannon
import numpy as np

def js_divergence_drift(z_window_a: np.ndarray, z_window_b: np.ndarray, bins=20) -> float:
    hist_a, edges = np.histogram(z_window_a, bins=bins, density=True)
    hist_b, _ = np.histogram(z_window_b, bins=edges, density=True)
    return jensenshannon(hist_a + 1e-12, hist_b + 1e-12)
```

**Verify:** run the detector over the full Mon→Fri embedding stream from the *already-trained*
(Day 9) model; plot `drift_points` against `PHASE_BOUNDARIES` — they should land close to the real
day cutovers. If ADWIN never fires, its sensitivity parameter (`delta`, default 0.002) needs
tightening — don't move on to Experiment C/D until this fires near the true boundaries (this is
explicitly a done-criterion in `PLAN.md`).

---

## 7. Adaptation strategies (also `src/models/drift.py` / `src/train.py`) — Days 10-14

Three strategies to implement behind one interface:

```python
def no_adapt(model, new_window_data):
    return model   # unchanged, baseline for comparison

def full_retrain(model_class, all_data_so_far, **train_kwargs):
    model = model_class(**train_kwargs)
    train(model, all_data_so_far, ...)   # from scratch, per §1.11's naive approach
    return model

def incremental_adapt(model, new_window_data, layers_to_unfreeze=("lstm", "anomaly_head")):
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    train(model, new_window_data, optimizer, ..., num_epochs=3)   # few epochs, only on recent data
    return model
```

Trigger logic to wire into the Mon→Fri evaluation loop:

```python
detector = EmbeddingDriftDetector()
for window_id, graph_window in enumerate(sequence):
    score, logits = model(graph_window)
    log_metrics(window_id, score, logits)     # always log before any adaptation

    z_stat = compute_scalar_stat(model, graph_window)
    if detector.update(z_stat, window_id):
        log(f"drift detected at window {window_id}")
        if STRATEGY == "incremental":
            model = incremental_adapt(model, recent_windows(sequence, window_id))
        elif STRATEGY == "full_retrain":
            model = full_retrain(TemporalGAT, sequence[:window_id])
        # STRATEGY == "no_adapt": do nothing
```

Also record, per adaptation event: wall-clock time, peak GPU memory (`torch.cuda.max_memory_allocated()`), and number of samples used — this feeds Experiment D / §1.14's cost table directly.

**Verify:** run all three strategies over the same Mon→Fri sequence, confirm `no_adapt` never
changes model weights (checksum `model.state_dict()` before/after), `full_retrain` and
`incremental` both produce a valid, loadable model after each adaptation event.

---

## 8. Experiments A–D + ablation (`src/eval.py`) — Days 12-14

Single reusable eval function:

```python
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score

def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "f1": f1_score(y_true, y_pred),
        "pr_auc": auc(recall, precision),
        "roc_auc": roc_auc_score(y_true, y_score),
        "fpr": ((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1),
    }
```

Run each experiment, appending rows to `results/experiment_log.csv` with columns
`[experiment, model, day, window_id, f1, pr_auc, roc_auc, fpr, phase]` — every downstream table
and figure is generated *from this one file*, never hand-typed (a `PLAN.md` done-criterion).

- **A (stationary):** filter to `day in {mon, tue}`, time-based split, run every model from
  §4/§5, write `results/experiment_a.csv`.
- **B (temporal contribution):** run static-ML / static-GCN / GAT+LSTM across the full
  Mon→Fri sequence with `no_adapt`, write `results/experiment_b.csv`.
- **C (drift degradation):** reuse B's run; plot F1 per window with `PHASE_BOUNDARIES` marked as
  vertical lines (matplotlib `axvline`). Save as `results/fig_drift_degradation.png`.
- **D (adaptation):** run `no_adapt` / `full_retrain` / `incremental` across Mon→Fri, write
  `results/experiment_d.csv` including timing/memory columns.
- **Ablation:** aggregate rows already produced by A/B/D into `results/ablation.csv` — Model1
  (flow-only MLP), Model2 (static GCN), Model3 (GAT+LSTM no-adapt), Model4 (GAT+LSTM incremental).

**Verify:** open each `results/*.csv`, confirm no NaN/placeholder values; sanity-check the
proposed model is not worse than every baseline in Experiment A — if it is, that's a bug to fix
before proceeding, not a result to report (explicit `PLAN.md` done-criterion).

---

## 9. Explainability — attention visualization (Days 12-14, parallel)

**(Reframed — no host identity.)** Attention now highlights which *neighboring flows* (not hosts)
most influenced a given flow's contribution to the window's anomaly signal — "primary contributing
flow(s) / suspicious feature-similarity relationships" rather than `base.md` §1.23's host framing.
Label each node by its `Destination Port` + truncated `Label` (the only identifying fields left)
instead of an IP.

```python
import matplotlib.pyplot as plt
import networkx as nx

def plot_attention(graph, edge_index, alpha, flow_labels, top_k=10, out_path="results/fig_attention.png"):
    """flow_labels: list[str], one per node in `graph`, e.g. f"port={port} label={label}" """
    alpha = alpha.mean(dim=1).detach().cpu().numpy()   # average over heads
    top_edges = alpha.argsort()[-top_k:]
    G = nx.DiGraph()
    for i in top_edges:
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(flow_labels[src], flow_labels[dst], weight=float(alpha[i]))
    pos = nx.spring_layout(G)
    weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=weights, node_size=800, font_size=8)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
```

Pick 2-3 true-positive anomaly windows (high `y_anomaly` score, correctly flagged), run
`model(graph_window, return_attention=True)`, produce one figure per window.

**Verify:** the highlighted top-attention edges in at least one figure visibly connect flows sharing
the actual attack `Label` for that window (cross-check against `Label` in the raw flow data) — i.e.
attention concentrates among same-attack flows, not scattered randomly across benign ones.

---

## 10. Paper writing (Days 13-18, parallel with experiments)

Write directly into `paper/paper.md`, structured per `base.md` §1.25 / `PLAN.md`'s section list.
Practical process:

1. **Draft the skeleton first** (section headers + one-sentence placeholders) on Day 10, before
   any results exist — this forces the formalism (`G_t=(V_t,E_t)`, `f(𝒢_{t-k:t})→A_t`) to be
   pinned down early and catches notation inconsistencies before they propagate into figures.
2. **Fill Introduction / Related Work / Problem Formulation / Method** as soon as the
   corresponding component (§1-§7 above) is implemented — don't wait for all experiments to
   finish to start writing about the method.
3. **Fill Experiments/Results last**, generating every number/figure programmatically from
   `results/*.csv` (never hand-typed) — re-run the figure-generation script if any experiment is
   re-run, don't silently edit numbers in the text.
4. **Limitations section**, written honestly per `PLAN.md`'s framing: single dataset, day-boundary
   drift (not fully organic), attention-only explainability.
5. **Conclusion**, framed around the central hypothesis from `base.md` §1.27 — state plainly
   whether results support or reject it.

**Verify:** no `X.XX` or `TODO` placeholders remain anywhere in `paper/paper.md`; every figure
file referenced actually exists in `results/`; every claimed number appears in some
`results/*.csv` row (spot-check 5 numbers by grep).

---

## Master checklist (tick top-to-bottom; matches `PLAN.md`'s day ranges)

- [ ] Local venv + CUDA-enabled PyTorch installed, GPU verified visible, repo skeleton created
- [ ] `data/interim/cicids2017_clean.parquet` built and spot-checked
- [ ] `host_behavior_vectors()` output validated against raw flow groupby
- [ ] `data/processed/graph_sequence.pt` built and spot-checked
- [ ] Baselines trained, `results/baselines.csv` populated
- [ ] `GraphEncoder` alone trains stably (probe-head sanity check)
- [ ] `TemporalGAT` full model trains; beats static-GNN baseline on stationary split
- [ ] Checkpoint/resume drill passed (kill + restart mid-training)
- [ ] Drift detector fires near true Mon-Fri phase boundaries
- [ ] All three adaptation strategies run end-to-end with correct weight-freeze behavior
- [ ] Experiments A-D + ablation CSVs populated with real numbers, no NaNs
- [ ] Proposed model beats all baselines in Experiment A (or bug fixed if not)
- [ ] Attention figures generated and cross-checked against ground-truth attacker IP
- [ ] Paper draft complete, all 6 sections, zero placeholders
- [ ] Buffer days used for polish or CSE-CIC-IDS2018 stretch goal if ahead of schedule