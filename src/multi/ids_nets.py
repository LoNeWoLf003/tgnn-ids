"""Window graphs + per-flow temporal graph models for the multi-dataset experiments.

Formulation (differs from the CIC-IDS2017 window-level pipeline on purpose):
  - node = flow, edge = k-NN similarity inside a fixed-size window (same graph idea as before);
  - the model predicts **every flow** in the target window (node-level anomaly score) instead of
    one any-attack label per window, which is degenerate on datasets that are 47-76% attacks;
  - temporal context = LSTM over the mean-pooled embeddings of the trailing K windows (including
    the target), concatenated to each node's embedding before the classification head.

Variants (`enc` x `temporal`): mlp (no graph, no time) | gcn / gat (graph, no time) |
gat+lstm (graph + time, the proposed model).
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import coalesce

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- graphs
@dataclass
class WindowStore:
    X: torch.Tensor          # (nW, W, d)
    y: torch.Tensor          # (nW, W)
    edges: list              # nW tensors (2, E)
    phase: np.ndarray        # (nW,)


def knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    n = x.shape[0]
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(min(k, n - 1), largest=False).indices
    src = torch.arange(n, device=x.device).repeat_interleave(idx.shape[1])
    dst = idx.reshape(-1)
    ei = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)  # symmetrise
    return coalesce(ei, num_nodes=n)


def build_store(prep, k: int = 8) -> WindowStore:
    W = prep.window
    X = torch.tensor(prep.X, device=DEVICE).view(-1, W, prep.n_features)
    y = torch.tensor(prep.y, dtype=torch.float32, device=DEVICE).view(-1, W)
    edges = [knn_edges(X[i], k) for i in range(X.shape[0])]
    return WindowStore(X, y, edges, prep.window_phase())


def history_index(targets: np.ndarray, K: int) -> np.ndarray:
    """(B,K) window indices ending at each target; clamped at window 0 for the very start."""
    idx = np.asarray(targets)[:, None] + np.arange(-K + 1, 1)[None, :]
    return np.clip(idx, 0, None)


# --------------------------------------------------------------------------- model
class NodeNet(nn.Module):
    def __init__(self, in_dim, enc="gat", temporal=False, hidden=64, lstm_hidden=32, heads=4,
                 drop=0.1):
        super().__init__()
        self.enc_kind, self.temporal, self.hidden = enc, temporal, hidden
        if enc == "mlp":
            self.c1, self.c2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
        elif enc == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)
        elif enc == "gat":
            self.c1 = GATConv(in_dim, hidden // heads, heads=heads)
            self.c2 = GATConv(hidden, hidden // heads, heads=heads)
        else:
            raise ValueError(enc)
        self.drop = nn.Dropout(drop)
        if temporal:
            self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        head_in = hidden + (lstm_hidden if temporal else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, X, edges):
        """X (G,W,d), edges: list of G (2,E) -> node embeddings (G,W,H)."""
        G, W, d = X.shape
        x = X.reshape(G * W, d)
        if self.enc_kind == "mlp":
            h = torch.relu(self.c1(x)); h = torch.relu(self.c2(self.drop(h)))
            return h.view(G, W, -1)
        ei = torch.cat([e + i * W for i, e in enumerate(edges)], dim=1)
        h = torch.relu(self.c1(x, ei)); h = self.drop(h)
        h = torch.relu(self.c2(h, ei))
        return h.view(G, W, -1)

    def forward(self, store: WindowStore, targets: np.ndarray, K: int):
        """-> logits (B,W), drift statistic ||pooled embedding of target|| (B,)."""
        hist = history_index(targets, K) if self.temporal else np.asarray(targets)[:, None]
        flat = hist.reshape(-1)
        h = self.encode(store.X[flat], [store.edges[i] for i in flat])
        B, KK = hist.shape
        h = h.view(B, KK, h.shape[1], -1)
        pooled = h.mean(2)                      # (B,KK,H)
        tgt = h[:, -1]                          # (B,W,H)
        if self.temporal:
            o, _ = self.lstm(pooled)
            tgt = torch.cat([tgt, o[:, -1:, :].expand(-1, tgt.shape[1], -1)], dim=-1)
        logits = self.head(tgt).squeeze(-1)
        p = torch.sigmoid(logits)
        ent = -(p * torch.log(p + 1e-9) + (1 - p) * torch.log(1 - p + 1e-9)).mean(1)
        # label-free drift statistics per target window: [||pooled embedding||, mean entropy]
        return logits, torch.stack([pooled[:, -1].norm(dim=-1), ent], dim=1)


def make_model(variant: str, in_dim: int) -> NodeNet:
    cfg = {"MLP": ("mlp", False), "GCN": ("gcn", False), "GAT": ("gat", False),
           "GAT+LSTM": ("gat", True)}[variant]
    return NodeNet(in_dim, enc=cfg[0], temporal=cfg[1]).to(DEVICE)


# --------------------------------------------------------------------------- train / predict
def fit(model, store, train_targets, epochs, lr, K, bs=8, val_targets=None, params=None,
        seed=0, patience=4):
    """Adam on BCE over every node of each target window. If val_targets is given, restores the
    best-val-loss weights (early stopping with `patience`)."""
    rng = np.random.default_rng(seed)
    params = [p for p in (params if params is not None else model.parameters())]
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    train_targets = np.asarray(train_targets)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(train_targets)
        for i in range(0, len(order), bs):
            t = order[i:i + bs]
            logits, _ = model(store, t, K)
            loss = lossf(logits, store.y[t])
            opt.zero_grad(); loss.backward(); opt.step()
        if val_targets is not None and len(val_targets):
            vp, _ = predict(model, store, val_targets, K, logits=True)
            vl = lossf(vp, store.y[np.asarray(val_targets)]).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, store, targets, K, bs=16, logits=False):
    """-> (probs or logits tensor (n,W), drift stats (n,)) for the given target windows."""
    model.eval()
    targets = np.asarray(targets)
    outs, stats = [], []
    for i in range(0, len(targets), bs):
        lg, st = model(store, targets[i:i + bs], K)
        outs.append(lg if logits else torch.sigmoid(lg)); stats.append(st)
    return torch.cat(outs), torch.cat(stats)


# --------------------------------------------------------------------------- drift loop
def adaptive_stream(base_model, variant, store, train_targets, stream_targets, strategy, K,
                    seed, delta=0.002, cooldown=10, recent_span=50, inc_epochs=3, inc_lr=1e-3,
                    retrain_epochs=10, retrain_lr=1e-3, escalate_span=40):
    """Score `stream_targets` window by window (always BEFORE adapting on that window), run ADWIN
    on ||pooled embedding||, adapt on a firing. Labels of already-scored windows are assumed to
    arrive afterwards (same supervised-feedback assumption as the CIC-IDS2017 pipeline).
    ADWIN is warm-started with the last 100 training windows. Returns (probs (n,W), events)."""
    import copy
    from river.drift import ADWIN

    model = copy.deepcopy(base_model)
    adwins = [ADWIN(delta=delta), ADWIN(delta=delta)]  # embedding norm, predictive entropy
    _, st = predict(model, store, np.asarray(train_targets)[-100:], K)
    st = st.cpu().numpy()
    mu, sd = st.mean(0), st.std(0) + 1e-6

    def squash(v):  # ADWIN's bound assumes values in ~[0,1]: z-score on warm-start stats -> [0,1]
        return np.clip(0.5 + (v - mu) / (8 * sd), 0.0, 1.0)

    for s in st:
        for a, v in zip(adwins, squash(s)):
            a.update(float(v))

    probs, events, last_fire, last_cheap = [], [], -10**9, -10**9
    for j, t in enumerate(stream_targets):
        p, st = predict(model, store, [t], K)
        probs.append(p[0])
        fired = False
        for a, v in zip(adwins, squash(st[0].cpu().numpy())):
            a.update(float(v))
            fired |= bool(a.drift_detected)
        if strategy == "no_adapt" or not fired or j - last_fire < cooldown:
            continue
        last_fire = j
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        action = strategy
        if strategy == "hybrid":
            # AdaOIL-style escalation: cheap update first; if drift re-fires within `escalate_span`
            # windows of the last cheap update (it did not resolve the drift), retrain fully.
            action = "full_retrain" if j - last_cheap < escalate_span else "incremental"
        if action in ("incremental", "incr_replay"):
            recent = np.asarray(stream_targets[max(0, j - recent_span + 1): j + 1])
            if action == "incr_replay":  # fine-tune with memory: recent + equal-size replay of old
                rr = np.random.default_rng(seed + j)
                old_w = rr.choice(np.asarray(train_targets), min(len(recent), len(train_targets)),
                                  replace=False)
                recent = np.concatenate([recent, old_w])
            sub = [p_ for n, p_ in model.named_parameters() if not n.startswith(("c1", "c2"))]
            fit(model, store, recent, inc_epochs, inc_lr, K, params=sub, seed=seed + j)
            n_used = len(recent)
            last_cheap = j
        else:  # full_retrain: from scratch on everything seen so far
            seen = np.concatenate([np.asarray(train_targets), np.asarray(stream_targets[: j + 1])])
            torch.manual_seed(seed + j)
            model = make_model(variant, store.X.shape[-1])
            fit(model, store, seen, retrain_epochs, retrain_lr, K, seed=seed + j)
            n_used = len(seen)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        events.append({"model": "GAT+LSTM", "action": action, "stream_idx": j, "window": int(t), "phase": int(store.phase[t]),
                       "wall_clock_s": time.perf_counter() - t0,
                       "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20
                                           if torch.cuda.is_available() else 0.0),
                       "n_windows": int(n_used)})
    return torch.stack(probs), events
