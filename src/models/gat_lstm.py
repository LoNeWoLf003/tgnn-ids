"""Core proposed model: GAT spatial encoder + LSTM temporal encoder (Days 7-9).

Operates on the windowed flow-similarity graph sequence from graph_build.py (node = flow, per
PLAN.md's data-constraint redesign). A forward pass takes a lookback of k+1 consecutive per-window
graphs [G_{t-k}, ..., G_t] and produces:
  - anomaly_score A_t in [0,1] for the current window t
  - class_logits: multi-class attack-type logits for window t (secondary head)

Device placement is handled internally (graphs are moved to the model's device on the fly), so
callers can pass raw torch_geometric Data objects straight from graph_sequence.pt.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphEncoder(nn.Module):
    """2-layer GAT spatial encoder. Returns node embeddings, optionally with the last layer's
    attention weights (for the explainability figures in §9 of IMPLEMENTATION.md)."""

    def __init__(self, in_dim, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, concat=False)
        self.gat2 = GATConv(hidden, hidden, heads=heads, concat=False)

    def forward(self, x, edge_index, return_attention=False):
        h, _ = self.gat1(x, edge_index, return_attention_weights=True)
        h = torch.relu(h)
        h, (ei2, alpha2) = self.gat2(h, edge_index, return_attention_weights=True)
        if return_attention:
            return h, (ei2, alpha2)
        return h


class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hidden=64, lstm_hidden=64, num_classes=15, heads=4):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden, heads=heads)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=lstm_hidden, batch_first=True)
        self.anomaly_head = nn.Linear(lstm_hidden, 1)
        self.class_head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, graph_window, return_attention=False):
        """graph_window: list of k+1 Data objects, [G_{t-k}, ..., G_t] (chronological order)."""
        device = next(self.parameters()).device
        embeddings = []
        attn = None
        for i, g in enumerate(graph_window):
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            if return_attention and i == len(graph_window) - 1:
                h, attn = self.encoder(x, edge_index, return_attention=True)
            else:
                h = self.encoder(x, edge_index)
            embeddings.append(h.mean(dim=0))  # graph-level readout per timestep

        seq = torch.stack(embeddings).unsqueeze(0)  # (1, k+1, hidden)
        z, _ = self.lstm(seq)
        z_t = z[:, -1, :]  # Z_t, per base.md §1.8

        anomaly_score = torch.sigmoid(self.anomaly_head(z_t))
        class_logits = self.class_head(z_t)
        if return_attention:
            return anomaly_score, class_logits, attn
        return anomaly_score, class_logits

    def embed(self, graph_window):
        """Returns Z_t only (no heads) -- used by drift.py's scalar drift statistic."""
        with torch.no_grad():
            device = next(self.parameters()).device
            embeddings = [self.encoder(g.x.to(device), g.edge_index.to(device)).mean(dim=0)
                          for g in graph_window]
            seq = torch.stack(embeddings).unsqueeze(0)
            z, _ = self.lstm(seq)
            return z[:, -1, :]
