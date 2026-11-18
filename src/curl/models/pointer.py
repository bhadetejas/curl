
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NODE_FEATURE_DIM = 12


def node_features(obs, scale: float | None = None, centre: np.ndarray | None = None):
    pos = obs["positions"][:, :2]
    if centre is None:
        centre = pos.mean(0)
    if scale is None:
        scale = max(float(np.ptp(pos, axis=0).max()), 1e-6)
    xy = (pos - centre) / scale
    n = obs["normals"]
    visited = obs["visited"].astype(float)
    current = np.zeros(len(pos))
    if obs["current"] >= 0:
        current[obs["current"]] = 1.0
    reachable = obs["reachable"].astype(float)
    uncovered = obs["uncovered"].astype(float)
    if obs["current"] >= 0:
        rel = (pos - pos[obs["current"]]) / scale
    else:
        rel = np.zeros_like(pos)
    global_cov = np.full(len(pos), float(obs["coverage"]))
    return np.column_stack([xy, n, visited, current, reachable,
                            uncovered, rel, global_cov]).astype(np.float32)


class PointerPolicy(nn.Module):
    def __init__(self, in_dim: int = NODE_FEATURE_DIM, d: int = 128, layers: int = 3,
                 heads: int = 8):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.start = nn.Parameter(torch.zeros(d))
        enc_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                               batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.q = nn.Linear(2 * d, d)
        self.k = nn.Linear(d, d)
        self.d = d

    def forward(self, feats: torch.Tensor, current: torch.Tensor, mask: torch.Tensor):
        h = self.encoder(self.embed(feats))
        g = h.mean(dim=1)
        idx = current.clamp(min=0)
        h_cur = h[torch.arange(h.size(0)), idx]
        h_cur = torch.where((current >= 0).unsqueeze(1), h_cur, self.start.expand_as(h_cur))
        q = self.q(torch.cat([g, h_cur], dim=1))
        logits = torch.einsum("bd,bnd->bn", q, self.k(h)) / np.sqrt(self.d)
        logits = 10.0 * torch.tanh(logits / 10.0)
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1)
