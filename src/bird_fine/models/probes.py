"""凍結埋め込みの上に乗せる軽量プローブ（線形 / 浅いMLP / attentive）。

レビュー論文(arXiv:2508.01277)準拠:
- 線形プローブ = 1線形層（CNN系=Perchはglobal avg pool埋め込みが入力）
- attentiveプローブ = 系列トークンに学習可能 attention pooling → 1線形層
  （patch/時系列トークンを公開する transformer系=BirdAVES のみ適用可）
"""
from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    """1線形層。入力 (B, D) 平均埋め込み。"""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class MLPProbe(nn.Module):
    """1隠れ層 MLP。入力 (B, D)。"""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentiveProbe(nn.Module):
    """学習可能 query による attention pooling → 線形。入力 (B, S, D) 系列。

    1本の学習 query が系列トークンを集約（multi-head attention）、その出力を分類。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        b = x.size(0)
        q = self.query.expand(b, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(q, x, x)  # (B, 1, D)
        pooled = self.norm(pooled.squeeze(1))  # (B, D)
        return self.fc(self.dropout(pooled))


def build_probe(probe: str, in_dim: int, num_classes: int, **kwargs) -> nn.Module:
    probe = probe.lower()
    if probe == "linear":
        return LinearProbe(in_dim, num_classes, dropout=kwargs.get("dropout", 0.0))
    if probe == "mlp":
        return MLPProbe(
            in_dim, num_classes,
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.3),
        )
    if probe == "attentive":
        return AttentiveProbe(
            in_dim, num_classes,
            num_heads=kwargs.get("num_heads", 4),
            dropout=kwargs.get("dropout", 0.1),
        )
    raise ValueError(f"unknown probe: {probe}")
