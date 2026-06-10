"""BirdAVES-biox-large 埋め込み抽出ラッパー（torch / esp-aves）。

16kHz 波形 → HuBERT系エンコーダ最終層 → (T, 1024) 系列。
- 線形/MLPプローブ用: 時間平均 (1024,)
- attentiveプローブ用: 時間方向を固定長 S にプーリングした系列 (S, 1024)

torch 環境（tools/aves_embed/.venv）で使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aves import load_feature_extractor

AVES_SR = 16_000


class BirdAVESExtractor:
    def __init__(
        self,
        config_path: str | Path,
        model_path: str | Path,
        device: str = "cpu",
        seq_tokens: int = 32,
    ):
        self.device = device
        self.seq_tokens = seq_tokens
        self.model = load_feature_extractor(
            config_path=str(config_path),
            model_path=str(model_path),
            device=device,
            for_inference=True,
        )
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """波形(16kHz, mono)→ (mean_1024, seq_S_1024)。"""
        x = torch.as_tensor(np.asarray(wav16k, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (1, samples)
        x = x.to(self.device)
        feats = self.model.extract_features(x, layers=-1)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        feats = feats.squeeze(0)  # (T, 1024)
        mean = feats.mean(dim=0)  # (1024,)
        seq = self._pool_seq(feats)  # (S, 1024)
        return mean.cpu().numpy(), seq.cpu().numpy()

    def _pool_seq(self, feats: torch.Tensor) -> torch.Tensor:
        """時間方向 T を固定 S トークンへ適応平均プーリング → (S, 1024)。"""
        t = feats.transpose(0, 1).unsqueeze(0)  # (1, D, T)
        pooled = F.adaptive_avg_pool1d(t, self.seq_tokens)  # (1, D, S)
        return pooled.squeeze(0).transpose(0, 1).contiguous()  # (S, D)
