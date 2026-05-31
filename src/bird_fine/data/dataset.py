"""PyTorch Dataset for AST fine-tuning.

split CSV (train.csv/val.csv/test.csv) を読んで音声をAST入力フォーマットに変換。
ASTFeatureExtractor.from_pretrained() がメルスペクトログラム化を担当。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_label_map(splits_dir: Path) -> dict[str, int]:
    """splits/label_map.csv から species -> label_id を読む。"""
    df = pd.read_csv(splits_dir / "label_map.csv")
    return {row["species"]: int(row["label_id"]) for _, row in df.iterrows()}


class DuckChunkDataset(Dataset):
    """前処理済み10秒チャンクをAST入力に変換するDataset。

    Args:
        split_csv: train.csv / val.csv / test.csv のパス
        label_map: species名 → label_id の辞書
        feature_extractor: ASTFeatureExtractor
        project_root: チャンクの相対パス解決用
    """

    def __init__(
        self,
        split_csv: Path,
        label_map: dict[str, int],
        feature_extractor: ASTFeatureExtractor,
        project_root: Optional[Path] = None,
        spec_augment_cfg: Optional[dict] = None,
    ):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.label_map = label_map
        self.feature_extractor = feature_extractor
        self.project_root = project_root or PROJECT_ROOT
        self.sampling_rate = feature_extractor.sampling_rate

        self.spec_augment = None
        if spec_augment_cfg and spec_augment_cfg.get("enabled", False):
            self.spec_augment = {
                "freq_mask": T.FrequencyMasking(
                    freq_mask_param=int(spec_augment_cfg["freq_mask_param"])
                ),
                "time_mask": T.TimeMasking(
                    time_mask_param=int(spec_augment_cfg["time_mask_param"])
                ),
                "num_freq": int(spec_augment_cfg["num_freq_masks"]),
                "num_time": int(spec_augment_cfg["num_time_masks"]),
            }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        file_path = self.project_root / Path(row["file_path"].replace("\\", "/"))

        audio, sr = sf.read(str(file_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            # 前処理段階で揃えてあるはずだが念のため
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].squeeze(0)  # (time, freq)

        if self.spec_augment is not None:
            # torchaudio の masking は (..., freq, time) を想定。AST は (time, freq) なので転置
            x = input_values.transpose(0, 1)  # (freq, time)
            for _ in range(self.spec_augment["num_freq"]):
                x = self.spec_augment["freq_mask"](x)
            for _ in range(self.spec_augment["num_time"]):
                x = self.spec_augment["time_mask"](x)
            input_values = x.transpose(0, 1)  # (time, freq) に戻す

        label = self.label_map[row["species"]]

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """DataLoader用 collate. AST入力をスタック。"""
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_values": input_values, "labels": labels}


def build_datasets(
    splits_dir: Path,
    pretrained: str,
    project_root: Optional[Path] = None,
    spec_augment_cfg: Optional[dict] = None,
    max_length: int = 1024,
) -> tuple[DuckChunkDataset, DuckChunkDataset, DuckChunkDataset, dict[str, int]]:
    """train/val/test の3つのDatasetを構築して返す。SpecAugmentはtrainのみ適用。"""
    project_root = project_root or PROJECT_ROOT
    label_map = load_label_map(splits_dir)
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained, max_length=max_length)

    train_ds = DuckChunkDataset(
        splits_dir / "train.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=spec_augment_cfg,
    )
    val_ds = DuckChunkDataset(
        splits_dir / "val.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=None,
    )
    test_ds = DuckChunkDataset(
        splits_dir / "test.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=None,
    )
    return train_ds, val_ds, test_ds, label_map
