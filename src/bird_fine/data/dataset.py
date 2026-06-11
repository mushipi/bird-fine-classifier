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
        teacher_dir: Optional[Path] = None,
    ):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.label_map = label_map
        self.feature_extractor = feature_extractor
        self.project_root = project_root or PROJECT_ROOT
        self.sampling_rate = feature_extractor.sampling_rate

        # KD 用教師ソフトラベル（任意）。teacher_dir/{split}.npz を (xc_id, chunk_index) で整列。
        self.teacher = None
        self.has_teacher = None
        if teacher_dir is not None:
            tp = Path(teacher_dir) / f"{Path(split_csv).stem}.npz"
            t = np.load(tp, allow_pickle=True)
            key2row = {(str(x), int(c)): i
                       for i, (x, c) in enumerate(zip(t["xc_id"], t["chunk_index"]))}
            tproba = t["teacher_proba"]
            K = tproba.shape[1]
            n = len(self.df)
            self.teacher = np.zeros((n, K), dtype=np.float32)
            self.has_teacher = np.zeros(n, dtype=bool)
            for i, row in self.df.iterrows():
                k = (str(row["xc_id"]), int(row["chunk_index"]))
                j = key2row.get(k)
                if j is not None:
                    self.teacher[i] = tproba[j]
                    self.has_teacher[i] = True
            print(f"[KD] {Path(split_csv).stem}: 教師 proba coverage "
                  f"{self.has_teacher.sum()}/{n} (K={K})")

        self.spec_augment = None
        self.spec_augment_other_only = False
        cfg = spec_augment_cfg or {}
        # other_only=True の場合、enabled フラグに関わらず "other" クラスにのみ適用する
        other_only = cfg.get("other_only", False)
        apply = cfg.get("enabled", False) or other_only
        if apply:
            self.spec_augment = {
                "freq_mask": T.FrequencyMasking(
                    freq_mask_param=int(cfg["freq_mask_param"])
                ),
                "time_mask": T.TimeMasking(
                    time_mask_param=int(cfg["time_mask_param"])
                ),
                "num_freq": int(cfg["num_freq_masks"]),
                "num_time": int(cfg["num_time_masks"]),
            }
            self.spec_augment_other_only = other_only
            other_label = label_map.get("other")
            self.other_label_id = other_label  # None なら "other" クラスなし

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

        label = self.label_map[row["species"]]

        if self.spec_augment is not None:
            # other_only=True の場合は "other" ラベルのサンプルにのみ適用
            apply_aug = (
                not self.spec_augment_other_only
                or (self.other_label_id is not None and label == self.other_label_id)
            )
            if apply_aug:
                x = input_values.transpose(0, 1)  # (freq, time)
                for _ in range(self.spec_augment["num_freq"]):
                    x = self.spec_augment["freq_mask"](x)
                for _ in range(self.spec_augment["num_time"]):
                    x = self.spec_augment["time_mask"](x)
                input_values = x.transpose(0, 1)

        out = {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }
        if self.teacher is not None:
            out["teacher_proba"] = torch.from_numpy(self.teacher[idx])
            out["has_teacher"] = torch.tensor(bool(self.has_teacher[idx]))
        return out


def collate_fn(batch: list[dict]) -> dict:
    """DataLoader用 collate. AST入力をスタック。KD教師があれば併せてスタック。"""
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    out = {"input_values": input_values, "labels": labels}
    if "teacher_proba" in batch[0]:
        out["teacher_proba"] = torch.stack([b["teacher_proba"] for b in batch])
        out["has_teacher"] = torch.stack([b["has_teacher"] for b in batch])
    return out


def build_datasets(
    splits_dir: Path,
    pretrained: str,
    project_root: Optional[Path] = None,
    spec_augment_cfg: Optional[dict] = None,
    max_length: int = 1024,
    duck_order: Optional[list[str]] = None,
    teacher_dir: Optional[Path] = None,
) -> tuple[DuckChunkDataset, DuckChunkDataset, DuckChunkDataset, dict[str, int]]:
    """train/val/test の3つのDatasetを構築して返す。SpecAugmentはtrainのみ適用。

    duck_order を渡すと label_map をその順（種名→index）で構成する（10カモ運用＝teacher と整列）。
    teacher_dir を渡すと train/val に KD 教師ソフトラベルを読み込む（test は付けない）。
    """
    project_root = project_root or PROJECT_ROOT
    label_map = {d: i for i, d in enumerate(duck_order)} if duck_order else load_label_map(splits_dir)
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained, max_length=max_length)

    train_ds = DuckChunkDataset(
        splits_dir / "train.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=spec_augment_cfg, teacher_dir=teacher_dir,
    )
    val_ds = DuckChunkDataset(
        splits_dir / "val.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=None, teacher_dir=teacher_dir,
    )
    test_ds = DuckChunkDataset(
        splits_dir / "test.csv", label_map, feature_extractor, project_root,
        spec_augment_cfg=None,
    )
    return train_ds, val_ds, test_ds, label_map
