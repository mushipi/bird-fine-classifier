"""埋め込み抽出の共通ユーティリティ（numpy/pandas のみ、torch/TF 非依存）。

- split CSV（train/val/test.csv）の読み込みとパス正規化
- preprocess._chunk_audio と同一のチャンク境界を再現するチャンカ（SR非依存）
- 埋め込み npz の保存/読み込みスキーマ

抽出器（Perch=TF環境 / BirdAVES=torch環境）の双方から import される土台。
重い依存を引かないよう、ここは標準库 + numpy + pandas だけに保つ。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# src/bird_fine/embeddings/io_utils.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# split CSV の列（preprocess.py / split.py と一致）
SPLIT_COLUMNS = ("species", "xc_id", "chunk_index", "file_path", "duration_sec", "source_file")


def load_label_map(splits_dir: Path) -> dict[str, int]:
    """splits/label_map.csv から species -> label_id。

    data.dataset.load_label_map と同等だが、torch を引かないよう pandas で実装。
    """
    df = pd.read_csv(Path(splits_dir) / "label_map.csv")
    return {row["species"]: int(row["label_id"]) for _, row in df.iterrows()}


def label_names_from_map(label_map: dict[str, int]) -> list[str]:
    """label_id 昇順の種名リスト。"""
    id2label = {v: k for k, v in label_map.items()}
    return [id2label[i] for i in range(len(id2label))]


def normalize_rel_path(rel: str) -> Path:
    """split CSV の file_path（Windows '\\' 区切りもある）を絶対パスへ正規化。"""
    return PROJECT_ROOT / str(rel).replace("\\", "/")


def raw_audio_path(species: str, source_file: str, raw_dir: Path | None = None) -> Path:
    """生音源（data/raw/<species>/<source_file>）の絶対パス。"""
    raw_dir = Path(raw_dir) if raw_dir is not None else PROJECT_ROOT / "data" / "raw"
    return raw_dir / species / source_file


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float = 10.0,
    min_duration: float = 3.0,
    overlap_ratio: float = 0.0,
) -> list[np.ndarray]:
    """preprocess._chunk_audio と同一規則のチャンク分割（SR非依存で境界一致）。

    overlap_ratio=0 のとき chunk_index i は時刻 [i*chunk, (i+1)*chunk] に対応。
    末尾が min_duration 以上ならゼロパディングして1チャンク追加。
    """
    chunk_samples = int(chunk_duration * sr)
    min_samples = int(min_duration * sr)
    step = int(chunk_samples * (1.0 - overlap_ratio))
    if step <= 0:
        step = chunk_samples

    n = len(audio)
    chunks: list[np.ndarray] = []
    if n < min_samples:
        return chunks
    if n <= chunk_samples:
        padded = np.zeros(chunk_samples, dtype=audio.dtype)
        padded[:n] = audio
        return [padded]

    start = 0
    while start < n:
        end = start + chunk_samples
        if end <= n:
            chunks.append(audio[start:end])
        else:
            remaining = n - start
            if remaining >= min_samples:
                padded = np.zeros(chunk_samples, dtype=audio.dtype)
                padded[:remaining] = audio[start:n]
                chunks.append(padded)
            break
        start += step
        if start + chunk_samples > n and (n - start) < min_samples:
            break
    return chunks


def read_split(split_csv: Path) -> pd.DataFrame:
    """split CSV を読み、chunk_index を int 化して返す。"""
    df = pd.read_csv(split_csv)
    df["chunk_index"] = df["chunk_index"].astype(int)
    return df.reset_index(drop=True)


def save_embeddings(
    out_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    xc_id: np.ndarray,
    chunk_index: np.ndarray,
    species: np.ndarray,
    label_names: list[str],
    model_name: str,
    X_seq: np.ndarray | None = None,
) -> None:
    """埋め込みキャッシュを npz で保存。

    スキーマ:
      X            (N, D)        平均プーリング埋め込み（線形/MLPプローブ用）
      X_seq        (N, S, D)     系列埋め込み（attentiveプローブ用、任意）
      y            (N,) int      label_id
      xc_id        (N,) str      録音ID（録音単位集約・リーク確認用）
      chunk_index  (N,) int
      species      (N,) str
      label_names  (C,) str      label_id 昇順の種名
      model_name   str
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        xc_id=np.asarray(xc_id, dtype=object),
        chunk_index=np.asarray(chunk_index, dtype=np.int64),
        species=np.asarray(species, dtype=object),
        label_names=np.asarray(label_names, dtype=object),
        model_name=np.asarray(model_name, dtype=object),
    )
    if X_seq is not None:
        arrays["X_seq"] = np.asarray(X_seq, dtype=np.float32)
    np.savez_compressed(out_path, **arrays)


def load_embeddings(path: Path) -> dict:
    """save_embeddings で保存した npz を dict で返す。"""
    data = np.load(Path(path), allow_pickle=True)
    out = {
        "X": data["X"],
        "y": data["y"],
        "xc_id": data["xc_id"],
        "chunk_index": data["chunk_index"],
        "species": data["species"],
        "label_names": list(data["label_names"]),
        "model_name": str(data["model_name"]),
    }
    if "X_seq" in data.files:
        out["X_seq"] = data["X_seq"]
    return out
