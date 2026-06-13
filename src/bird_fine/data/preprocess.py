"""data/raw 配下の音声を16kHz mono化して10秒チャンクに分割。

出力:
    data/processed/{Species_name}/XC{id}_chunk{i}.wav
    data/processed/chunks_index.csv  # 全チャンクのメタデータ

使い方:
    uv run python -m bird_fine.data.preprocess
    uv run python -m bird_fine.data.preprocess --overwrite
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path=None) -> dict:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_xc_id(audio_path: Path) -> str:
    """ファイル名からXC IDを抽出。例: 'XC123456 - Mallard.mp3' → 'XC123456'"""
    stem = audio_path.stem
    for token in stem.replace("_", " ").split():
        if token.startswith("XC") and token[2:].isdigit():
            return token
    return stem.split()[0] if stem else "unknown"


def _chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float,
    min_duration: float,
    overlap_ratio: float,
) -> list[np.ndarray]:
    """音声を固定長チャンクに分割。短すぎるものはパディング。"""
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
        chunks.append(padded)
        return chunks

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


def process_species(
    species_dir: Path,
    output_dir: Path,
    sample_rate: int,
    chunk_duration: float,
    min_duration: float,
    overlap_ratio: float,
    overwrite: bool,
) -> list[dict]:
    """1種ぶんのディレクトリを処理。チャンクメタデータのリストを返す。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    audio_files = sorted(
        p for p in species_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}
    )

    for audio_path in tqdm(audio_files, desc=f"  {species_dir.name}", leave=False):
        xc_id = _extract_xc_id(audio_path)
        try:
            audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"    [WARN] {audio_path.name}: {e}")
            continue

        chunks = _chunk_audio(audio, sample_rate, chunk_duration, min_duration, overlap_ratio)

        for i, chunk in enumerate(chunks):
            out_name = f"{xc_id}_chunk{i:03d}.wav"
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                records.append(
                    {
                        "species": species_dir.name,
                        "xc_id": xc_id,
                        "chunk_index": i,
                        "file_path": str(out_path.relative_to(PROJECT_ROOT)),
                        "duration_sec": chunk_duration,
                        "source_file": audio_path.name,
                    }
                )
                continue
            sf.write(out_path, chunk, sample_rate, subtype="PCM_16")
            records.append(
                {
                    "species": species_dir.name,
                    "xc_id": xc_id,
                    "chunk_index": i,
                    "file_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "duration_sec": chunk_duration,
                    "source_file": audio_path.name,
                }
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存チャンクを上書き再生成",
    )
    parser.add_argument("--config", default=None,
                        help="設定ファイル（省略時 config.yaml）。群別は config-<group>.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pp = config["preprocessing"]
    sample_rate = int(pp["sample_rate"])
    chunk_duration = float(pp["chunk_duration_sec"])
    min_duration = float(pp["min_chunk_duration_sec"])
    overlap_ratio = float(pp["overlap_ratio"])

    raw_dir = PROJECT_ROOT / config["download"]["output_dir"]
    processed_dir = PROJECT_ROOT / pp["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[ERROR] {raw_dir} が存在しない。先に download.py を実行して。")
        return

    species_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"[INFO] 対象: {len(species_dirs)} 種")
    print(f"[CFG]  {sample_rate}Hz mono / {chunk_duration}秒チャンク / overlap={overlap_ratio}")
    print("=" * 60)

    all_records: list[dict] = []
    for species_dir in species_dirs:
        out_dir = processed_dir / species_dir.name
        records = process_species(
            species_dir=species_dir,
            output_dir=out_dir,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            min_duration=min_duration,
            overlap_ratio=overlap_ratio,
            overwrite=args.overwrite,
        )
        all_records.extend(records)
        print(f"  [OK] {species_dir.name}: {len(records)} チャンク")

    index_path = processed_dir / "chunks_index.csv"
    fieldnames = ["species", "xc_id", "chunk_index", "file_path", "duration_sec", "source_file"]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 60)
    print(f"[INFO] 合計 {len(all_records)} チャンク")
    print(f"[NOTE] インデックス: {index_path}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
