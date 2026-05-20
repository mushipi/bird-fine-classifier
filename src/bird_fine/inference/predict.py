"""学習済みASTモデルで任意の音声ファイルを推論。

BirdNet後段として呼び出される想定: BirdNetが「カモ類」と判定した音声を
渡すと、本モデルが具体的な種を返す。

使い方:
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav --top-k 3
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav --model-dir models/ast-duck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import ASTFeatureExtractor, ASTForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_map(model_dir: Path) -> dict[int, str]:
    """学習時に保存した label_map.csv を読む。"""
    df = pd.read_csv(model_dir / "label_map.csv")
    return {int(row["label_id"]): row["species"] for _, row in df.iterrows()}


def split_into_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_duration: float,
    min_duration: float,
) -> list[np.ndarray]:
    """音声を固定長チャンクに分割。短すぎるものはパディング。"""
    chunk_samples = int(chunk_duration * sr)
    min_samples = int(min_duration * sr)
    n = len(audio)

    if n < min_samples:
        return []
    if n <= chunk_samples:
        padded = np.zeros(chunk_samples, dtype=audio.dtype)
        padded[:n] = audio
        return [padded]

    chunks: list[np.ndarray] = []
    for start in range(0, n, chunk_samples):
        end = start + chunk_samples
        if end <= n:
            chunks.append(audio[start:end])
        else:
            remaining = n - start
            if remaining >= min_samples:
                padded = np.zeros(chunk_samples, dtype=audio.dtype)
                padded[:remaining] = audio[start:n]
                chunks.append(padded)
    return chunks


def predict_audio(
    audio_path: Path,
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    id2label: dict[int, str],
    device: torch.device,
    chunk_duration: float,
    min_duration: float,
    top_k: int,
) -> dict:
    """音声ファイルを推論。チャンクごとの予測を平均してtop-Kを返す。"""
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)
    chunks = split_into_chunks(audio, sr, chunk_duration, min_duration)

    if not chunks:
        return {
            "error": f"音声が短すぎ ({len(audio)/sr:.2f}秒 < {min_duration}秒)",
            "predictions": [],
        }

    all_probs = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = feature_extractor(
                chunk, sampling_rate=sr, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(device)
            logits = model(input_values=input_values).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    top_indices = np.argsort(mean_probs)[::-1][:top_k]

    predictions = [
        {
            "rank": rank + 1,
            "species": id2label[int(idx)],
            "probability": float(mean_probs[idx]),
        }
        for rank, idx in enumerate(top_indices)
    ]

    return {
        "audio_file": str(audio_path),
        "duration_sec": float(len(audio) / sr),
        "n_chunks": len(chunks),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="入力音声ファイルパス")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="学習済みモデルディレクトリ。省略時はconfig.training.output_dir",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    pp = config["preprocessing"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    model_dir = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / train_cfg["output_dir"]
    if not model_dir.exists():
        print(f"[ERROR] {model_dir} が見つからない。先に train.py を実行して。")
        return

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] 音声ファイルが見つからない: {audio_path}")
        return

    top_k = args.top_k or int(eval_cfg.get("top_k", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] device: {device}")

    print(f"[LOAD] モデル: {model_dir}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_cfg["pretrained"])
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id2label = load_label_map(model_dir)

    print(f"[AUDIO] 推論: {audio_path}")
    result = predict_audio(
        audio_path=audio_path,
        model=model,
        feature_extractor=feature_extractor,
        id2label=id2label,
        device=device,
        chunk_duration=float(pp["chunk_duration_sec"]),
        min_duration=float(pp["min_chunk_duration_sec"]),
        top_k=top_k,
    )

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print(f"\n  音声長: {result['duration_sec']:.2f} 秒 ({result['n_chunks']} チャンク)")
    print(f"\n[INFO] Top-{top_k} 予測:")
    for p in result["predictions"]:
        bar = "█" * int(p["probability"] * 40)
        print(f"  {p['rank']}. {p['species']:25s} {p['probability']:6.2%}  {bar}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
