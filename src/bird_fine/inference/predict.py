"""学習済みASTモデルで任意の音声ファイルを推論。

BirdNet後段として呼び出される想定: BirdNetが「カモ類」と判定した音声を
渡すと、本モデルが具体的な種を返す。

Energy-based OOD gate を内蔵しており、カモ科以外の音（OOD）を
energy スコアの閾値で弾いてから8種分類を行う。

  energy = T * logsumexp(logits / T)   高い = in-distribution
  energy < energy_threshold → "unknown" を返す（OOD 判定）

閾値・温度は species_taxonomy.yaml で管理する。

使い方:
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav --top-k 3
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav --model-dir models/ast-duck
    uv run python -m bird_fine.inference.predict --audio path/to/duck.wav --no-ood-gate
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


TAXONOMY_PATH = PROJECT_ROOT / "species_taxonomy.yaml"


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config(group: str = "duck") -> tuple[str | None, float | None, float]:
    """species_taxonomy.yaml の <group>.pipeline から推論設定を読む。

    推論モデルと OOD params の単一の真実。返り値 = (stage2_model, energy_threshold, energy_temperature)。
    threshold が None の場合は energy gate 無効。
    """
    if not TAXONOMY_PATH.exists():
        return None, None, 1.0
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    pipeline = taxonomy.get(group, {}).get("pipeline", {})
    stage2_model = pipeline.get("stage2_model")
    threshold = pipeline.get("energy_threshold")  # None の場合は gate 無効
    temperature = float(pipeline.get("energy_temperature", 1.0))
    return stage2_model, threshold, temperature


def load_display_groups(group: str = "duck") -> dict:
    """species_taxonomy.yaml の <group>.display_groups（内部クラス→複合表示）を読む。"""
    if not TAXONOMY_PATH.exists():
        return {}
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    return taxonomy.get(group, {}).get("display_groups", {}) or {}


def load_species_display() -> dict[str, dict]:
    """data/species_master.csv から 内部英名(正規化)→{ja, sci} を作る（和名/学名表示用）。"""
    csv = PROJECT_ROOT / "data" / "species_master.csv"
    out: dict[str, dict] = {}
    if not csv.exists():
        return out
    df = pd.read_csv(csv)
    for _, r in df.iterrows():
        info = {"ja": str(r.get("ja", "")), "sci": str(r.get("sci", ""))}
        for col in ("en_inat", "en_birdnet"):
            v = r.get(col)
            if pd.notna(v):
                out[str(v).replace(" ", "_")] = info
    return out


def resolve_display(internal: str, groups: dict, sp_disp: dict) -> tuple[str, str]:
    """内部英名 → (表示ラベル, 学名)。複合(groups)優先、なければ species_master の和名/学名。"""
    if internal in groups:
        g = groups[internal]
        return str(g.get("label", internal)), str(g.get("sci", ""))
    info = sp_disp.get(internal, {})
    return info.get("ja") or internal, info.get("sci", "")


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


def compute_energy(logits_np: np.ndarray, T: float = 1.0) -> float:
    """Energy score = T * logsumexp(logits / T)。高い = in-distribution。"""
    scaled = logits_np / T
    a = scaled.max()
    return float(T * (a + np.log(np.sum(np.exp(scaled - a)))))


def predict_audio(
    audio_path: Path,
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    id2label: dict[int, str],
    device: torch.device,
    chunk_duration: float,
    min_duration: float,
    top_k: int,
    energy_threshold: float | None = None,
    energy_temperature: float = 1.0,
) -> dict:
    """音声ファイルを推論。チャンクごとの予測を平均して top-K を返す。

    energy_threshold が設定されている場合、energy スコアが閾値未満なら
    OOD と判定して predictions=[] / ood_rejected=True を返す。
    "other" ラベル（学習時に追加した第9クラス）は推論時に除外する。
    """
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)
    chunks = split_into_chunks(audio, sr, chunk_duration, min_duration)

    if not chunks:
        return {
            "error": f"音声が短すぎ ({len(audio)/sr:.2f}秒 < {min_duration}秒)",
            "predictions": [],
        }

    # "other" ラベルを除いた有効ラベル一覧
    valid_labels = {k: v for k, v in id2label.items() if v != "other"}
    n_target = len(valid_labels)

    all_probs: list[np.ndarray] = []
    all_energies: list[float] = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
            logits = model(input_values=inputs["input_values"].to(device)).logits
            logits_np = logits.cpu().numpy()[0]

            all_energies.append(compute_energy(logits_np, energy_temperature))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)

    mean_energy = float(np.mean(all_energies))
    mean_probs = np.mean(all_probs, axis=0)

    base = {
        "audio_file": str(audio_path),
        "duration_sec": float(len(audio) / sr),
        "n_chunks": len(chunks),
        "energy_score": round(mean_energy, 4),
        "energy_threshold": energy_threshold,
    }

    # OOD gate
    if energy_threshold is not None and mean_energy < energy_threshold:
        return {**base, "ood_rejected": True, "predictions": []}

    # 8種（有効ラベル）のみで top-K を構成
    target_probs = mean_probs[:n_target]
    top_indices = np.argsort(target_probs)[::-1][:top_k]

    predictions = [
        {
            "rank": rank + 1,
            "species": valid_labels[int(idx)],
            "probability": float(target_probs[idx]),
        }
        for rank, idx in enumerate(top_indices)
    ]

    return {**base, "ood_rejected": False, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="入力音声ファイルパス")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="学習済みモデルディレクトリ。省略時はconfig.training.output_dir",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--group", default="duck",
                        help="species_taxonomy.yaml のグループ（推論モデル/OOD params の参照先）")
    parser.add_argument(
        "--no-ood-gate",
        action="store_true",
        help="OOD gate を無効化（閾値なしで分類のみ実行）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="結果を1行JSONで stdout 出力（process.py 等のプログラム連携用。人間向け print は抑制）",
    )
    args = parser.parse_args()

    config = load_config()
    pp = config["preprocessing"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    # 推論モデル/OOD params の単一の真実 = species_taxonomy.yaml の <group>.pipeline。
    # training.output_dir は学習の保存先専用（taxonomy 未設定時のフォールバックのみ）。
    stage2_model, energy_threshold, energy_temperature = load_pipeline_config(args.group)
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif stage2_model:
        model_dir = PROJECT_ROOT / stage2_model
    else:
        model_dir = PROJECT_ROOT / train_cfg["output_dir"]
    if not model_dir.exists():
        print(f"[ERROR] {model_dir} が見つからない。先に train.py を実行して。")
        return

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] 音声ファイルが見つからない: {audio_path}")
        return

    top_k = args.top_k or int(eval_cfg.get("top_k", 3))
    gate_off_reason = None
    if args.no_ood_gate:
        energy_threshold = None
        gate_off_reason = "--no-ood-gate"
    elif energy_threshold is None:
        gate_off_reason = f"taxonomy[{args.group}].pipeline に energy_threshold 未設定"

    import sys
    # --json 時は診断ログを stderr に流し、stdout は1行JSONだけにする（subprocess 連携用）。
    log = (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[DEV] device: {device}")

    log(f"[LOAD] モデル: {model_dir}")
    max_length = int(model_cfg.get("feature_extractor_max_length", 1024))
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        model_cfg["pretrained"], max_length=max_length
    )
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id2label = load_label_map(model_dir)

    if energy_threshold is not None:
        log(f"[OOD] energy gate 有効: threshold={energy_threshold} / T={energy_temperature}")
    else:
        log(f"[OOD] energy gate 無効（{gate_off_reason}）")

    log(f"[AUDIO] 推論: {audio_path}")
    result = predict_audio(
        audio_path=audio_path,
        model=model,
        feature_extractor=feature_extractor,
        id2label=id2label,
        device=device,
        chunk_duration=float(pp["chunk_duration_sec"]),
        min_duration=float(pp["min_chunk_duration_sec"]),
        top_k=top_k,
        energy_threshold=energy_threshold,
        energy_temperature=energy_temperature,
    )

    # 表示ラベル（複合・和名/学名）を予測に付与
    groups = load_display_groups(args.group)
    sp_disp = load_species_display()
    for p in result.get("predictions", []):
        label, sci = resolve_display(p["species"], groups, sp_disp)
        p["label"], p["sci"] = label, sci

    # --json: 機械可読な1行を stdout へ
    if args.json:
        preds = result.get("predictions", [])
        out = {
            "group": args.group,
            "model": model_dir.name,
            "audio": str(audio_path),
            "energy_score": result.get("energy_score"),
            "energy_threshold": energy_threshold,
            "ood_rejected": bool(result.get("ood_rejected")),
            "error": result.get("error"),
            "predictions": preds,
            "top": preds[0] if preds else None,
        }
        import json as _json
        print(_json.dumps(out, ensure_ascii=False))
        return

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print(f"\n  音声長: {result['duration_sec']:.2f} 秒 ({result['n_chunks']} チャンク)")
    print(f"  energy score: {result['energy_score']:.4f}", end="")
    if energy_threshold is not None:
        print(f"  (閾値: {energy_threshold})", end="")
    print()

    if result.get("ood_rejected"):
        print(f"\n[OOD] ⚠ 非対象種と判定（energy {result['energy_score']:.4f} < {energy_threshold}）")
        print("  → BirdNet の誤検出またはカモ科以外の音声の可能性があります")
        return

    print(f"\n[INFO] Top-{top_k} 予測:")
    for p in result["predictions"]:
        bar = "█" * int(p["probability"] * 40)
        disp = f"{p['label']}（{p['sci']}）" if p['sci'] else p['label']
        print(f"  {p['rank']}. {disp:28s} {p['probability']:6.2%}  {bar}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
