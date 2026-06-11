"""BirdAVES-biox-large 埋め込み抽出ドライバ。

隔離環境（tools/aves_embed/.venv: CPU torch + esp-aves）で実行。
split CSV の processed 16kHz チャンク（file_path）を読み、埋め込みを
data/embeddings/birdaves/{split}.npz にキャッシュする。

実行例:
  cd tools/aves_embed
  .venv/bin/python extract_birdaves.py --splits train val test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import librosa  # noqa: E402
from bird_fine.embeddings import io_utils  # noqa: E402
from bird_fine.embeddings.birdaves import BirdAVESExtractor, AVES_SR  # noqa: E402

WEIGHTS = REPO_ROOT / "references" / "weights" / "birdaves"
CONFIG = WEIGHTS / "birdaves-biox-large.torchaudio.model_config.json"
MODEL = WEIGHTS / "birdaves-biox-large.torchaudio.pt"


def _load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != AVES_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AVES_SR)
    return audio


def extract_split(extractor: BirdAVESExtractor, df) -> dict:
    means: list[np.ndarray] = []
    seqs: list[np.ndarray] = []
    ys, xcs, cidx, specs = [], [], [], []
    missing = 0
    for _, row in df.iterrows():
        p = io_utils.normalize_rel_path(row["file_path"])
        if not p.exists():
            missing += 1
            continue
        mean, seq = extractor.embed(_load_wav(p))
        means.append(mean); seqs.append(seq)
        ys.append(extractor_label(row)); xcs.append(row["xc_id"])
        cidx.append(int(row["chunk_index"])); specs.append(row["species"])
    return dict(
        X=np.asarray(means, dtype=np.float32) if means else np.zeros((0, 1024), np.float32),
        X_seq=np.asarray(seqs, dtype=np.float32) if seqs else None,
        y=np.asarray(ys, dtype=np.int64),
        xc_id=np.asarray(xcs, dtype=object),
        chunk_index=np.asarray(cidx, dtype=np.int64),
        species=np.asarray(specs, dtype=object),
        missing=missing,
    )


# label_map をクロージャで持たせる代わりにグローバルを使う簡易実装
_LABEL_MAP: dict[str, int] = {}


def extractor_label(row) -> int:
    return _LABEL_MAP[row["species"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", default=str(REPO_ROOT / "data" / "splits"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "embeddings" / "birdaves"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--seq-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    help="auto/cuda/cpu。auto は cuda 利用可なら cuda")
    args = ap.parse_args()

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    if not MODEL.exists():
        print(f"[ERROR] {MODEL} なし。BirdAVES重みを references/weights/birdaves に置いて。")
        return

    splits_dir = Path(args.splits_dir)
    global _LABEL_MAP
    _LABEL_MAP = io_utils.load_label_map(splits_dir)
    label_names = io_utils.label_names_from_map(_LABEL_MAP)

    print(f"[load] BirdAVES-biox-large (device={device})...", flush=True)
    extractor = BirdAVESExtractor(CONFIG, MODEL, device=device, seq_tokens=args.seq_tokens)
    print("[ok] loaded", flush=True)

    for split in args.splits:
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            print(f"[SKIP] {csv} なし", flush=True)
            continue
        df = io_utils.read_split(csv)
        print(f"[{split}] {len(df)} chunks 抽出中...", flush=True)
        res = extract_split(extractor, df)
        out_path = Path(args.out_dir) / f"{split}.npz"
        io_utils.save_embeddings(
            out_path, res["X"], res["y"], res["xc_id"], res["chunk_index"],
            res["species"], label_names, "birdaves_biox_large", X_seq=res["X_seq"],
        )
        print(f"  [OK] mean{res['X'].shape} seq{None if res['X_seq'] is None else res['X_seq'].shape}"
              f" -> {out_path} (missing={res['missing']})", flush=True)


if __name__ == "__main__":
    main()
