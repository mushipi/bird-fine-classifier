"""ディレクトリ配下の wav を Perch 埋め込み化（label_map 不要・OOD監査用）。

data/ood_processed/<tier>/<species>/*.wav を走査し、各 3秒チャンク wav を Perch 埋め込み(1536)へ。
OOD のリーク率/疑陽性監査のため、in-dist の label_map に縛られず species/tier を記録して保存。

隔離環境（tools/perch_embed/.venv: TF + perch-hoplite）で実行:
  .venv/bin/python tools/perch_embed/extract_perch_dir.py --root data/ood_processed \
    --out data/embeddings/perch_ood/ood.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_perch as ep  # load_perch / embed_chunk / PERCH_SR を再利用  # noqa: E402

import librosa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="走査ルート（<tier>/<species>/*.wav 想定）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    wavs = sorted(root.rglob("*.wav"))
    print(f"[ood-embed] {len(wavs)} wav under {root}", flush=True)
    model = ep.load_perch()

    feats, tiers, specs, files = [], [], [], []
    miss = 0
    for i, w in enumerate(wavs):
        rel = w.relative_to(root)
        tier = rel.parts[0] if len(rel.parts) >= 2 else "?"
        species = rel.parts[1] if len(rel.parts) >= 3 else (rel.parts[0] if len(rel.parts) == 2 else "?")
        try:
            audio, _ = librosa.load(str(w), sr=ep.PERCH_SR, mono=True)
            feats.append(ep.embed_chunk(model, audio))
            tiers.append(tier); specs.append(species); files.append(str(rel))
        except Exception as e:  # noqa: BLE001
            miss += 1
            if miss <= 5:
                print(f"  [WARN] {w.name}: {e}", flush=True)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{len(wavs)}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=np.asarray(feats, dtype=np.float32),
        tier=np.asarray(tiers, dtype=object),
        species=np.asarray(specs, dtype=object),
        path=np.asarray(files, dtype=object),
    )
    print(f"[OK] {len(feats)} embedded (miss={miss}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
