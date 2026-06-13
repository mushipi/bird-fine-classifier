"""学習済み AST の test 各チャンク proba + xc_id/chunk_index/species をダンプ。

KD の相補性ゲート(Phase0)・CI評価(Phase3)用。evaluate.py のモデルロード/推論ループを踏襲。
出力: outputs/ast_proba/{tag}_test.npz (proba(N,C), xc_id, chunk_index, species, id2label)

実行（mainPC, 本体.venv）:
  .venv/bin/python tools/ast_eval_proba.py --model-dir models/ast-duck-v15 --tag v15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bird_fine.data.dataset import build_datasets, collate_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "ast_proba"))
    ap.add_argument("--config", default="config.yaml", help="設定ファイル（前処理/モデル基盤）")
    ap.add_argument("--splits-dir", default="data/splits", help="splits ディレクトリ（群/アーム別）")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, _ = build_datasets(
        splits_dir=PROJECT_ROOT / args.splits_dir,
        pretrained=model_cfg["pretrained"],
        project_root=PROJECT_ROOT,
        max_length=int(model_cfg.get("feature_extractor_max_length", 1024)),
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    model = ASTForAudioClassification.from_pretrained(args.model_dir).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"[load] {args.model_dir}  C={model.config.num_labels}  {args.split}={len(ds)}chunks", flush=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_values=batch["input_values"].to(device)).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    proba = np.concatenate(probs, axis=0)

    df = ds.df
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_{args.split}.npz"
    np.savez(
        out_path,
        proba=proba.astype(np.float32),
        xc_id=df["xc_id"].to_numpy(dtype=object),
        chunk_index=df["chunk_index"].to_numpy(dtype=np.int64),
        species=df["species"].to_numpy(dtype=object),
        id2label=np.asarray([id2label[i] for i in range(len(id2label))], dtype=object),
    )
    print(f"[OK] proba{proba.shape} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
