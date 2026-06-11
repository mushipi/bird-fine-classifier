"""AST モデル soup: 複数チェックポイントの重みを平均して1つの頑健なモデルにする。

同一 pretrained から fine-tune した同一アーキの複数 seed を平均（model soup）。
浮動小数テンソルのみ平均、整数/buffer は先頭を流用。label_map.csv も先頭からコピー。

実行:
  .venv/bin/python tools/soup_ast.py --out models/ast-duck-C-kd-soup \
    models/ast-duck-C-kd models/ast-duck-C-kd-s1 models/ast-duck-C-kd-s2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import ASTForAudioClassification


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+", help="平均するモデルディレクトリ（2個以上）")
    args = ap.parse_args()
    assert len(args.dirs) >= 2, "2個以上を指定して"

    sds = [ASTForAudioClassification.from_pretrained(d).state_dict() for d in args.dirs]
    n = len(sds)
    avg = {}
    for k, v in sds[0].items():
        if v.is_floating_point():
            avg[k] = sum(sd[k].float() for sd in sds) / n
            avg[k] = avg[k].to(v.dtype)
        else:
            avg[k] = v
    model = ASTForAudioClassification.from_pretrained(args.dirs[0])
    model.load_state_dict(avg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    lm = Path(args.dirs[0]) / "label_map.csv"
    if lm.exists():
        shutil.copy(lm, out / "label_map.csv")
    print(f"[soup] {n} models -> {out}  (params averaged)", flush=True)


if __name__ == "__main__":
    main()
