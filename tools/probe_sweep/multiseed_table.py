"""多seed AST KD の集計: seed ごとに base / KD の test 録音単位f1 と差を出す。

KD>base が seed をまたいで一貫するか（運任せを殴れているか）を見る。soup 等の追加 tag も併記。
各 tag は outputs/ast_proba/{tag}_test.npz（ast_eval_proba 出力）。

実行:
  .venv/bin/python tools/probe_sweep/multiseed_table.py \
    --base Cbase Cbase_s1 Cbase_s2 --kd Ckd Ckd_s1 Ckd_s2 --extra KDsoup
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bird_fine.training.eval_probe import aggregate_by_recording  # noqa: E402

PROBA = PROJECT_ROOT / "outputs" / "ast_proba"


def rec_f1(tag: str) -> float:
    d = np.load(PROBA / f"{tag}_test.npz", allow_pickle=True)
    duck = [str(x) for x in d["id2label"]]
    idx = {s: i for i, s in enumerate(duck)}
    y = np.array([idx[str(s)] for s in d["species"]])
    ry, rp = aggregate_by_recording(d["proba"], y, np.array([str(x) for x in d["xc_id"]]))
    op = sorted(np.unique(ry).tolist())
    return f1_score(ry, rp.argmax(1), labels=op, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--kd", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    assert len(args.base) == len(args.kd), "base と kd の数を揃えて"

    print("=== 多seed AST KD（test 録音単位 f1）===", flush=True)
    print(f"  {'seed':6s} {'base':>8s} {'KD':>8s} {'KD-base':>9s}", flush=True)
    diffs, bases, kds = [], [], []
    for i, (b, k) in enumerate(zip(args.base, args.kd)):
        fb, fk = rec_f1(b), rec_f1(k)
        bases.append(fb); kds.append(fk); diffs.append(fk - fb)
        print(f"  {('s'+str(i)):6s} {fb:8.4f} {fk:8.4f} {fk-fb:+9.4f}", flush=True)
    bases, kds, diffs = map(np.array, (bases, kds, diffs))
    print(f"  {'mean':6s} {bases.mean():8.4f} {kds.mean():8.4f} {diffs.mean():+9.4f}", flush=True)
    print(f"  {'std':6s} {bases.std():8.4f} {kds.std():8.4f} {diffs.std():9.4f}", flush=True)
    npos = int((diffs > 0).sum())
    print(f"\n  KD>base が {npos}/{len(diffs)} seed で成立"
          f"（一貫性: {'全seedでKD優位' if npos==len(diffs) else '一部のみ'}）", flush=True)
    if args.extra:
        print("\n=== 追加（soup等）===", flush=True)
        for t in args.extra:
            try:
                print(f"  {t:16s}: {rec_f1(t):.4f}  (base平均 {bases.mean():.4f} を {rec_f1(t)-bases.mean():+.4f})", flush=True)
            except FileNotFoundError:
                print(f"  {t}: proba 無し")


if __name__ == "__main__":
    main()
