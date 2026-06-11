"""CNN KD グリッドの集計: 各 (λ,T) を val 録音単位f1で選定し test を報告。

outputs/ast_proba/{tag}_{val,test}.npz を読み、録音単位 macro-F1 を計算。
val でランキング（test選択バイアス回避）、best-by-val の test と base 比較、CI は kd_compare_ci 併用。

実行:
  .venv/bin/python tools/probe_sweep/cnn_kd_grid_eval.py --base CNNbase --tags CNN_l1_t2 CNN_l2_t2 ...
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


def rec_f1(tag: str, split: str) -> float:
    d = np.load(PROBA / f"{tag}_{split}.npz", allow_pickle=True)
    duck = [str(x) for x in d["id2label"]]
    idx = {s: i for i, s in enumerate(duck)}
    y = np.array([idx[str(s)] for s in d["species"]])
    ry, rp = aggregate_by_recording(d["proba"], y, np.array([str(x) for x in d["xc_id"]]))
    op = sorted(np.unique(ry).tolist())
    return f1_score(ry, rp.argmax(1), labels=op, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="CNNbase")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    base_val = rec_f1(args.base, "val"); base_test = rec_f1(args.base, "test")
    rows = []
    for t in args.tags:
        try:
            rows.append((t, rec_f1(t, "val"), rec_f1(t, "test")))
        except FileNotFoundError:
            print(f"  [skip] {t}: proba 無し")
    rows.sort(key=lambda r: r[1], reverse=True)  # val でランキング

    print(f"\n=== CNN KD グリッド（録音単位f1, val降順）===", flush=True)
    print(f"  {'tag':16s} {'val_f1':>8s} {'test_f1':>8s} {'Δtest_vs_base':>14s}", flush=True)
    print(f"  {'base(λ0)':16s} {base_val:8.4f} {base_test:8.4f} {0.0:+14.4f}", flush=True)
    for t, v, te in rows:
        print(f"  {t:16s} {v:8.4f} {te:8.4f} {te - base_test:+14.4f}", flush=True)

    best = rows[0]
    print(f"\n[best-by-val] {best[0]}: val={best[1]:.4f} test={best[2]:.4f} "
          f"(base test {base_test:.4f} を {best[2]-base_test:+.4f})", flush=True)
    print(f"  → 統計判定は: kd_compare_ci.py --base {args.base} --kd {best[0]}", flush=True)


if __name__ == "__main__":
    main()
