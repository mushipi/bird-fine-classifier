"""teacher非依存の録音単位 bootstrap CI 比較（拡大テスト対応）。

kd_compare_ci の teacher参照でKeyError(拡大テストの新録音がteacher probaに無い)を回避。
任意個の AST proba tag を録音単位 macro-f1 + 95%CI(録音クラスタ bootstrap)で並べ、
指定ペアの差(95%CI, 0を跨げば差なし)と、種別f1+録音support(test拡大の解像度)を出す。

実行:
  .venv/bin/python tools/probe_sweep/soup_ci.py \
    --tags Cv2BASEsoup Cv2KDsoup Cprod \
    --pairs Cv2KDsoup:Cv2BASEsoup Cv2KDsoup:Cprod Cv2BASEsoup:Cprod
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
B = 2000
SEED = 42


def load_rec(tag: str):
    d = np.load(PROBA / f"{tag}_test.npz", allow_pickle=True)
    duck = [str(x) for x in d["id2label"]]
    idx = {s: i for i, s in enumerate(duck)}
    y = np.array([idx[str(s)] for s in d["species"]])
    xc = np.array([str(x) for x in d["xc_id"]])
    ry, rp = aggregate_by_recording(d["proba"], y, xc)
    return ry, rp.argmax(1), duck


def f1_sample(true, pred, operative, idx):
    return f1_score(true[idx], pred[idx], labels=operative, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="kd:base 形式（kd-base の差）")
    args = ap.parse_args()

    recs = {t: load_rec(t) for t in args.tags}
    # 録音集合の整合（同一 test 前提）
    ref_ry = recs[args.tags[0]][0]
    duck = recs[args.tags[0]][2]
    for t in args.tags[1:]:
        assert np.array_equal(recs[t][0], ref_ry), f"{t} の録音集合が {args.tags[0]} とズレてる"
    operative = sorted(np.unique(ref_ry).tolist())
    R = len(ref_ry)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, R, R) for _ in range(B)]

    print(f"=== 録音単位 macro-F1（n={R}録音, {len(operative)}カモ）点推定 + 95%CI ===", flush=True)
    for t in args.tags:
        ry, pred, _ = recs[t]
        pt = f1_sample(ry, pred, operative, np.arange(R))
        bs = np.array([f1_sample(ry, pred, operative, ix) for ix in boot_idx])
        print(f"  {t:14s}: {pt:.4f}  95%CI[{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]", flush=True)

    for pair in args.pairs:
        kd, base = pair.split(":")
        ry = ref_ry
        pk = recs[kd][1]; pb = recs[base][1]
        diffs = np.array([f1_sample(ry, pk, operative, ix) - f1_sample(ry, pb, operative, ix) for ix in boot_idx])
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        ptd = f1_sample(ry, pk, operative, np.arange(R)) - f1_sample(ry, pb, operative, np.arange(R))
        sig = "有意" if (lo > 0 or hi < 0) else "★差なし(CIが0を含む)"
        print(f"\n=== {kd} − {base} 95%CI ===", flush=True)
        print(f"  点差={ptd:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  {sig}", flush=True)
        kr = (pk == ry); br = (pb == ry)
        print(f"  録音: {kd}正/{base}誤={int((kr&~br).sum())}  {kd}誤/{base}正={int((~kr&br).sum())}  不一致={int((pk!=pb).sum())}/{R}", flush=True)

    # 種別 f1 + 録音support（test拡大の解像度）
    print(f"\n=== 種別 f1（録音support付き, support昇順）===", flush=True)
    ry = ref_ry
    supp = {c: int((ry == c).sum()) for c in operative}
    perclass = {t: f1_score(ry, recs[t][1], labels=operative, average=None, zero_division=0) for t in args.tags}
    hdr = "  ".join(f"{t[:10]:>10s}" for t in args.tags)
    print(f"  {'species':22s} {'n_rec':>5s}  {hdr}", flush=True)
    for c in sorted(operative, key=lambda c: supp[c]):
        vals = "  ".join(f"{perclass[t][operative.index(c)]:>10.3f}" for t in args.tags)
        print(f"  {duck[c]:22s} {supp[c]:>5d}  {vals}", flush=True)


if __name__ == "__main__":
    main()
