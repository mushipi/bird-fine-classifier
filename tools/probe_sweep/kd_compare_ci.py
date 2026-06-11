"""KD実験の録音単位 bootstrap CI 比較（compare_runs_ci の方法論を proba 出力に適用）。

C-base(蒸留なし) vs C-kd(蒸留) を同一 test 録音上で比較。録音単位 macro-F1 の点推定+95%CI、
ペア差(KD−base)の95%CI（0を含めば「差なし」）。参考に teacher / ensemble も併記。

集約は softmax 平均（eval_probe.aggregate_by_recording）。リサンプルは録音単位（クラスタ bootstrap）。

実行:
  .venv/bin/python tools/probe_sweep/kd_compare_ci.py --base Cbase --kd Ckd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bird_fine.training.eval_probe import aggregate_by_recording  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

B = 2000
SEED = 42


def load_ast(tag: str):
    d = np.load(PROJECT_ROOT / "outputs" / "ast_proba" / f"{tag}_test.npz", allow_pickle=True)
    return d


def rec_pred_true(proba, species, xc_id, duck_order):
    """species→duck index を true に、softmax平均集約の argmax を pred に。"""
    duck_idx = {d: i for i, d in enumerate(duck_order)}
    y = np.array([duck_idx[s] for s in species])
    ry, rp = aggregate_by_recording(proba, y, xc_id)   # 録音単位（xc ソート順）
    return ry, rp.argmax(1)


def f1_sample(true, pred, operative, idx):
    return f1_score(true[idx], pred[idx], labels=operative, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Cbase")
    ap.add_argument("--kd", default="Ckd")
    ap.add_argument("--teacher", default=str(PROJECT_ROOT / "data" / "embeddings" / "teacher_proba" / "test.npz"))
    args = ap.parse_args()

    a = load_ast(args.base)
    duck_order = [str(x) for x in a["id2label"]]        # 10カモ（other無し前提）
    species = np.array([str(s) for s in a["species"]])
    xc = np.array([str(x) for x in a["xc_id"]])

    # base / kd の録音単位 true/pred（同一 test → 同一録音集合・同順）
    ry, pb = rec_pred_true(a["proba"], species, xc, duck_order)
    k = load_ast(args.kd)
    ryk, pk = rec_pred_true(k["proba"], np.array([str(s) for s in k["species"]]),
                            np.array([str(x) for x in k["xc_id"]]), duck_order)
    assert np.array_equal(ry, ryk), "base と kd で録音集合がズレている"

    # teacher（参考）: (xc,chunk) で proba を AST 行順に揃え不要、独立に集約
    t = np.load(args.teacher, allow_pickle=True)
    tdo = [str(x) for x in t["duck_order"]]
    assert tdo == duck_order, "teacher と AST のカモ順が不一致"
    # teacher proba を (xc,chunk) で AST test の行順に並べて集約する（true は AST の species 由来）
    a_keys = list(zip(xc, [int(c) for c in a["chunk_index"]]))
    tk = {(str(x), int(c)): i for i, (x, c) in enumerate(zip(t["xc_id"], t["chunk_index"]))}
    tperm = [tk[kk] for kk in a_keys]
    tproba = t["teacher_proba"][tperm]
    duck_idx = {d: i for i, d in enumerate(duck_order)}
    yy = np.array([duck_idx[s] for s in species])
    ryt, rpt = aggregate_by_recording(tproba, yy, xc)
    pt = rpt.argmax(1)
    # ensemble α=0.4（Phase0 best）
    rae, rpe = aggregate_by_recording(0.4 * a["proba"] + 0.6 * tproba, yy, xc)
    pe = rpe.argmax(1)

    operative = sorted(np.unique(ry).tolist())
    R = len(ry)
    rng = np.random.default_rng(SEED)
    idx_all = np.arange(R)

    def ci(pred):
        pt_ = f1_sample(ry, pred, operative, idx_all)
        bs = np.array([f1_sample(ry, pred, operative, rng.integers(0, R, R)) for _ in range(B)])
        return pt_, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"=== 録音単位 macro-F1（n={R}録音, {len(operative)}カモ）点推定 + 95%CI ===", flush=True)
    for name, pred in [("C-base(蒸留なし)", pb), ("C-kd(蒸留)", pk),
                       ("teacher(参考)", pt), ("ensemble α0.4(参考)", pe)]:
        p, lo, hi = ci(pred)
        print(f"  {name:18s}: {p:.4f}  95%CI[{lo:.4f}, {hi:.4f}]", flush=True)

    print(f"\n=== ペア差 KD−base（同一録音 resample）95%CI ===", flush=True)
    rng2 = np.random.default_rng(SEED)
    diffs = []
    for _ in range(B):
        s = rng2.integers(0, R, R)
        diffs.append(f1_sample(ry, pk, operative, s) - f1_sample(ry, pb, operative, s))
    lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    pt_diff = f1_sample(ry, pk, operative, idx_all) - f1_sample(ry, pb, operative, idx_all)
    sig = "有意" if (lo > 0 or hi < 0) else "★差なし(CIが0を含む)"
    print(f"  KD−base: 点差={pt_diff:+.4f}  95%CI[{lo:+.4f}, {hi:+.4f}]  {sig}", flush=True)
    # 録音単位の改善内訳
    base_right = (pb == ry); kd_right = (pk == ry)
    print(f"  録音: KD正/base誤={int((kd_right&~base_right).sum())}  "
          f"KD誤/base正={int((~kd_right&base_right).sum())}  "
          f"不一致={int((pb!=pk).sum())}/{R}", flush=True)


if __name__ == "__main__":
    main()
