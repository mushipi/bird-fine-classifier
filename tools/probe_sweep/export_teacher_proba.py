"""Perch プローブ（複数arm平均）の教師ソフトラベルを各 split のチャンク単位で出力。

KD 蒸留用。linear/mlp/sklearn の proba を平均 → Perch operative 10カモ列を **種名で AST の
カモ index(0..9) に写像** → 行ごとに10種で再正規化 → data/embeddings/teacher_proba/{split}.npz。

AST v15 のカモ index 順（models/ast-duck-v15/config.json id2label の 0..9, 10=other）に
教師 proba の列を合わせるので、学習時は AST ロジットのカモ10列とそのまま対応する。

実行（mainPC, 本体.venv）:
  .venv/bin/python tools/probe_sweep/export_teacher_proba.py --splits train val test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))          # run_sweep 再利用
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_sweep as rs  # noqa: E402  (train_torch/torch_proba/fit_sklearn/sklearn_proba_full)
from bird_fine.embeddings import io_utils  # noqa: E402
from bird_fine.training.train_probe import standardize_apply, standardize_fit  # noqa: E402

# 教師の各 arm 構成（sweep の強構成。averaged teacher）
LINEAR_WD = 0.0
MLP_WD = 5e-3
SK_C = 0.01


def ast_duck_order(ast_dir: Path) -> list[str]:
    """AST config.json の id2label から 'other' を除くカモ種を index 順で返す。"""
    cfg = json.loads((ast_dir / "config.json").read_text())
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    return [id2label[i] for i in sorted(id2label) if id2label[i] != "other"]


def perch_name_to_col(label_names: list[str]) -> dict[str, int]:
    """Perch の label_names（index=列）から 種名→列 を作る。"""
    return {name: i for i, name in enumerate(label_names)}


def avg_proba(lin, mlp, clf, Xs, num_classes, device) -> np.ndarray:
    return (rs.torch_proba(lin, Xs, device)
            + rs.torch_proba(mlp, Xs, device)
            + rs.sklearn_proba_full(clf, Xs, num_classes)) / 3.0


def oof_train_proba(Xtr_s, ytr, xc_tr, Xva_s, yva, num_classes, operative,
                    device, folds: int, seed: int = 42) -> np.ndarray:
    """train の教師 proba を録音単位 K-fold OOF で生成（過学習した過信ラベルを避ける）。

    各 fold: 残り fold で 3 arm を学習（torch の epoch 選択にはグローバル val を使用）→
    held-out fold を予測。fold 割当は録音(xc_id)単位＝同一録音内のチャンク漏洩を防ぐ。
    """
    rng = np.random.RandomState(seed)
    recs = np.array(sorted({str(x) for x in xc_tr}))
    rng.shuffle(recs)
    fold_of = {r: i % folds for i, r in enumerate(recs)}
    fid = np.array([fold_of[str(x)] for x in xc_tr])
    out = np.zeros((len(ytr), num_classes), dtype=np.float64)
    for k in range(folds):
        tr = np.where(fid != k)[0]; te = np.where(fid == k)[0]
        lin, _ = rs.train_torch("linear", Xtr_s[tr], ytr[tr], Xva_s, yva, num_classes, operative, LINEAR_WD, device)
        mlp, _ = rs.train_torch("mlp", Xtr_s[tr], ytr[tr], Xva_s, yva, num_classes, operative, MLP_WD, device)
        clf = rs.fit_sklearn(Xtr_s[tr], ytr[tr], SK_C)
        out[te] = avg_proba(lin, mlp, clf, Xtr_s[te], num_classes, device)
        print(f"    [oof] fold {k+1}/{folds}: fit={len(tr)} pred={len(te)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", default=str(PROJECT_ROOT / "data" / "embeddings" / "perch"))
    ap.add_argument("--ast-dir", default=str(PROJECT_ROOT / "models" / "ast-duck-v15"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "embeddings" / "teacher_proba"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--oof-folds", type=int, default=5,
                    help="train の教師 proba を録音単位 K-fold OOF で生成（0で無効=full-fit）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(args.emb_dir)
    tr = io_utils.load_embeddings(emb_dir / "train.npz")
    label_names = tr["label_names"]
    num_classes = len(label_names)
    Xtr = tr["X"]; ytr = tr["y"]
    mean, std = standardize_fit(Xtr)
    Xtr_s = standardize_apply(Xtr, mean, std)
    operative = sorted(np.unique(ytr).tolist())

    # 教師 3 arm を train で学習（val は torch の epoch 選択にだけ使う）
    va = io_utils.load_embeddings(emb_dir / "val.npz")
    Xva_s = standardize_apply(va["X"], mean, std); yva = va["y"]
    print(f"[teacher] train {len(ytr)} / classes {num_classes}(運用{len(operative)})  device={device}", flush=True)
    lin, _ = rs.train_torch("linear", Xtr_s, ytr, Xva_s, yva, num_classes, operative, LINEAR_WD, device)
    mlp, _ = rs.train_torch("mlp", Xtr_s, ytr, Xva_s, yva, num_classes, operative, MLP_WD, device)
    clf = rs.fit_sklearn(Xtr_s, ytr, SK_C)
    print(f"[teacher] arms ready: linear(wd={LINEAR_WD}) mlp(wd={MLP_WD}) sklearn(C={SK_C})", flush=True)

    # 列写像: AST カモ index(0..9) -> Perch 列
    duck_order = ast_duck_order(Path(args.ast_dir))
    p2c = perch_name_to_col(label_names)
    missing = [d for d in duck_order if d not in p2c]
    assert not missing, f"AST カモ種が Perch label_names に無い: {missing}"
    perch_cols = [p2c[d] for d in duck_order]  # 長さ10
    print(f"[map] AST duck order -> perch cols: " +
          ", ".join(f"{d}->{c}" for d, c in zip(duck_order, perch_cols)), flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        d = io_utils.load_embeddings(emb_dir / f"{split}.npz")
        Xs = standardize_apply(d["X"], mean, std)
        if split == "train" and args.oof_folds > 0:
            print(f"[train] OOF {args.oof_folds}-fold（録音単位）で教師 proba を生成", flush=True)
            p = oof_train_proba(Xtr_s, ytr, tr["xc_id"], Xva_s, yva,
                                num_classes, operative, device, args.oof_folds)
        else:
            p = avg_proba(lin, mlp, clf, Xs, num_classes, device)
        # AST カモ10列へ写像 → 行正規化（10種上の分布に）
        t = p[:, perch_cols].astype(np.float64)
        t = t / np.clip(t.sum(axis=1, keepdims=True), 1e-12, None)
        out_path = out_dir / f"{split}.npz"
        np.savez(
            out_path,
            teacher_proba=t.astype(np.float32),
            xc_id=d["xc_id"], chunk_index=d["chunk_index"],
            duck_order=np.asarray(duck_order, dtype=object),
        )
        print(f"  [OK] {split}: teacher{t.shape} -> {out_path}", flush=True)

    # 種名写像を CSV でも残す（監査用）
    with open(out_dir / "duck_order.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ast_duck_index", "species", "perch_col"])
        for i, (dname, c) in enumerate(zip(duck_order, perch_cols)):
            w.writerow([i, dname, c])
    print(f"[done] duck_order.csv 保存", flush=True)


if __name__ == "__main__":
    main()
