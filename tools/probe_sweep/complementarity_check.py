"""Phase0 相補性ゲート: AST(生徒) と Perch教師 が相補的か測る。

蒸留の上限 = 両者アンサンブル。相補性が無ければ蒸留は無駄。リークに頑健な
「予測不一致率」「オラクル利得(どちらかが正解な率)」を主信号、αブレンドの録音単位f1を併記。

入力:
  outputs/ast_proba/{ast_tag}_test.npz   (proba(N,Cast), xc_id, chunk_index, species, id2label)
  data/embeddings/teacher_proba/test.npz (teacher_proba(N,10), xc_id, chunk_index, duck_order)
出力: stdout + outputs/probe_sweep/complementarity_{ast_tag}.csv

実行:
  .venv/bin/python tools/probe_sweep/complementarity_check.py --ast-tag v15
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bird_fine.training.eval_probe import aggregate_by_recording, metrics_block  # noqa: E402


def f1_of(proba, y, xc_id, operative):
    ry, rp = aggregate_by_recording(proba, y, xc_id)
    m, _ = metrics_block(ry, rp, operative)
    return m["f1_macro"], ry, rp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ast-tag", default="v15")
    ap.add_argument("--ast-proba", default=None)
    ap.add_argument("--teacher", default=str(PROJECT_ROOT / "data" / "embeddings" / "teacher_proba" / "test.npz"))
    args = ap.parse_args()

    ast_path = Path(args.ast_proba) if args.ast_proba else \
        PROJECT_ROOT / "outputs" / "ast_proba" / f"{args.ast_tag}_test.npz"
    a = np.load(ast_path, allow_pickle=True)
    t = np.load(args.teacher, allow_pickle=True)

    duck_order = [str(x) for x in t["duck_order"]]          # AST カモ index 0..9 の種名
    ast_labels = [str(x) for x in a["id2label"]]            # AST 全クラス名（末尾 other 等）
    # AST のカモ10列を duck_order 順で取り出し（id2label と duck_order は同源だが念のため名で引く）
    ast_col = {n: i for i, n in enumerate(ast_labels)}
    duck_cols = [ast_col[d] for d in duck_order]

    # (xc_id, chunk_index) で teacher を AST 行順に整列
    def keyed(npz):
        return {(str(x), int(c)): i for i, (x, c) in enumerate(zip(npz["xc_id"], npz["chunk_index"]))}
    tk = keyed(t)
    a_keys = list(zip([str(x) for x in a["xc_id"]], [int(c) for c in a["chunk_index"]]))
    order = [tk[k] for k in a_keys]  # KeyError が出れば整列不能（=チャンク不一致）

    ast_proba_full = a["proba"]
    species = np.array([str(s) for s in a["species"]])
    teacher = t["teacher_proba"][order]                    # (N,10) AST行順

    # カモ10種チャンクのみ（other/OOD を除外）。true = duck_order の index
    duck_set = {d: i for i, d in enumerate(duck_order)}
    mask = np.array([s in duck_set for s in species])
    y = np.array([duck_set.get(s, -1) for s in species])[mask]
    xc = np.array([str(x) for x in a["xc_id"]])[mask]

    ast10 = ast_proba_full[mask][:, duck_cols].astype(np.float64)
    ast10 = ast10 / np.clip(ast10.sum(1, keepdims=True), 1e-12, None)   # other を捨てて10種で再正規化
    tea10 = teacher[mask].astype(np.float64)
    operative = sorted(np.unique(y).tolist())

    n = len(y)
    print(f"[gate] duck chunks={n} (other除外)  recordings={len(np.unique(xc))}  運用{len(operative)}種", flush=True)

    # --- 単体 & アンサンブル(α) 録音単位 f1 ---
    f_ast, ry, _ = f1_of(ast10, y, xc, operative)
    f_tea, _, _ = f1_of(tea10, y, xc, operative)
    best = (-1.0, None)
    rows = []
    for al in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        ens = al * ast10 + (1 - al) * tea10
        f, _, _ = f1_of(ens, y, xc, operative)
        rows.append(("ensemble", al, f))
        if f > best[0]:
            best = (f, al)

    # --- 不一致 & オラクル利得（録音単位 argmax）---
    ry_a, rp_a = aggregate_by_recording(ast10, y, xc)
    ry_t, rp_t = aggregate_by_recording(tea10, y, xc)
    assert np.array_equal(ry_a, ry_t)
    pa, pt, yt = rp_a.argmax(1), rp_t.argmax(1), ry_a
    disagree = (pa != pt)
    a_right = (pa == yt); t_right = (pt == yt)
    oracle = (a_right | t_right)                            # どちらかが正解
    comp_gain = float(oracle.mean() - a_right.mean())       # AST単体に対する「上限」余地
    print(f"\n[録音単位] AST f1={f_ast:.4f}  teacher f1={f_tea:.4f}  "
          f"ensemble best f1={best[0]:.4f}(α={best[1]})", flush=True)
    print(f"[不一致] 録音の {disagree.mean()*100:.1f}% で AST と教師の予測が割れる", flush=True)
    print(f"[オラクル] AST正解率={a_right.mean():.4f}  どちらか正解={oracle.mean():.4f}  "
          f"→ 相補的上限の余地=+{comp_gain:.4f}", flush=True)
    print(f"  内訳: AST正/教師誤={int((a_right&~t_right).sum())}  "
          f"AST誤/教師正={int((~a_right&t_right).sum())}  両誤={int((~a_right&~t_right).sum())}", flush=True)

    # --- ゲート判定 ---
    ens_gain = best[0] - f_ast
    verdict = "GO" if (ens_gain > 0.005 or comp_gain > 0.02) else "NO-GO"
    print(f"\n===== GATE: {verdict} =====", flush=True)
    print(f"  ensemble利得={ens_gain:+.4f} / 相補余地={comp_gain:+.4f} "
          f"(どちらか有意なら GO)", flush=True)
    if verdict == "NO-GO":
        print("  相補性が乏しい → 蒸留しても上限が低い。Phase1以降は再検討。", flush=True)
    else:
        print("  相補性あり → 蒸留(Phase1〜)へ進む価値あり。", flush=True)

    out = PROJECT_ROOT / "outputs" / "probe_sweep" / f"complementarity_{args.ast_tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["ast_rec_f1", f_ast]); w.writerow(["teacher_rec_f1", f_tea])
        w.writerow(["ensemble_best_f1", best[0]]); w.writerow(["ensemble_best_alpha", best[1]])
        w.writerow(["disagree_rate", disagree.mean()])
        w.writerow(["ast_acc", a_right.mean()]); w.writerow(["oracle_acc", oracle.mean()])
        w.writerow(["comp_gain", comp_gain]); w.writerow(["verdict", verdict])
        for nm, al, fv in rows:
            w.writerow([f"{nm}_a{al}", fv])
    print(f"[OK] -> {out}", flush=True)


if __name__ == "__main__":
    main()
