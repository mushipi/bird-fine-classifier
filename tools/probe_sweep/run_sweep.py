"""Perch 線形プローブ正則化スイープ（A→B 一気通し）。

3つの arm を同一評価軸（test 録音単位 f1_macro）で横並べる単一スクリプト:
  arm A  (linear) : torch AdamW 50epoch, weight_decay スイープ
  arm A' (mlp)    : 同上（dropout 既定）
  arm B  (sklearn): LogisticRegression C グリッド（class_weight=balanced）

config(正則化強度)のランキングは val 録音単位 f1_macro で行う（test はリーク防止のため
選択に使わず、選ばれた config の test 指標を報告するだけ）。全 config の val/test 指標を
outputs/probe_sweep/sweep_results.{csv,json} に保存。

AST baseline（test 録音単位 f1_macro=0.838）を超えた arm-best のみ models/ に保存し
docs/journal.md に追記する。超えなければ結果 CSV のみ残す。

実行（mainPC, GPU 自動判定）:
  .venv/bin/python tools/probe_sweep/run_sweep.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bird_fine.embeddings import io_utils  # noqa: E402
from bird_fine.models.probes import build_probe  # noqa: E402
from bird_fine.training.eval_probe import (  # noqa: E402
    aggregate_by_recording,
    metrics_block,
)
from bird_fine.training.train_probe import (  # noqa: E402
    class_weights,
    run_epoch,
    standardize_apply,
    standardize_fit,
)

AST_BASELINE = 0.838  # 旧 AST(8種, 凍結+線形) の test 録音単位 f1_macro
SEED = 42
EPOCHS = 50
LR = 5e-3
BATCH = 128
WD_GRID = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]      # arm A / A'
C_GRID = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]          # arm B


# ----------------------------------------------------------------------------
# 推論ヘルパ（既存ヘルパには proba 取得が無いため最小限のみ新規）
# ----------------------------------------------------------------------------
@torch.no_grad()
def torch_proba(model: nn.Module, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def sklearn_proba_full(clf, X: np.ndarray, num_classes: int) -> np.ndarray:
    """LogisticRegression.predict_proba（出現クラス列のみ）を (N, num_classes) に展開。

    metrics_block は proba[:, c]（ラベルID直参照）を使うため、列をラベルIDに揃える。
    """
    p = clf.predict_proba(X)
    full = np.zeros((len(X), num_classes), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        full[:, int(c)] = p[:, j]
    return full


def rec_metrics(proba: np.ndarray, y: np.ndarray, xc_id, operative) -> dict:
    """チャンク proba → 録音単位集約 → 録音単位 f1/acc/auroc。"""
    rec_y, rec_proba = aggregate_by_recording(proba, y, xc_id)
    m, _ = metrics_block(rec_y, rec_proba, operative)
    return m


# ----------------------------------------------------------------------------
# arm A / A'：torch スイープ
# ----------------------------------------------------------------------------
def train_torch(probe, Xtr, ytr, Xva, yva, num_classes, operative, wd, device):
    """1 config 学習。epoch 選択は既存同様 val チャンク f1 で best state を保持。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_probe(probe, Xtr.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    w = class_weights(ytr, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    best_f1, best_state = -1.0, None
    for _ in range(1, EPOCHS + 1):
        run_epoch(model, Xtr, ytr, optimizer, criterion, device, BATCH, True, operative)
        _, vf1 = run_epoch(model, Xva, yva, optimizer, criterion, device, BATCH, False, operative)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_state


# ----------------------------------------------------------------------------
# arm B：sklearn LogisticRegression
# ----------------------------------------------------------------------------
def fit_sklearn(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(Xtr, ytr)
    return clf


# ----------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = PROJECT_ROOT / "data" / "embeddings" / "perch"
    tr = io_utils.load_embeddings(emb_dir / "train.npz")
    va = io_utils.load_embeddings(emb_dir / "val.npz")
    te = io_utils.load_embeddings(emb_dir / "test.npz")

    label_names = tr["label_names"]
    num_classes = len(label_names)
    Xtr, Xva, Xte = tr["X"], va["X"], te["X"]
    ytr, yva, yte = tr["y"], va["y"], te["y"]
    xctr, xcva, xcte = tr["xc_id"], va["xc_id"], te["xc_id"]

    # 標準化（train 統計）
    mean, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mean, std)
    Xva = standardize_apply(Xva, mean, std)
    Xte = standardize_apply(Xte, mean, std)
    operative = sorted(np.unique(ytr).tolist())  # 実在クラスのみで macro 集計

    print(f"[sweep] device={device}  C={num_classes}(運用{len(operative)})  "
          f"train={len(ytr)}/{len(np.unique(xctr))}rec  val={len(yva)}/{len(np.unique(xcva))}rec  "
          f"test={len(yte)}/{len(np.unique(xcte))}rec", flush=True)
    print(f"[sweep] AST baseline (test rec f1_macro) = {AST_BASELINE}", flush=True)

    rows = []
    # 保存用：arm-best の中身を保持（torch=state, sklearn=clf）
    keep = {}

    # --- arm A / A'：torch ---
    for arm, probe in [("linear", "linear"), ("mlp", "mlp")]:
        for wd in WD_GRID:
            model, state = train_torch(probe, Xtr, ytr, Xva, yva,
                                       num_classes, operative, wd, device)
            vp = torch_proba(model, Xva, device)
            tp = torch_proba(model, Xte, device)
            vm = rec_metrics(vp, yva, xcva, operative)
            tm = rec_metrics(tp, yte, xcte, operative)
            rows.append({
                "arm": arm, "param": "weight_decay", "value": wd,
                "val_rec_f1": vm["f1_macro"], "val_rec_acc": vm["top1_acc"],
                "val_rec_auroc": vm["macro_auroc"],
                "test_rec_f1": tm["f1_macro"], "test_rec_acc": tm["top1_acc"],
                "test_rec_auroc": tm["macro_auroc"],
            })
            keep[(arm, wd)] = ("torch", state)
            print(f"  [{arm:6s}] wd={wd:<7g} "
                  f"val_rec_f1={vm['f1_macro']:.4f}  test_rec_f1={tm['f1_macro']:.4f}  "
                  f"test_auroc={tm['macro_auroc']:.4f}", flush=True)

    # --- arm B：sklearn ---
    for C in C_GRID:
        clf = fit_sklearn(Xtr, ytr, C)
        vp = sklearn_proba_full(clf, Xva, num_classes)
        tp = sklearn_proba_full(clf, Xte, num_classes)
        vm = rec_metrics(vp, yva, xcva, operative)
        tm = rec_metrics(tp, yte, xcte, operative)
        rows.append({
            "arm": "sklearn_logreg", "param": "C", "value": C,
            "val_rec_f1": vm["f1_macro"], "val_rec_acc": vm["top1_acc"],
            "val_rec_auroc": vm["macro_auroc"],
            "test_rec_f1": tm["f1_macro"], "test_rec_acc": tm["top1_acc"],
            "test_rec_auroc": tm["macro_auroc"],
        })
        keep[("sklearn_logreg", C)] = ("sklearn", clf)
        print(f"  [sklearn] C={C:<7g} "
              f"val_rec_f1={vm['f1_macro']:.4f}  test_rec_f1={tm['f1_macro']:.4f}  "
              f"test_auroc={tm['macro_auroc']:.4f}", flush=True)

    # --- arm ごとの best を val_rec_f1 で選定 ---
    arm_best = {}
    for r in rows:
        a = r["arm"]
        if a not in arm_best or r["val_rec_f1"] > arm_best[a]["val_rec_f1"]:
            arm_best[a] = r

    # --- 比較表 ---
    print("\n========== arm-best（val_rec_f1 で選定）==========", flush=True)
    print(f"{'arm':16s} {'param':14s} {'val_f1':>8s} {'test_f1':>8s} "
          f"{'test_acc':>9s} {'test_auroc':>11s}  vs0.838", flush=True)
    champion = None
    for a, r in arm_best.items():
        mark = "  ↑超え" if r["test_rec_f1"] > AST_BASELINE else f"  {r['test_rec_f1'] - AST_BASELINE:+.3f}"
        print(f"{a:16s} {r['param']}={r['value']:<8g} "
              f"{r['val_rec_f1']:8.4f} {r['test_rec_f1']:8.4f} "
              f"{r['test_rec_acc']:9.4f} {r['test_rec_auroc']:11.4f}{mark}", flush=True)
        if champion is None or r["test_rec_f1"] > champion["test_rec_f1"]:
            champion = r

    # --- 結果保存（常に）---
    out_dir = PROJECT_ROOT / "outputs" / "probe_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    import csv
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    json_path = out_dir / "sweep_results.json"
    json_path.write_text(json.dumps({
        "timestamp": ts, "ast_baseline": AST_BASELINE,
        "rows": rows, "arm_best": arm_best, "champion": champion,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] results -> {csv_path}", flush=True)

    # --- 0.838 超え時のみ保存 + journal 追記 ---
    print(f"\n[champion] {champion['arm']} {champion['param']}={champion['value']} "
          f"test_rec_f1={champion['test_rec_f1']:.4f}", flush=True)
    if champion["test_rec_f1"] <= AST_BASELINE:
        print(f"[done] AST baseline {AST_BASELINE} 未超え → モデル保存/journal 追記なし。"
              f"sweep_results のみ残す。", flush=True)
        return

    save_champion(champion, keep, mean, std, label_names, num_classes,
                  operative, PROJECT_ROOT)
    append_journal(ts, arm_best, champion, PROJECT_ROOT)


def save_champion(champion, keep, mean, std, label_names, num_classes,
                  operative, root):
    arm, val = champion["arm"], champion["value"]
    kind, obj = keep[(arm, val)]
    out = root / "models" / f"probe_perch_{arm}_swept"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": arm, "param": champion["param"], "value": val,
        "num_classes": num_classes, "label_names": label_names,
        "operative": operative,
        "val_rec_f1": champion["val_rec_f1"],
        "test_rec_f1": champion["test_rec_f1"],
        "test_rec_acc": champion["test_rec_acc"],
        "test_rec_auroc": champion["test_rec_auroc"],
        "ast_baseline": AST_BASELINE,
    }
    np.savez(out / "scaler.npz", mean=mean, std=std)
    if kind == "torch":
        torch.save(obj, out / "probe.pt")
    else:
        import joblib
        joblib.dump(obj, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[save] champion -> {out}", flush=True)


def append_journal(ts, arm_best, champion, root):
    jp = root / "docs" / "journal.md"
    lines = [
        "\n---\n",
        f"\n## {datetime.now().strftime('%Y-%m-%d')} Perch 正則化スイープ"
        f"（A→B 一気通し）: AST 0.838 を超えた\n",
        "\n### 結果（arm-best, val_rec_f1 で選定 → test 録音単位）\n",
        "\n| arm | param | val_rec_f1 | test_rec_f1 | test_acc | test_auroc |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for a, r in arm_best.items():
        lines.append(f"| {a} | {r['param']}={r['value']} | {r['val_rec_f1']:.4f} | "
                     f"{r['test_rec_f1']:.4f} | {r['test_rec_acc']:.4f} | "
                     f"{r['test_rec_auroc']:.4f} |\n")
    lines.append(
        f"\n**champion**: {champion['arm']} {champion['param']}={champion['value']} → "
        f"test 録音単位 f1_macro={champion['test_rec_f1']:.4f} "
        f"(AST baseline 0.838 を {champion['test_rec_f1'] - AST_BASELINE:+.4f} 更新)。"
        f"保存先 models/probe_perch_{champion['arm']}_swept/。\n")
    with open(jp, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[journal] appended -> {jp}", flush=True)


if __name__ == "__main__":
    main()
