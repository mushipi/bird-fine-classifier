"""学習済みプローブを test 埋め込みで評価。

models/{run}/ の probe.pt + scaler.npz + meta.json と data/embeddings/{model}/test.npz を読み、
chunk単位と録音(xc_id)単位の両方で f1_macro / top-1 acc / macro-AUROC / 混同行列を出力。
結果は outputs/{run}_{timestamp}/ に保存。AST baseline(test f1_macro 0.838)との比較用。

torch 環境で実行:
  .venv/bin/python -m bird_fine.training.eval_probe --run probe_perch_linear
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from bird_fine.embeddings import io_utils
from bird_fine.models.probes import build_probe
from bird_fine.training.train_probe import get_features, standardize_apply

PROJECT_ROOT = io_utils.PROJECT_ROOT


def plot_confusion_matrix(cm, labels, save_path: Path, normalize: bool = True) -> None:
    """evaluate.plot_confusion_matrix と同等（transformers を引かないため再実装）。"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    m = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(m, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def macro_auroc(y_true: np.ndarray, proba: np.ndarray, num_classes: int) -> float:
    """macro-AUROC (OvR)。test に存在しないクラスがあると未定義になるため安全に処理。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=list(range(num_classes))
        ))
    except ValueError:
        return float("nan")  # 一部クラスが test に欠損 等で未定義のとき


def aggregate_by_recording(proba, y, xc_id):
    """xc_id 単位に softmax 平均 → 1録音1サンプルへ集約。"""
    df = pd.DataFrame({"xc": xc_id, "y": y})
    rec_y, rec_proba = [], []
    for xc, g in df.groupby("xc"):
        idx = g.index.to_numpy()
        rec_proba.append(proba[idx].mean(axis=0))
        rec_y.append(int(g["y"].iloc[0]))
    return np.array(rec_y), np.vstack(rec_proba)


def metrics_block(y, proba, num_classes, label_names):
    preds = proba.argmax(axis=1)
    return {
        "n": int(len(y)),
        "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        "top1_acc": float((y == preds).mean()),
        "macro_auroc": macro_auroc(y, proba, num_classes),
    }, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="models/ 配下の run 名（例: probe_perch_linear）")
    ap.add_argument("--emb-root", default=str(PROJECT_ROOT / "data" / "embeddings"))
    ap.add_argument("--models-root", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs"))
    args = ap.parse_args()

    run_dir = Path(args.models_root) / args.run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    scaler = np.load(run_dir / "scaler.npz")
    mean, std = scaler["mean"], scaler["std"]
    label_names = meta["label_names"]; num_classes = meta["num_classes"]

    test = io_utils.load_embeddings(Path(args.emb_root) / meta["model"] / "test.npz")
    X = standardize_apply(get_features(test, meta["probe"]), mean, std)
    y, xc = test["y"], test["xc_id"]

    model = build_probe(meta["probe"], meta["in_dim"], num_classes,
                        dropout=(meta["hparams"].get("dropout") or 0.0))
    model.load_state_dict(torch.load(run_dir / "probe.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.from_numpy(X)), dim=-1).numpy()

    chunk_m, preds = metrics_block(y, proba, num_classes, label_names)
    rec_y, rec_proba = aggregate_by_recording(proba, y, xc)
    rec_m, _ = metrics_block(rec_y, rec_proba, num_classes, label_names)

    cm = confusion_matrix(y, preds, labels=list(range(num_classes)))
    report = classification_report(y, preds, labels=list(range(num_classes)),
                                   target_names=label_names, output_dict=True, zero_division=0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{args.run}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, label_names, out_dir / "confusion_matrix_norm.png", True)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame({"y_true": y, "y_pred": preds, "xc_id": xc}).to_csv(out_dir / "predictions.csv", index=False)
    result = {"run": args.run, "model": meta["model"], "probe": meta["probe"],
              "chunk_level": chunk_m, "recording_level": rec_m,
              "best_val_f1": meta.get("best_val_f1"), "classification_report": report}
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {args.run} (test) ===")
    print(f"  chunk : f1_macro={chunk_m['f1_macro']:.4f}  top1={chunk_m['top1_acc']:.4f}  "
          f"AUROC={chunk_m['macro_auroc']:.4f}  (n={chunk_m['n']})")
    print(f"  record: f1_macro={rec_m['f1_macro']:.4f}  top1={rec_m['top1_acc']:.4f}  "
          f"AUROC={rec_m['macro_auroc']:.4f}  (n={rec_m['n']})")
    print(f"  vs AST baseline (test f1_macro 0.838): "
          f"{'UP' if chunk_m['f1_macro'] > 0.838 else 'DOWN'} ({chunk_m['f1_macro']-0.838:+.4f})")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
