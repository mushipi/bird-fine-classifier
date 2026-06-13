"""学習済みASTモデルをテストセットで評価。

混同行列、種別メトリクス、attention可視化サンプルを outputs/ に保存。

使い方:
    uv run python -m bird_fine.training.evaluate
    uv run python -m bird_fine.training.evaluate --model-dir models/ast-duck
    uv run python -m bird_fine.training.evaluate --no-attention
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

from bird_fine.data.dataset import build_datasets, collate_fn

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path=None) -> dict:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    save_path: Path,
    normalize: bool = True,
) -> None:
    if normalize:
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def visualize_attention(
    model: ASTForAudioClassification,
    dataset,
    id2label: dict[int, str],
    output_dir: Path,
    n_samples: int = 3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """サンプルのattention mapを可視化（最終層の平均attention）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    n = min(n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    for idx in indices:
        sample = dataset[int(idx)]
        input_values = sample["input_values"].unsqueeze(0).to(device)
        true_label = int(sample["labels"].item())

        with torch.no_grad():
            outputs = model(input_values=input_values, output_attentions=True)

        pred = int(torch.argmax(outputs.logits, dim=-1).item())
        # 最終層のattention: (batch, num_heads, seq, seq) → 全headの平均、CLS行
        last_attn = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()  # (seq, seq)
        cls_attn = last_attn[0, 1:]  # CLSから各パッチへの注目度

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(cls_attn)
        axes[0].set_title(
            f"True: {id2label[true_label]}  |  Pred: {id2label[pred]} (idx={idx})"
        )
        axes[0].set_xlabel("Patch index (time × freq)")
        axes[0].set_ylabel("CLS attention weight")

        axes[1].imshow(last_attn, cmap="viridis", aspect="auto")
        axes[1].set_title("Full attention matrix (last layer mean across heads)")
        axes[1].set_xlabel("key patch")
        axes[1].set_ylabel("query patch")

        plt.tight_layout()
        fig.savefig(output_dir / f"attention_sample_{idx:04d}.png", dpi=100)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=None,
        help="学習済みモデルディレクトリ。省略時はconfig.training.output_dir",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="attention可視化をスキップ",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", default=None,
                        help="設定ファイル（省略時 config.yaml）。群別は config-<group>.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pp = config["preprocessing"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    train_cfg = config["training"]

    model_dir = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / train_cfg["output_dir"]
    if not model_dir.exists():
        print(f"[ERROR] {model_dir} が見つからない。先に train.py を実行して。")
        return

    splits_dir = PROJECT_ROOT / pp["splits_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / eval_cfg["output_dir"] / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] device: {device}")

    print("[LOAD] Dataset構築...")
    _, _, test_ds, label_map = build_datasets(
        splits_dir=splits_dir,
        pretrained=model_cfg["pretrained"],
        project_root=PROJECT_ROOT,
        max_length=int(model_cfg.get("feature_extractor_max_length", 1024)),
    )
    id2label = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    print(f"[LOAD] モデル: {model_dir}")
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_preds: list[int] = []
    all_labels: list[int] = []
    print(f"[SEARCH] テスト推論: {len(test_ds)} chunks")
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_values=input_values).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = (y_true == y_pred).mean()

    print("\n[INFO] 評価結果:")
    print(f"  accuracy: {acc:.4f}")
    print(f"  f1_macro: {f1_macro:.4f}")
    print("\n  種ごと:")
    for name in label_names:
        r = report.get(name, {})
        print(
            f"    {name:20s}  P={r.get('precision', 0):.3f}  "
            f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}  "
            f"n={int(r.get('support', 0))}"
        )

    # 結果保存
    plot_confusion_matrix(cm, label_names, out_dir / "confusion_matrix_norm.png", normalize=True)
    plot_confusion_matrix(cm, label_names, out_dir / "confusion_matrix_raw.png", normalize=False)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(out_dir / "confusion_matrix.csv")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": acc, "f1_macro": f1_macro, "classification_report": report},
            f,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)

    if not args.no_attention and eval_cfg.get("attention_visualization", True):
        print("\n[VIZ] attention可視化...")
        visualize_attention(
            model=model,
            dataset=test_ds,
            id2label=id2label,
            output_dir=out_dir / "attention",
            n_samples=3,
            device=device,
        )

    print(f"\n[OK] 結果保存先: {out_dir}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
