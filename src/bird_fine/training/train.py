"""AST fine-tune 学習スクリプト。HuggingFace Trainer ベース。

使い方:
    uv run python -m bird_fine.training.train
    uv run python -m bird_fine.training.train --epochs 5 --batch-size 4
    uv run python -m bird_fine.training.train --dry-run    # 1 epoch サブセットで動作確認
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from bird_fine.data.dataset import build_datasets, collate_fn
from bird_fine.models.ast_classifier import build_ast_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred) -> dict:
    """accuracy / precision / recall / f1_macro を計算。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="1 epoch・サブセットで動作確認",
    )
    args = parser.parse_args()

    config = load_config()
    pp = config["preprocessing"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    splits_dir = PROJECT_ROOT / pp["splits_dir"]
    if not (splits_dir / "train.csv").exists():
        print(f"[ERROR] {splits_dir / 'train.csv'} が見つからない。先に split.py を実行して。")
        return

    spec_augment_cfg = config.get("augmentation", {}).get("spec_augment")
    if spec_augment_cfg and spec_augment_cfg.get("enabled", False):
        print(
            f"[AUG] SpecAugment ON (freq={spec_augment_cfg['freq_mask_param']}×{spec_augment_cfg['num_freq_masks']}, "
            f"time={spec_augment_cfg['time_mask_param']}×{spec_augment_cfg['num_time_masks']})"
        )

    print("[LOAD] Dataset構築...")
    train_ds, val_ds, test_ds, label_map = build_datasets(
        splits_dir=splits_dir,
        pretrained=model_cfg["pretrained"],
        project_root=PROJECT_ROOT,
        spec_augment_cfg=spec_augment_cfg,
    )
    id2label = {v: k for k, v in label_map.items()}
    label2id = label_map

    if args.dry_run:
        print("[DRY] dry-run: 各speciesから層化サンプリング（train 各6 / val 各3）")
        train_ds.df = train_ds.df.groupby("species").head(6).reset_index(drop=True)
        val_ds.df = val_ds.df.groupby("species").head(3).reset_index(drop=True)

    print(f"  train: {len(train_ds)} / val: {len(val_ds)} / test: {len(test_ds)}")

    print(f"[LOAD] モデル: {model_cfg['pretrained']}")
    model = build_ast_classifier(
        pretrained=model_cfg["pretrained"],
        num_labels=int(model_cfg["num_labels"]),
        id2label=id2label,
        label2id=label2id,
    )

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    output_dir = str(PROJECT_ROOT / train_cfg["output_dir"])
    epochs = args.epochs or int(train_cfg["num_train_epochs"])
    train_bs = args.batch_size or int(train_cfg["per_device_train_batch_size"])
    if args.dry_run:
        epochs = 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        fp16=bool(train_cfg["fp16"]) and torch.cuda.is_available(),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=int(train_cfg["logging_steps"]),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Windowsはmultiprocessでハマりがちなので0
    )

    callbacks = []
    patience = int(train_cfg.get("early_stopping_patience", 0))
    if patience > 0 and not args.dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("=" * 60)
    print(f"[START] 学習開始 (epochs={epochs}, train_bs={train_bs}, fp16={training_args.fp16})")
    print("=" * 60)
    trainer.train()

    print("\n[SAVE] best model保存...")
    trainer.save_model(output_dir)

    # ラベルマップも一緒に保存（推論時に使う）
    pd.DataFrame(
        [{"label_id": v, "species": k} for k, v in label_map.items()]
    ).to_csv(Path(output_dir) / "label_map.csv", index=False)

    print(f"\n[INFO] valで最終評価:")
    eval_result = trainer.evaluate(val_ds)
    for k, v in eval_result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    print(f"\n[OK] 完了。チェックポイント: {output_dir}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
