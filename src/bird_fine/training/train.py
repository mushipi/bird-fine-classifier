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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import yaml
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score
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


def make_compute_metrics(num_target: int = 8):
    """9クラス全体 + 既存8種のみ F1 を返す compute_metrics を生成する。

    metric_for_best_model に 'f1_macro_8class' を指定することで、
    "other" クラスの F1 に引きずられずに既存8種の精度でモデルを選択できる。
    """
    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        target_mask = labels < num_target

        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_macro_8class": float(
                f1_score(
                    labels[target_mask], preds[target_mask],
                    average="macro", zero_division=0,
                    labels=list(range(num_target)),
                )
            ) if target_mask.sum() > 0 else 0.0,
        }
        # "other" クラスが存在する場合のみ recall を追加
        other_mask = labels == num_target
        if other_mask.sum() > 0:
            metrics["other_recall"] = float(recall_score(
                other_mask.astype(int), (preds == num_target).astype(int),
                average="binary", zero_division=0,
            ))
        return metrics
    return compute_metrics


def expand_classifier(model, new_num_labels: int) -> None:
    """既存の分類ヘッドを new_num_labels に拡張する。

    既存クラスの重みを保持したまま "other" ニューロンをゼロ初期化で追加する。
    これにより run10 の学習済み8クラス境界を崩さずに run11 を開始できる。
    """
    old_linear = model.classifier.dense
    old_w = old_linear.weight.data   # (old_n, hidden)
    old_b = old_linear.bias.data     # (old_n,)
    old_n, hidden = old_w.shape

    if old_n == new_num_labels:
        return

    new_linear = nn.Linear(hidden, new_num_labels)
    new_linear.weight.data[:old_n] = old_w
    new_linear.weight.data[old_n:] = torch.zeros(new_num_labels - old_n, hidden)
    new_linear.bias.data[:old_n] = old_b
    new_linear.bias.data[old_n:] = torch.zeros(new_num_labels - old_n)

    model.classifier.dense = new_linear
    model.config.num_labels = new_num_labels
    print(f"[EXPAND] 分類ヘッド拡張: {old_n} → {new_num_labels} クラス（新規ニューロンはゼロ初期化）")


class DuckTrainer(Trainer):
    """WeightedRandomSampler + カスタム CE Loss を組み込んだ Trainer。

    Sampler でバッチ内クラス比率を均等化し、Loss には固定重み alpha を "other" のみに付与する。
    Sampler と Loss の二重補正（Double Dipping）を避けるため、Loss 重みはデータ数ベースではなく
    固定値 alpha のみを使用する。
    """

    def __init__(self, *args, other_label_id: int | None = None, other_alpha: float = 1.0,
                 focal_gamma: float = 0.0, kd_lambda: float = 0.0, kd_temp: float = 2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.other_label_id = other_label_id
        self.other_alpha = other_alpha
        self.focal_gamma = focal_gamma  # 0.0 で従来 CE と等価（後方互換）
        self.kd_lambda = kd_lambda      # 0.0 で蒸留なし（後方互換）
        self.kd_temp = kd_temp

    def _get_train_sampler(self, dataset=None) -> WeightedRandomSampler | None:
        # transformers 5.x は dataset 引数を渡してくる
        ds = dataset if dataset is not None else self.train_dataset
        if ds is None:
            return None
        species_counts = Counter(ds.df["species"])
        sample_weights = [
            1.0 / species_counts[ds.df.iloc[i]["species"]]
            for i in range(len(ds))
        ]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ds),
            replacement=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher = inputs.pop("teacher_proba", None)   # (B, K) or None
        has_t = inputs.pop("has_teacher", None)        # (B,) bool or None
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        num_labels = logits.shape[-1]
        weights = torch.ones(num_labels, device=logits.device)
        if self.other_label_id is not None and self.other_label_id < num_labels:
            weights[self.other_label_id] = self.other_alpha

        if self.focal_gamma > 0.0:
            # focal loss: 易しいサンプル(高 p_t)の損失を抑え、難しいサンプルを強調。
            # class weight は per-sample CE に乗ったまま focal 係数を掛ける。
            ce = nn.CrossEntropyLoss(weight=weights, reduction="none")(logits, labels)
            pt = torch.exp(-ce)  # 正解クラスの確率（weight 込みだが単調性は保たれる）
            loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()
        else:
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        # 知識蒸留: 教師(K=カモ10種)の soft label を温度付き KL で生徒に注入。
        # KD は logits の先頭 K 列（カモ）のみ・教師ありサンプルのみ。other 列は対象外。
        if self.kd_lambda > 0.0 and teacher is not None:
            T = self.kd_temp
            K = teacher.shape[-1]
            log_p = F.log_softmax(logits[:, :K] / T, dim=-1)
            t = teacher.float().clamp_min(1e-8)
            t = t / t.sum(-1, keepdim=True)
            tT = t.pow(1.0 / T)
            tT = tT / tT.sum(-1, keepdim=True)
            kd_per = (tT * (tT.clamp_min(1e-8).log() - log_p)).sum(-1)  # KL(tT‖p) per sample
            if has_t is not None:
                m = has_t.bool()
                kd = kd_per[m].mean() if m.any() else logits.sum() * 0.0
            else:
                kd = kd_per.mean()
            loss = loss + self.kd_lambda * (T * T) * kd
        return (loss, outputs) if return_outputs else loss


def resize_position_embeddings(model, max_length: int) -> None:
    """AST の位置埋め込みを max_length に対応したパッチ数へ線形補間でリサイズする。

    事前学習済み重みは max_length=1024 向けの 1214 埋め込みを持つが、
    3s チャンク（max_length=304）では 350 パッチしか生成されないため不整合が起きる。
    CLS/distillation トークン (先頭2) は流用し、パッチ埋め込み部分のみ補間する。
    """
    import torch.nn as nn

    cfg = model.config
    patch_size: int = cfg.patch_size
    freq_patches = (cfg.num_mel_bins - patch_size) // cfg.frequency_stride + 1
    time_patches = (max_length - patch_size) // cfg.time_stride + 1
    new_num_patches = freq_patches * time_patches
    new_n = new_num_patches + 2  # CLS + distillation token

    emb = model.audio_spectrogram_transformer.embeddings
    old_pos = emb.position_embeddings.data  # (1, old_n, hidden) — nn.Parameter
    _, old_n, hidden = old_pos.shape

    if old_n == new_n:
        return

    cls_tokens = old_pos[:, :2, :]    # (1, 2, hidden)
    patch_tokens = old_pos[:, 2:, :]  # (1, old_n-2, hidden)

    new_patch_tokens = F.interpolate(
        patch_tokens.permute(0, 2, 1).float(),  # (1, hidden, old_n-2)
        size=new_n - 2,
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1)  # (1, new_n-2, hidden)

    new_pos = torch.cat([cls_tokens, new_patch_tokens], dim=1)  # (1, new_n, hidden)
    emb.position_embeddings = nn.Parameter(new_pos)
    print(f"[POS] 位置埋め込みをリサイズ: {old_n} → {new_n} ({time_patches}×{freq_patches} patches)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="1 epoch・サブセットで動作確認",
    )
    parser.add_argument("--output-dir", default=None, help="config の output_dir を上書き")
    parser.add_argument("--duck-order", default=None,
                        help="teacher_proba/duck_order.csv のパス。指定で 10カモ運用(label順=teacher)")
    parser.add_argument("--teacher-dir", default=None, help="KD 教師 proba ディレクトリ")
    parser.add_argument("--distill", action="store_true", help="KD を有効化")
    parser.add_argument("--kd-lambda", type=float, default=1.0)
    parser.add_argument("--kd-temp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader並列数。Linuxなら8等で大幅高速化")
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

    duck_order = None
    if args.duck_order:
        duck_order = pd.read_csv(args.duck_order)["species"].tolist()
        print(f"[REGIME] 10カモ運用: duck_order={duck_order}")
    teacher_dir = Path(args.teacher_dir) if (args.distill and args.teacher_dir) else None
    if args.distill:
        print(f"[KD] 蒸留 ON: λ={args.kd_lambda} T={args.kd_temp} teacher_dir={teacher_dir}")

    print("[LOAD] Dataset構築...")
    train_ds, val_ds, test_ds, label_map = build_datasets(
        splits_dir=splits_dir,
        pretrained=model_cfg["pretrained"],
        project_root=PROJECT_ROOT,
        spec_augment_cfg=spec_augment_cfg,
        max_length=int(model_cfg.get("feature_extractor_max_length", 1024)),
        duck_order=duck_order,
        teacher_dir=teacher_dir,
    )
    id2label = {v: k for k, v in label_map.items()}
    label2id = label_map

    if args.dry_run:
        print("[DRY] dry-run: 各speciesから層化サンプリング（train 各6 / val 各3）")
        train_ds.df = train_ds.df.groupby("species").head(6).reset_index(drop=True)
        val_ds.df = val_ds.df.groupby("species").head(3).reset_index(drop=True)

    print(f"  train: {len(train_ds)} / val: {len(val_ds)} / test: {len(test_ds)}")

    num_labels = len(duck_order) if duck_order else int(model_cfg["num_labels"])
    init_from = model_cfg.get("init_from")
    max_length = int(model_cfg.get("feature_extractor_max_length", 1024))

    if init_from:
        import json
        init_path = PROJECT_ROOT / init_from
        saved_cfg = json.loads((init_path / "config.json").read_text())
        saved_n = int(saved_cfg.get("num_labels", 8))
        print(f"[LOAD] チェックポイントから初期化: {init_path} (saved num_labels={saved_n})")
        # saved_n で読み込む → ヘッドが完全に一致してランダム再初期化されない
        saved_id2label = {i: f"cls{i}" for i in range(saved_n)}
        saved_label2id = {v: k for k, v in saved_id2label.items()}
        model = build_ast_classifier(
            pretrained=str(init_path),
            num_labels=saved_n,
            id2label=saved_id2label,
            label2id=saved_label2id,
        )
        if saved_n < num_labels:
            expand_classifier(model, num_labels)
        # 正しい id2label/label2id をセット
        model.config.id2label = id2label
        model.config.label2id = label2id
    else:
        print(f"[LOAD] モデル: {model_cfg['pretrained']}")
        model = build_ast_classifier(
            pretrained=model_cfg["pretrained"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

    if max_length != 1024:
        resize_position_embeddings(model, max_length)
        model.config.max_length = max_length

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    output_dir = str(PROJECT_ROOT / (args.output_dir or train_cfg["output_dir"]))
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
        dataloader_num_workers=args.num_workers,  # Linux なら >0 で高速化
        seed=args.seed,
    )

    callbacks = []
    patience = int(train_cfg.get("early_stopping_patience", 0))
    if patience > 0 and not args.dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # "other" クラスの設定
    other_label_id = label_map.get("other")
    other_alpha = float(config.get("other_class", {}).get("loss_alpha", 1.0))
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    num_target = num_labels - (1 if other_label_id is not None else 0)
    if other_label_id is not None:
        print(f"[OTHER] label_id={other_label_id} / loss_alpha={other_alpha} / metric=f1_macro_8class")
    if focal_gamma > 0.0:
        print(f"[FOCAL] focal loss 有効: gamma={focal_gamma}")

    trainer = DuckTrainer(
        model=model,
        args=training_args,
        other_label_id=other_label_id,
        other_alpha=other_alpha,
        focal_gamma=focal_gamma,
        kd_lambda=args.kd_lambda if args.distill else 0.0,
        kd_temp=args.kd_temp,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=make_compute_metrics(num_target=num_target),
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
