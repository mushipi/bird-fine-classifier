"""素の小CNN への Perch→KD 蒸留（手法検証用・自己完結）。

弱い from-scratch CNN に同じ Perch 教師を蒸留し、KD の効果が大きく有意化するかを見る
（AST は天井近くで KD 効果が CI に埋もれた、その補完検証）。dataset / 教師proba / KD損失 /
評価フォーマットは全て既存を再利用。変えるのは生徒アーキ（AST → 素CNN）のみ。

実行（mainPC, 本体.venv）:
  base: .venv/bin/python tools/train_cnn_kd.py --tag CNNbase --duck-order data/embeddings/teacher_proba/duck_order.csv
  KD  : .venv/bin/python tools/train_cnn_kd.py --tag CNNkd --distill --kd-lambda 1.0 --kd-temp 2.0 \
          --teacher-dir data/embeddings/teacher_proba --duck-order data/embeddings/teacher_proba/duck_order.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bird_fine.data.dataset import build_datasets, collate_fn  # noqa: E402


class DuckCNN(nn.Module):
    """素の小CNN（~0.4M params）。入力 (B,304,128) mel → (B,1,304,128) → conv×4 → GAP → FC。"""

    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Linear(256, n_classes)

    def forward(self, input_values):          # (B, T=304, F=128)
        x = input_values.unsqueeze(1)         # (B,1,T,F)
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)  # (B,256)
        return self.head(h)


def kd_loss(logits, teacher, has_t, T):
    """train.py DuckTrainer.compute_loss と同一の温度付き KL（教師ありサンプルのみ）。"""
    K = teacher.shape[-1]
    log_p = F.log_softmax(logits[:, :K].float() / T, dim=-1)
    t = teacher.float().clamp_min(1e-8)
    t = t / t.sum(-1, keepdim=True)
    tT = t.pow(1.0 / T)
    tT = tT / tT.sum(-1, keepdim=True)
    kd_per = (tT * (tT.clamp_min(1e-8).log() - log_p)).sum(-1)
    m = has_t.bool()
    return kd_per[m].mean() if m.any() else (logits.sum() * 0.0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logits = model(batch["input_values"].to(device))
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--duck-order", required=True)
    ap.add_argument("--distill", action="store_true")
    ap.add_argument("--kd-lambda", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--teacher-dir", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    model_cfg = cfg["model"]

    duck_order = pd.read_csv(args.duck_order)["species"].tolist()
    n_classes = len(duck_order)
    teacher_dir = Path(args.teacher_dir) if (args.distill and args.teacher_dir) else None
    print(f"[cnn] tag={args.tag} distill={args.distill} λ={args.kd_lambda} T={args.kd_temp} "
          f"classes={n_classes} device={device}", flush=True)

    train_ds, val_ds, test_ds, _ = build_datasets(
        splits_dir=PROJECT_ROOT / "data" / "splits",
        pretrained=model_cfg["pretrained"],
        project_root=PROJECT_ROOT,
        max_length=int(model_cfg.get("feature_extractor_max_length", 1024)),
        duck_order=duck_order,
        teacher_dir=teacher_dir,
    )
    if args.dry_run:
        train_ds.df = train_ds.df.groupby("species").head(8).reset_index(drop=True)
        val_ds.df = val_ds.df.groupby("species").head(4).reset_index(drop=True)
        # dry-run は df を切るので teacher 整合のため再ロード回避＝teacher 無効化
        train_ds.teacher = None; val_ds.teacher = None
        args.epochs = 1

    # 不均衡対策: AST 同様 WeightedRandomSampler（1/species_count）
    counts = Counter(train_ds.df["species"])
    w = [1.0 / counts[train_ds.df.iloc[i]["species"]] for i in range(len(train_ds))]
    g = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(w, len(train_ds), replacement=True, generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    model = DuckCNN(n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cnn] params={n_params/1e6:.2f}M  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()
    operative = list(range(n_classes))

    best_f1, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = ce(logits.float(), y)
            if args.distill and "teacher_proba" in batch:
                loss = loss + args.kd_lambda * (args.kd_temp ** 2) * kd_loss(
                    logits, batch["teacher_proba"].to(device), batch["has_teacher"].to(device), args.kd_temp)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        vp, vy = predict(model, val_loader, device)
        vf1 = f1_score(vy, vp.argmax(1), labels=operative, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  ep{ep:3d}  val_f1(chunk)={vf1:.4f}  (best={best_f1:.4f}@{best_ep})", flush=True)
        if bad >= args.patience and not args.dry_run:
            print(f"  [early stop] {args.patience} epoch 改善なし", flush=True); break

    model.load_state_dict(best_state)

    out_model = PROJECT_ROOT / "models" / f"cnn-duck-{args.tag}"
    out_model.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_model / "model.pt")

    # val/test proba ダンプ（ast_eval_proba と同フォーマット → kd_compare_ci / グリッド選定が食える）
    out_dir = PROJECT_ROOT / "outputs" / "ast_proba"; out_dir.mkdir(parents=True, exist_ok=True)
    for split, ds, loader in [("val", val_ds, val_loader), ("test", test_ds, test_loader)]:
        pp, _ = predict(model, loader, device)
        d = ds.df
        np.savez(
            out_dir / f"{args.tag}_{split}.npz",
            proba=pp.astype(np.float32),
            xc_id=d["xc_id"].to_numpy(dtype=object),
            chunk_index=d["chunk_index"].to_numpy(dtype=np.int64),
            species=d["species"].to_numpy(dtype=object),
            id2label=np.asarray(duck_order, dtype=object),
        )
    print(f"[OK] best val f1(chunk)={best_f1:.4f}@{best_ep}  -> {out_dir}/{args.tag}_{{val,test}}.npz", flush=True)


if __name__ == "__main__":
    main()
