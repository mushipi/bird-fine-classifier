"""凍結埋め込み上の軽量プローブ学習。

data/embeddings/{model}/{train,val}.npz を読み、標準化(train統計)した埋め込みで
プローブ(線形/MLP/attentive)を学習。val f1_macro で best を選び models/{run}/ に保存。

レビュー論文準拠の既定: AdamW, lr=0.005, weight_decay=5e-4, epochs=50, batch=128。
データ不均衡には class weights(逆頻度)で対処。

torch 環境(tools/aves_embed/.venv)で実行:
  .venv/bin/python -m bird_fine.training.train_probe --model perch --probe linear
（PYTHONPATH=src か、tools/aves_embed から sys.path 経由で起動）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from bird_fine.embeddings import io_utils
from bird_fine.models.probes import build_probe

PROJECT_ROOT = io_utils.PROJECT_ROOT


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最終次元 D に対する mean/std を train から推定。X は (N,D) or (N,S,D)。"""
    axes = tuple(range(X.ndim - 1))
    mean = X.mean(axis=axes)
    std = X.std(axis=axes)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def get_features(data: dict, probe: str) -> np.ndarray:
    """probe に応じて入力特徴を選択（attentive=系列, それ以外=平均）。"""
    if probe == "attentive":
        if "X_seq" not in data:
            raise ValueError("attentive プローブには X_seq（系列埋め込み）が必要。BirdAVES を使って。")
        return data["X_seq"]
    return data["X"]


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)  # 逆頻度（平均1付近に正規化）
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, X, y, optimizer, criterion, device, batch_size, train: bool, labels=None):
    model.train(train)
    n = len(y)
    idx = np.random.permutation(n) if train else np.arange(n)
    total = 0.0
    preds = np.empty(n, dtype=np.int64)
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        xb = torch.from_numpy(X[b]).to(device)
        yb = torch.from_numpy(y[b]).to(device)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss) * len(b)
        preds[b] = logits.argmax(-1).detach().cpu().numpy()
    f1 = f1_score(y, preds, labels=labels, average="macro", zero_division=0)
    return total / n, f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["perch", "birdaves"])
    ap.add_argument("--probe", required=True, choices=["linear", "mlp", "attentive"])
    ap.add_argument("--emb-root", default=str(PROJECT_ROOT / "data" / "embeddings"))
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_dir = Path(args.emb_root) / args.model
    tr = io_utils.load_embeddings(emb_dir / "train.npz")
    va = io_utils.load_embeddings(emb_dir / "val.npz")
    label_names = tr["label_names"]; num_classes = len(label_names)

    Xtr = get_features(tr, args.probe); Xva = get_features(va, args.probe)
    mean, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mean, std); Xva = standardize_apply(Xva, mean, std)
    ytr, yva = tr["y"], va["y"]
    in_dim = Xtr.shape[-1]
    # 実在クラスのみで macro 集計（label_map に残る空クラスを除外して f1 を正しくする）
    operative = sorted(np.unique(ytr).tolist())

    kwargs = {}
    if args.dropout is not None:
        kwargs["dropout"] = args.dropout
    model = build_probe(args.probe, in_dim, num_classes, **kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = None if args.no_class_weights else class_weights(ytr, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    run_name = f"probe_{args.model}_{args.probe}"
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] {run_name}  in_dim={in_dim}  C={num_classes}(運用{len(operative)})  "
          f"train={len(ytr)} val={len(yva)}  device={device}", flush=True)
    best_f1, best_state, best_ep = -1.0, None, -1
    history = []
    for ep in range(1, args.epochs + 1):
        tl, tf1 = run_epoch(model, Xtr, ytr, optimizer, criterion, device, args.batch_size, True, operative)
        vl, vf1 = run_epoch(model, Xva, yva, optimizer, criterion, device, args.batch_size, False, operative)
        history.append({"epoch": ep, "train_loss": tl, "train_f1": tf1, "val_loss": vl, "val_f1": vf1})
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  train_f1={tf1:.4f}  val_f1={vf1:.4f}  (best={best_f1:.4f}@{best_ep})", flush=True)

    torch.save(best_state, out_dir / "probe.pt")
    np.savez(out_dir / "scaler.npz", mean=mean, std=std)
    meta = {
        "model": args.model, "probe": args.probe, "in_dim": in_dim,
        "num_classes": num_classes, "label_names": label_names,
        "best_val_f1": best_f1, "best_epoch": best_ep,
        "hparams": {"lr": args.lr, "weight_decay": args.weight_decay,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "dropout": args.dropout, "class_weights": not args.no_class_weights},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] best val f1_macro={best_f1:.4f} (epoch {best_ep}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
