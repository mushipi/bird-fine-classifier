"""chunks_index.csv を録音ID単位で train/val/test に分割。

同一録音(XC ID)のチャンクは同じsplitに入れる(leakage防止)。
種ごとの比率を保つ層化分割。

使い方:
    uv run python -m bird_fine.data.split
    # 既存 splits を維持し、新規録音だけ train に追加（追加収集 run 用）
    uv run python -m bird_fine.data.split --preserve-existing
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path=None) -> dict:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_by_recording(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """種ごとに xc_id 単位で層化分割。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    train_ids: set[tuple[str, str]] = set()
    val_ids: set[tuple[str, str]] = set()
    test_ids: set[tuple[str, str]] = set()

    for species, sub in df.groupby("species"):
        ids = sorted(sub["xc_id"].unique().tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = 0

        for rid in ids[:n_train]:
            train_ids.add((species, rid))
        for rid in ids[n_train : n_train + n_val]:
            val_ids.add((species, rid))
        for rid in ids[n_train + n_val :]:
            test_ids.add((species, rid))

    def belongs(row, target):
        return (row["species"], row["xc_id"]) in target

    train_df = df[df.apply(lambda r: belongs(r, train_ids), axis=1)].reset_index(drop=True)
    val_df = df[df.apply(lambda r: belongs(r, val_ids), axis=1)].reset_index(drop=True)
    test_df = df[df.apply(lambda r: belongs(r, test_ids), axis=1)].reset_index(drop=True)
    return train_df, val_df, test_df


def split_preserving_existing(
    df: pd.DataFrame,
    splits_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """既存 splits の (species, xc_id) を維持し、新規録音は全て train に追加する。

    用途: 既に学習・評価済みの run と比較するため test/val を固定したまま、
    追加収集した録音のみを train に組み込む追加収集 run。
    """
    existing = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} が見つからない。--preserve-existing は既存 splits を前提とする")
        existing[name] = pd.read_csv(path)

    train_ids = set(zip(existing["train"]["species"], existing["train"]["xc_id"]))
    val_ids = set(zip(existing["val"]["species"], existing["val"]["xc_id"]))
    test_ids = set(zip(existing["test"]["species"], existing["test"]["xc_id"]))
    known_ids = train_ids | val_ids | test_ids

    df_keys = set(zip(df["species"], df["xc_id"]))
    new_ids = df_keys - known_ids
    removed_ids = known_ids - df_keys

    if removed_ids:
        print(f"[WARN] 既存 split にあって現在の chunks_index にない録音が {len(removed_ids)} 件。preprocess の状態を確認して")
        for sp, rid in sorted(removed_ids)[:5]:
            print(f"  - {sp} / {rid}")

    print(f"[INFO] 既存録音: train={len(train_ids)} / val={len(val_ids)} / test={len(test_ids)}")
    print(f"[INFO] 新規録音（train に追加）: {len(new_ids)}")
    if new_ids:
        per_species: dict[str, int] = defaultdict(int)
        for sp, _ in new_ids:
            per_species[sp] += 1
        for sp, n in sorted(per_species.items()):
            print(f"  + {sp}: {n} 録音")

    train_ids_updated = train_ids | new_ids

    def belongs(row, target):
        return (row["species"], row["xc_id"]) in target

    train_df = df[df.apply(lambda r: belongs(r, train_ids_updated), axis=1)].reset_index(drop=True)
    val_df = df[df.apply(lambda r: belongs(r, val_ids), axis=1)].reset_index(drop=True)
    test_df = df[df.apply(lambda r: belongs(r, test_ids), axis=1)].reset_index(drop=True)
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="既存 splits の (species, xc_id) を維持し、新規録音だけ train に追加する",
    )
    parser.add_argument("--config", default=None,
                        help="設定ファイル（省略時 config.yaml）。群別は config-<group>.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pp = config["preprocessing"]
    processed_dir = PROJECT_ROOT / pp["processed_dir"]
    splits_dir = PROJECT_ROOT / pp["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    index_csv = processed_dir / "chunks_index.csv"
    if not index_csv.exists():
        print(f"[ERROR] {index_csv} が見つからない。先に preprocess.py を実行して。")
        return

    df = pd.read_csv(index_csv)
    print(f"[INFO] 全チャンク数: {len(df)}")
    print(f"[INFO] 種数: {df['species'].nunique()}")

    if args.preserve_existing:
        print(f"[MODE] preserve-existing — 既存 val/test を固定、新規録音は train に追加")
        train_df, val_df, test_df = split_preserving_existing(df, splits_dir)
    else:
        train_df, val_df, test_df = split_by_recording(
            df,
            train_ratio=float(pp["train_ratio"]),
            val_ratio=float(pp["val_ratio"]),
            test_ratio=float(pp["test_ratio"]),
            seed=int(pp["random_seed"]),
        )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # ラベルマップ生成
    species_list = sorted(df["species"].unique().tolist())
    label_map = {sp: i for i, sp in enumerate(species_list)}
    pd.DataFrame(
        [{"label_id": v, "species": k} for k, v in label_map.items()]
    ).to_csv(splits_dir / "label_map.csv", index=False)

    print("\n[INFO] 分割結果（チャンク数 / 録音数）:")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_chunks = len(d)
        n_recordings = d["xc_id"].nunique() if len(d) else 0
        print(f"  {name}: {n_chunks} chunks / {n_recordings} recordings")

    print("\n[INFO] 種ごとの分布:")
    pivot = (
        pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                test_df.assign(split="test"),
            ]
        )
        .groupby(["species", "split"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    print(f"\n[NOTE] splits保存先: {splits_dir}")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
