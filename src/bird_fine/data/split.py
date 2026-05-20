"""chunks_index.csv を録音ID単位で train/val/test に分割。

同一録音(XC ID)のチャンクは同じsplitに入れる(leakage防止)。
種ごとの比率を保つ層化分割。

使い方:
    uv run python -m bird_fine.data.split
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    config = load_config()
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
