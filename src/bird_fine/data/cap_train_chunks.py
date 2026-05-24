"""train.csv の 1録音あたりチャンク数に上限を設けるサブサンプリング。

val/test は変更しない。train.csv のみ xc_id 単位で chunks を cap にクリップする。
1録音から大量のチャンクが生成されてチャンク不均衡を起こす種（例: Tufted_Duck の
XC488112/XC488113 が約270 chunks ずつ）の影響を緩和する目的。

使い方:
    uv run python -m bird_fine.data.cap_train_chunks --cap 100
    uv run python -m bird_fine.data.cap_train_chunks --cap 50 --seed 42 --dry-run
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cap_train(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """xc_id 単位で chunks 数が cap を超える行を seed 固定でサブサンプリング。"""
    rng = random.Random(seed)
    keep_indices: list[int] = []
    for (_species, _xc_id), sub in df.groupby(["species", "xc_id"], sort=False):
        indices = sub.index.tolist()
        if len(indices) <= cap:
            keep_indices.extend(indices)
        else:
            # chunk_index 順を保ちつつ、ランダムに cap 件選ぶ
            sampled = rng.sample(indices, cap)
            keep_indices.extend(sorted(sampled))
    return df.loc[keep_indices].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, required=True, help="1録音あたりチャンク数の上限")
    parser.add_argument("--seed", type=int, default=42, help="サブサンプリングの乱数 seed")
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き換えせず差分のみ表示")
    args = parser.parse_args()

    config = load_config()
    splits_dir = PROJECT_ROOT / config["preprocessing"]["splits_dir"]
    train_csv = splits_dir / "train.csv"
    if not train_csv.exists():
        print(f"[ERROR] {train_csv} が見つからない")
        sys.exit(1)

    df = pd.read_csv(train_csv)
    print(f"[INFO] train.csv: {len(df)} chunks / {df['xc_id'].nunique()} unique xc_id")

    df_capped = cap_train(df, args.cap, args.seed)
    print(f"[INFO] cap={args.cap} 適用後: {len(df_capped)} chunks ({len(df_capped) - len(df):+d})")
    print()

    # 種ごとの diff
    before = df.groupby("species").size()
    after = df_capped.groupby("species").size()
    print(f"{'species':22s} {'before':>7s} {'after':>7s} {'diff':>7s}")
    for sp in sorted(before.index):
        b, a = int(before[sp]), int(after.get(sp, 0))
        diff = a - b
        marker = " *" if diff != 0 else ""
        print(f"{sp:22s} {b:7d} {a:7d} {diff:+7d}{marker}")

    if args.dry_run:
        print("\n[DRY-RUN] ファイル書き換えなし")
        return

    df_capped.to_csv(train_csv, index=False)
    print(f"\n[OK] {train_csv} を上書き")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
