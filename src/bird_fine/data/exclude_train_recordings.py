"""train.csv から指定 XCID（または部分一致するパターン）の録音を完全除外する。

run09 で「Tufted_Duck の長尺2録音 XC488112 / XC488113 が決定境界を歪めている」
仮説を検証するために実装。cap_train_chunks のサブサンプリングと異なり、録音そのもの
を train から消す（chunks も全削除）。val/test は変更しない。

使い方:
    uv run python -m bird_fine.data.exclude_train_recordings --xc-ids XC488112 XC488113
    uv run python -m bird_fine.data.exclude_train_recordings --xc-ids XC488112 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xc-ids",
        nargs="+",
        required=True,
        help="除外する録音の識別子。train.csv の xc_id 列に **部分一致** する文字列。例: XC488112",
    )
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き換えせず差分のみ表示")
    args = parser.parse_args()

    config = load_config()
    splits_dir = PROJECT_ROOT / config["preprocessing"]["splits_dir"]
    train_csv = splits_dir / "train.csv"
    df = pd.read_csv(train_csv)
    print(f"[INFO] train.csv: {len(df)} chunks / {df['xc_id'].nunique()} unique xc_id")

    # 部分一致で除外（xc_id が "XC488112-..." のように prefix のことがあるため）
    excluded_mask = pd.Series(False, index=df.index)
    for xid in args.xc_ids:
        m = df["xc_id"].str.contains(xid, na=False, regex=False)
        n = int(m.sum())
        unique_ids = df.loc[m, "xc_id"].unique()
        print(f"  '{xid}' に一致: {n} chunks / {len(unique_ids)} unique xc_id")
        for u in unique_ids:
            cnt = int((df["xc_id"] == u).sum())
            sp = df[df["xc_id"] == u]["species"].iloc[0] if cnt else "?"
            print(f"    {cnt:4d} chunks  {sp:20s}  {u[:60]}")
        excluded_mask |= m

    df_filtered = df[~excluded_mask].reset_index(drop=True)
    print(f"\n[INFO] 除外後: {len(df_filtered)} chunks ({len(df_filtered) - len(df):+d})")
    print()

    before = df.groupby("species").size()
    after = df_filtered.groupby("species").size()
    print(f"{'species':22s} {'before':>7s} {'after':>7s} {'diff':>7s}")
    for sp in sorted(before.index):
        b = int(before[sp])
        a = int(after.get(sp, 0))
        diff = a - b
        marker = " *" if diff != 0 else ""
        print(f"{sp:22s} {b:7d} {a:7d} {diff:+7d}{marker}")

    if args.dry_run:
        print("\n[DRY-RUN] ファイル書き換えなし")
        return

    df_filtered.to_csv(train_csv, index=False)
    print(f"\n[OK] {train_csv} を上書き")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
