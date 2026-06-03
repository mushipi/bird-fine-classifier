"""val/test split を「冬の日本」モニタリングのドメインに整合させる。

背景:
    test の juvenile 録音 XC667403/667392 は 2021-08-05・フランスで録られた雛の
    begging call で、冬の日本では遭遇しない音響パターン（→ journal.md 2026-06-03）。
    これらを評価に含めると Tufted_Duck の見かけの recall が 0.000 に崩れ、本来の
    モニタリング性能を過小評価する。stage=juvenile/nestling をドメイン外として
    val/test から除外する。

設計判断:
    - train は変更しない（学習の音響多様性を保持する。冬の日本に出ない音でも、特徴
      抽出の汎化には寄与しうる）。除外は評価セット（val/test）に限定する。
    - ドメイン定義は stage ベース。月（繁殖期）での除外は adult call まで巻き込んで
      評価を歪めるため採らない（再評価で (C) が悪化したことを確認済み）。

使い方:
    uv run python -m bird_fine.data.filter_domain --dry-run   # 除外内容のみ表示
    uv run python -m bird_fine.data.filter_domain             # val/test を上書き（.bak 退避）
    uv run python -m bird_fine.data.filter_domain --splits test
"""
from __future__ import annotations

import argparse
import shutil
import sys

import pandas as pd

from bird_fine.data.enrich_metadata import (
    PROJECT_ROOT,
    build_metadata_index,
    enrich_split,
    load_config,
)

# 元の split csv が持つ列（メタ付与前に戻すため）
ORIG_COLS = ["species", "xc_id", "chunk_index", "file_path", "duration_sec", "source_file"]
# ドメイン外とみなす stage キーワード（部分一致・小文字）
OUT_OF_DOMAIN_STAGES = ("juvenile", "nestling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="対象 split（既定: val test）")
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き換えせず除外内容のみ表示")
    args = parser.parse_args()

    config = load_config()
    splits_dir = PROJECT_ROOT / config["preprocessing"]["splits_dir"]
    meta_idx, fn_map = build_metadata_index()

    for name in args.splits:
        csv = splits_dir / f"{name}.csv"
        if not csv.exists():
            print(f"[WARN] {csv} が無いのでスキップ")
            continue

        enriched, _ = enrich_split(csv, meta_idx, fn_map)
        stage = enriched["stage"].fillna("").str.lower()
        mask = pd.Series(False, index=enriched.index)
        for kw in OUT_OF_DOMAIN_STAGES:
            mask |= stage.str.contains(kw, na=False)

        removed = enriched[mask]
        kept = enriched[~mask]
        print(f"\n=== {name}.csv: {len(enriched)} chunks / {enriched['xc_id'].nunique()} 録音 ===")
        print(f"  ドメイン外(stage∋juvenile/nestling)除外: {int(mask.sum())} chunks / "
              f"{removed['xc_id'].nunique()} 録音")
        if len(removed):
            agg = (removed.groupby(["xc_id", "species", "stage"])
                   .size().reset_index(name="chunks").sort_values("chunks", ascending=False))
            for _, r in agg.iterrows():
                print(f"    {int(r['chunks']):4d} chunks  {str(r['species']):20s} "
                      f"stage={str(r['stage']):16s} {str(r['xc_id'])[:40]}")
        print(f"  → 除外後: {len(kept)} chunks / {kept['xc_id'].nunique()} 録音")

        if args.dry_run:
            continue

        # メタ列を落として元の列構成に戻して書き戻す（.bak 退避）
        out = kept[ORIG_COLS].reset_index(drop=True)
        shutil.copy(csv, csv.with_suffix(".csv.bak"))
        out.to_csv(csv, index=False)
        print(f"  [OK] {csv} を上書き（元ファイルは {csv.name}.bak に退避）")

    if args.dry_run:
        print("\n[DRY-RUN] ファイル書き換えなし")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
