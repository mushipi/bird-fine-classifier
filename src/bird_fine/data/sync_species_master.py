"""iNaturalist API から北部九州の観察記録を取得して species_master.csv を更新する。

新規観察種は status="candidate" として追記する。既存種は obs_count と
last_observed を更新するのみで、手動設定済みの status / group 等は変更しない。

使い方:
    uv run python -m bird_fine.data.sync_species_master
    uv run python -m bird_fine.data.sync_species_master --dry-run    # 件数確認のみ
    uv run python -m bird_fine.data.sync_species_master --limit 200  # 取得上限を変更
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MASTER_PATH = PROJECT_ROOT / "data" / "species_master.csv"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

INAT_API = "https://api.inaturalist.org/v1/observations/species_counts"
AVES_TAXON_ID = 3  # 鳥綱


def load_bbox() -> dict:
    """config.yaml から bbox を読む。なければ北部九州デフォルトを使用。"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # BirdProject 形式と bird-fine-classifier 形式の両方に対応
        bbox = cfg.get("northern_kyushu_bbox") or cfg.get("bbox")
        if bbox:
            return bbox
    # デフォルト: 北部九州
    return {"lat_min": 32.0, "lon_min": 128.5, "lat_max": 34.0, "lon_max": 132.0}


def fetch_inat_species(bbox: dict, limit: int = 200) -> list[dict]:
    """iNaturalist から research grade 観察の種リストを取得する。"""
    params = {
        "taxon_id": AVES_TAXON_ID,
        "swlat": bbox["lat_min"],
        "swlng": bbox["lon_min"],
        "nelat": bbox["lat_max"],
        "nelng": bbox["lon_max"],
        "verifiable": "true",
        "quality_grade": "research",
        "per_page": min(limit, 500),
    }
    print(f"[iNAT] bbox: ({bbox['lat_min']},{bbox['lon_min']}) → ({bbox['lat_max']},{bbox['lon_max']})")
    print(f"[iNAT] 上限: {limit} 種")

    resp = requests.get(INAT_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        taxon = item["taxon"]
        sci = taxon.get("name", "")
        en = taxon.get("preferred_common_name") or taxon.get("english_common_name", "")
        taxon_id = taxon.get("id")
        obs_count = item.get("count", 0)
        if sci:
            results.append({
                "taxon_id": taxon_id,
                "sci": sci,
                "en_inat": en,
                "obs_count": obs_count,
            })

    print(f"[iNAT] {len(results)} 種を取得")
    return results


def load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        print(f"[WARN] {MASTER_PATH} が存在しない。新規作成する。")
        return pd.DataFrame(columns=[
            "taxon_id", "sci", "en_inat", "en_birdnet", "ja",
            "family", "order", "obs_count", "last_observed",
            "status", "group", "birdproject", "data_source", "notes",
        ])
    return pd.read_csv(MASTER_PATH, dtype=str).fillna("")


def sync(inat_records: list[dict], master: pd.DataFrame, dry_run: bool) -> pd.DataFrame:
    """iNaturalist 取得結果を master に反映する。"""
    today = date.today().isoformat()
    new_rows = []
    updated = 0

    for rec in inat_records:
        sci = rec["sci"]
        mask = master["sci"] == sci

        if mask.any():
            # 既存種: obs_count と last_observed のみ更新
            master.loc[mask, "obs_count"] = str(rec["obs_count"])
            master.loc[mask, "last_observed"] = today
            # taxon_id が空なら補完
            if master.loc[mask, "taxon_id"].iloc[0] == "":
                master.loc[mask, "taxon_id"] = str(rec["taxon_id"] or "")
            updated += 1
        else:
            # 新規種: candidate として追加
            new_rows.append({
                "taxon_id": str(rec["taxon_id"] or ""),
                "sci": sci,
                "en_inat": rec["en_inat"],
                "en_birdnet": rec["en_inat"],  # 初期値は en_inat と同じ（要手動確認）
                "ja": "",
                "family": "",
                "order": "",
                "obs_count": str(rec["obs_count"]),
                "last_observed": today,
                "status": "candidate",
                "group": "",
                "birdproject": "",
                "data_source": "xeno-canto",
                "notes": "",
            })

    print(f"\n[SYNC] 既存種の更新: {updated} 種")
    print(f"[SYNC] 新規 candidate: {len(new_rows)} 種")

    if new_rows:
        print("\n新規追加種:")
        for r in new_rows:
            print(f"  {r['sci']:40s}  {r['en_inat']}")

    if dry_run:
        print("\n[DRY-RUN] ファイル書き換えなし")
        return master

    if new_rows:
        master = pd.concat(
            [master, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    return master


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=200, help="iNaturalist 取得上限 (default: 200)")
    args = parser.parse_args()

    bbox = load_bbox()
    master = load_master()
    print(f"[MASTER] 現在: {len(master)} 種 ({MASTER_PATH})")

    inat_records = fetch_inat_species(bbox, args.limit)
    master = sync(inat_records, master, dry_run=args.dry_run)

    if not args.dry_run:
        master.to_csv(MASTER_PATH, index=False)
        print(f"\n[OK] {MASTER_PATH} を更新 ({len(master)} 種)")

    # ステータス集計を表示
    print("\n[STATUS 集計]")
    counts = master["status"].value_counts()
    for status, count in counts.items():
        print(f"  {status:12s}: {count} 種")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
