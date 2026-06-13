"""カモ類8種の音声データをXeno-cantoからダウンロード。

BirdProject/scripts/01_download_data.py をベースに簡略化。
県別bboxは使わず、Japan→worldwideのフォールバック戦略のみ。

使い方:
    uv run python -m bird_fine.data.download                 # 全種DL
    uv run python -m bird_fine.data.download --metadata-only # メタデータのみ
    uv run python -m bird_fine.data.download --species Mallard "Common Teal"
    # 追加収集（既存 XCID を除外して quality=B から10本ずつ）
    uv run python -m bird_fine.data.download --species "Tufted Duck" "Eurasian Wigeon" \
        --quality B --exclude-existing --max-per-species 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

from xcapi.client import XenoCantoClient
from xcapi.downloader import Downloader
from xcapi.query import QueryBuilder


def load_config(config_path=None) -> dict:
    path = config_path or (PROJECT_ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_existing_ids(species_dir: str, output_dir: str) -> set[str]:
    """既存の metadata.csv（実際に DL 済み）から XCID を集める。

    metadata_only.csv は「メタのみ取得した候補」であり実 DL ではないため除外しない。
    """
    ids: set[str] = set()
    meta = Path(output_dir) / species_dir / "metadata.csv"
    if not meta.exists():
        return ids
    with open(meta, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("id")
            if rid:
                ids.add(str(rid))
    return ids


def _build_query(species_en: str, quality: str, country: str | None) -> str:
    qb = QueryBuilder().group("birds").english_name(species_en).quality(quality)
    if country:
        qb = qb.country(country)
    return qb.build()


def _download(
    client: XenoCantoClient,
    species_en: str,
    quality: str,
    country: str | None,
    output_dir: str,
    metadata_only: bool,
    max_recordings: int,
    exclude_ids: set[str] | None = None,
) -> int:
    scope = country if country else "worldwide"
    # quality は単一グレード前提（Xeno-canto は q:"A B" を受け付けず常に0件）。
    # 複数グレード（"A B"）はグレード毎に検索し、XCID で統合（dedup）する。
    merged: dict[str, dict] = {}
    for grade in quality.split():
        q = _build_query(species_en, grade, country)
        print(f"  query({scope}): {q}")
        try:
            recs = client.search(q)
        except Exception as e:
            print(f"  [WARN] search error: {e}")
            recs = []
        for r in (recs or []):
            merged[str(r.get("id"))] = r
    recordings = list(merged.values())

    if not recordings:
        print(f"  -> no recordings")
        return 0

    if exclude_ids:
        before = len(recordings)
        recordings = [r for r in recordings if str(r.get("id")) not in exclude_ids]
        skipped = before - len(recordings)
        if skipped:
            print(f"  -> excluded {skipped} existing recordings (kept {len(recordings)})")

    if not recordings:
        print(f"  -> no new recordings after exclusion")
        return 0

    # 再現性のため XCID 昇順で安定ソート
    recordings = sorted(recordings, key=lambda r: int(r.get("id", 0)))

    if len(recordings) > max_recordings:
        recordings = recordings[:max_recordings]

    print(f"  -> {len(recordings)} recordings found")

    species_dir = species_en.replace(" ", "_")
    species_output = os.path.join(output_dir, species_dir)
    downloader = Downloader(output_dir=species_output)

    if metadata_only:
        downloader.save_metadata_only(recordings)
        print(f"  -> metadata saved")
    else:
        downloader.download_recordings(recordings, verbose=True)
        print(f"  -> download complete")

    return len(recordings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="対象種（英名）。省略時はconfig.yamlの全種",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="メタデータのみ取得（音声ファイルはDLしない）",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=None,
        help="種あたりの最大録音数",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        help="quality 上書き（例: B）。省略時は config.yaml の download.quality を使用",
    )
    parser.add_argument(
        "--exclude-existing",
        action="store_true",
        help="data/raw/{Species}/metadata*.csv に既出の XCID を除外して追加収集モードで動く",
    )
    parser.add_argument(
        "--worldwide-only",
        action="store_true",
        help="Japan を試さず worldwide だけ叩く。追加収集で地域多様性を取りたい時に使う",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="設定ファイル（省略時 config.yaml）。群別は config-<group>.yaml を指定",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dl_cfg = config["download"]
    output_dir = str(PROJECT_ROOT / dl_cfg["output_dir"])
    quality = args.quality if args.quality else dl_cfg["quality"]
    if args.worldwide_only:
        countries = []
        fallback_worldwide = True
    else:
        countries = dl_cfg.get("countries", ["Japan"])
        fallback_worldwide = dl_cfg.get("fallback_worldwide", True)
    max_recordings = args.max_per_species or dl_cfg.get("max_recordings_per_species", 100)

    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        print("[ERROR] XENO_CANTO_API_KEY が未設定。.env を確認して。")
        sys.exit(1)
    print(f"[KEY] {api_key[:8]}...{api_key[-4:]}")

    client = XenoCantoClient(api_key=api_key)

    if args.species:
        species_list = [{"en": s} for s in args.species]
    else:
        species_list = config["target_species"]

    print(f"[INFO] 対象種数: {len(species_list)}")
    print(f"[INFO] 品質: {quality}")
    print(f"[INFO] 出力先: {output_dir}")
    print(f"[INFO] 検索範囲: {', '.join(countries)}" + (" -> worldwide" if fallback_worldwide else ""))
    print(f"[INFO] {'メタデータのみ' if args.metadata_only else 'ダウンロードモード'}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []
    total = 0

    for i, sp in enumerate(species_list, 1):
        species_en = sp["en"]
        ja = sp.get("ja", "")
        print(f"\n[{i}/{len(species_list)}] {species_en} ({ja})")

        species_dir = species_en.replace(" ", "_")
        exclude_ids: set[str] | None = None
        if args.exclude_existing:
            exclude_ids = _load_existing_ids(species_dir, output_dir)
            print(f"  exclude-existing: {len(exclude_ids)} XCID をスキップ対象に")

        count = 0
        for country in countries:
            n = _download(
                client=client,
                species_en=species_en,
                quality=quality,
                country=country,
                output_dir=output_dir,
                metadata_only=args.metadata_only,
                max_recordings=max_recordings,
                exclude_ids=exclude_ids,
            )
            count += n
            time.sleep(1)
            if count > 0:
                break

        if count == 0 and fallback_worldwide:
            print(f"  [WARN] 指定国で見つからず。worldwideで検索")
            n = _download(
                client=client,
                species_en=species_en,
                quality=quality,
                country=None,
                output_dir=output_dir,
                metadata_only=args.metadata_only,
                max_recordings=max_recordings,
                exclude_ids=exclude_ids,
            )
            count += n
            time.sleep(1)

        total += count
        results.append({"species": species_en, "ja": ja, "count": count})

    print("\n" + "=" * 60)
    print("[SUMMARY] ダウンロード結果")
    print("=" * 60)
    for r in results:
        status = "[OK]  " if r["count"] > 0 else "[WARN]"
        print(f"  {status} {r['species']} ({r['ja']}): {r['count']} 件")
    print(f"\n  合計: {total} 件")
    print(f"  録音なし: {sum(1 for r in results if r['count'] == 0)} 種")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
