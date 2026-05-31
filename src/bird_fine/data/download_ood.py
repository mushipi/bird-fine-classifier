"""OOD テスト用音声を species_taxonomy.yaml からダウンロードし 3s チャンクに前処理する。

data/ood/tier{N}/{Species}/  ← 生 MP3
data/ood_processed/tier{N}/{Species}/  ← 3s WAV チャンク

使い方:
    uv run python -m bird_fine.data.download_ood               # 全 tier DL + 前処理
    uv run python -m bird_fine.data.download_ood --metadata-only
    uv run python -m bird_fine.data.download_ood --tiers 1 3   # Tier1,3 のみ
    uv run python -m bird_fine.data.download_ood --no-preprocess  # DL のみ
    uv run python -m bird_fine.data.download_ood --preprocess-only  # 前処理のみ（再実行用）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

from xcapi.client import XenoCantoClient
from xcapi.downloader import Downloader
from xcapi.query import QueryBuilder

TAXONOMY_PATH = PROJECT_ROOT / "species_taxonomy.yaml"
OOD_RAW_DIR = PROJECT_ROOT / "data" / "ood"
OOD_PROCESSED_DIR = PROJECT_ROOT / "data" / "ood_processed"


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chunk_audio(
    audio_path: Path,
    out_dir: Path,
    sample_rate: int,
    chunk_duration: float,
    min_duration: float,
) -> int:
    """MP3 を読んで 3s WAV チャンクに分割して out_dir に保存。チャンク数を返す。"""
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"    [WARN] 読み込み失敗: {audio_path.name} — {e}")
        return 0

    chunk_samples = int(chunk_duration * sample_rate)
    min_samples = int(min_duration * sample_rate)
    n = len(audio)

    if n < min_samples:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    count = 0

    for start in range(0, n, chunk_samples):
        end = start + chunk_samples
        if end <= n:
            chunk = audio[start:end]
        else:
            remaining = n - start
            if remaining < min_samples:
                break
            chunk = np.zeros(chunk_samples, dtype=np.float32)
            chunk[:remaining] = audio[start:n]

        out_path = out_dir / f"{stem}_chunk{count:03d}.wav"
        sf.write(str(out_path), chunk, sample_rate, subtype="PCM_16")
        count += 1

    return count


def download_tier(
    client: XenoCantoClient,
    species_list: list[dict],
    tier: int,
    quality: str,
    max_per_species: int,
    metadata_only: bool,
) -> dict[str, int]:
    """1 tier 分の種をダウンロード。{species_en: count} を返す。"""
    results: dict[str, int] = {}
    tier_dir = OOD_RAW_DIR / f"tier{tier}"

    for sp in species_list:
        en = sp["en"]
        ja = sp.get("ja", "")
        print(f"\n  [{en}] ({ja})")
        species_dir = tier_dir / en.replace(" ", "_")

        # Japan → worldwide フォールバック
        recorded = 0
        for country in ["Japan", None]:
            qb = QueryBuilder().group("birds").english_name(en).quality(quality)
            if country:
                qb = qb.country(country)
            query = qb.build()
            scope = country or "worldwide"
            print(f"    query({scope}): {query}")

            try:
                recordings = client.search(query)
            except Exception as e:
                print(f"    [WARN] search error: {e}")
                time.sleep(1)
                continue

            if not recordings:
                time.sleep(1)
                continue

            recordings = sorted(recordings, key=lambda r: int(r.get("id", 0)))[:max_per_species]
            print(f"    -> {len(recordings)} 件")

            downloader = Downloader(output_dir=str(species_dir))
            if metadata_only:
                downloader.save_metadata_only(recordings)
            else:
                downloader.download_recordings(recordings, verbose=False)

            recorded = len(recordings)
            time.sleep(1)
            break

        results[en] = recorded

    return results


def preprocess_tier(
    tier: int,
    species_list: list[dict],
    sample_rate: int,
    chunk_duration: float,
    min_duration: float,
) -> dict[str, int]:
    """1 tier 分の生音声を 3s チャンクに前処理。{species_en: chunk_count} を返す。"""
    results: dict[str, int] = {}
    tier_raw = OOD_RAW_DIR / f"tier{tier}"
    tier_proc = OOD_PROCESSED_DIR / f"tier{tier}"

    for sp in species_list:
        en = sp["en"]
        species_raw = tier_raw / en.replace(" ", "_")
        species_proc = tier_proc / en.replace(" ", "_")

        if not species_raw.exists():
            print(f"  [SKIP] {en}: data/ood/tier{tier}/{en.replace(' ', '_')}/ が存在しない")
            results[en] = 0
            continue

        # xcapi は {Species}/{Scientific_name}/ にファイルを置くので再帰検索
        mp3_files = (
            list(species_raw.rglob("*.mp3"))
            + list(species_raw.rglob("*.wav"))
            + list(species_raw.rglob("*.flac"))
        )
        if not mp3_files:
            print(f"  [SKIP] {en}: 音声ファイルなし")
            results[en] = 0
            continue

        total_chunks = 0
        for f in tqdm(mp3_files, desc=f"  {en}", leave=False):
            total_chunks += _chunk_audio(f, species_proc, sample_rate, chunk_duration, min_duration)

        print(f"  {en}: {len(mp3_files)} 録音 → {total_chunks} チャンク")
        results[en] = total_chunks

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
        help="対象 tier (デフォルト: 1 2 3)",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--no-preprocess", action="store_true", help="DL のみ（前処理スキップ）")
    parser.add_argument("--preprocess-only", action="store_true", help="前処理のみ（DL スキップ）")
    parser.add_argument(
        "--max-per-species", type=int, default=30,
        help="種あたり最大録音数 (デフォルト: 30)",
    )
    parser.add_argument("--quality", type=str, default="A", help="Xeno-canto 品質フィルタ")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    config = load_config()
    pp = config["preprocessing"]
    sample_rate = int(pp["sample_rate"])
    chunk_duration = float(pp["chunk_duration_sec"])
    min_duration = float(pp["min_chunk_duration_sec"])

    ood = taxonomy["ood_species"]
    tier_map = {1: ood["tier1"], 2: ood["tier2"], 3: ood["tier3"]}

    print(f"[CFG] sample_rate={sample_rate}Hz / chunk={chunk_duration}s / min={min_duration}s")
    print(f"[CFG] tiers={args.tiers} / quality={args.quality} / max={args.max_per_species}/種")
    print(f"[CFG] 出力先: {OOD_RAW_DIR} / {OOD_PROCESSED_DIR}")

    if not args.preprocess_only:
        api_key = os.environ.get("XENO_CANTO_API_KEY")
        if not api_key:
            print("[ERROR] XENO_CANTO_API_KEY が未設定。.env を確認して。")
            sys.exit(1)
        client = XenoCantoClient(api_key=api_key)

        for tier in args.tiers:
            species_list = tier_map[tier]
            print(f"\n{'='*60}")
            print(f"[DL] Tier{tier} — {len(species_list)} 種")
            print(f"{'='*60}")
            results = download_tier(
                client, species_list, tier, args.quality, args.max_per_species, args.metadata_only,
            )
            ok = sum(1 for v in results.values() if v > 0)
            print(f"\n  Tier{tier} 完了: {ok}/{len(results)} 種で録音取得")

    if not args.no_preprocess and not args.metadata_only:
        for tier in args.tiers:
            species_list = tier_map[tier]
            print(f"\n{'='*60}")
            print(f"[PREP] Tier{tier} — {len(species_list)} 種")
            print(f"{'='*60}")
            results = preprocess_tier(tier, species_list, sample_rate, chunk_duration, min_duration)
            total = sum(results.values())
            print(f"\n  Tier{tier} 前処理完了: 合計 {total} チャンク")

    print("\n[OK] 完了")
    print(f"  生音声: {OOD_RAW_DIR}")
    print(f"  チャンク: {OOD_PROCESSED_DIR}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
