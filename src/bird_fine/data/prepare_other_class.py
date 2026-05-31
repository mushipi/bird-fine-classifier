"""OOD 録音から "other" クラスのチャンクを準備し、train.csv / val.csv に追記する。

録音単位で train20 / eval10 に分割し、per-recording cap を適用することで
過学習と recording-level leakage を同時に防ぐ。

使い方:
    uv run python -m bird_fine.data.prepare_other_class
    uv run python -m bird_fine.data.prepare_other_class --dry-run   # 件数確認のみ
    uv run python -m bird_fine.data.prepare_other_class --reset     # 追記済み other を削除してやり直し
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OOD_PROCESSED_DIR = PROJECT_ROOT / "data" / "ood_processed"
TAXONOMY_PATH = PROJECT_ROOT / "species_taxonomy.yaml"


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_by_recording(wav_files: list[Path]) -> dict[str, list[Path]]:
    """chunk ファイルを録音単位でグループ化。stem から _chunkNNN.wav を除去して ID とする。"""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in wav_files:
        rec_id = re.sub(r"_chunk\d+\.wav$", "", f.name)
        groups[rec_id].append(f)
    return {k: sorted(v) for k, v in groups.items()}


def collect_other_chunks(
    ood_spec: list[dict],
    tier: int,
    n_train_rec: int,
    n_eval_rec: int,
    cap_per_rec: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """1 tier 分の OOD 種から train / eval チャンクリストを返す。"""
    train_rows, val_rows = [], []

    for sp in ood_spec:
        en = sp["en"]
        species_dir = OOD_PROCESSED_DIR / f"tier{tier}" / en.replace(" ", "_")
        if not species_dir.exists():
            print(f"  [SKIP] {en}: {species_dir} が存在しない（先に download_ood.py を実行して）")
            continue

        wavs = sorted(species_dir.rglob("*.wav"))
        if not wavs:
            print(f"  [SKIP] {en}: WAV チャンクなし")
            continue

        groups = group_by_recording(wavs)
        rec_ids = sorted(groups.keys())
        rng.shuffle(rec_ids)

        train_recs = rec_ids[:n_train_rec]
        eval_recs = rec_ids[n_train_rec: n_train_rec + n_eval_rec]

        for recs, rows in [(train_recs, train_rows), (eval_recs, val_rows)]:
            for rec_id in recs:
                chunks = groups[rec_id][:cap_per_rec]
                for i, f in enumerate(chunks):
                    rows.append({
                        "species": "other",
                        "xc_id": rec_id,
                        "chunk_index": i,
                        "file_path": str(f.relative_to(PROJECT_ROOT)),
                        "duration_sec": 3.0,
                        "source_file": rec_id,
                    })

        print(
            f"  {en} (tier{tier}): "
            f"train {len(train_recs)} 録音 / eval {len(eval_recs)} 録音 "
            f"(cap={cap_per_rec}/rec)"
        )

    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="件数確認のみ（CSV 書き換えなし）")
    parser.add_argument("--reset", action="store_true", help="既追記の other 行を削除してやり直し")
    args = parser.parse_args()

    config = load_config()
    taxonomy = load_taxonomy()
    pp = config["preprocessing"]
    oc = config.get("other_class", {})

    n_train_rec = int(oc.get("recordings_for_train", 20))
    n_eval_rec = int(oc.get("recordings_for_eval", 10))
    cap_per_rec = int(oc.get("max_chunks_per_recording", 30))
    seed = int(oc.get("seed", 42))
    tier1_names = set(oc.get("tier1_species", []))
    tier2_names = set(oc.get("tier2_species", []))

    rng = random.Random(seed)

    splits_dir = PROJECT_ROOT / pp["splits_dir"]
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    label_map_csv = splits_dir / "label_map.csv"

    # --reset: other 行を削除して clean state に戻す
    if args.reset:
        for csv_path in [train_csv, val_csv]:
            df = pd.read_csv(csv_path)
            before = len(df)
            df = df[df["species"] != "other"].reset_index(drop=True)
            if not args.dry_run:
                df.to_csv(csv_path, index=False)
            print(f"  {csv_path.name}: {before} → {len(df)} 行（{before - len(df)} 件削除）")
        lm = pd.read_csv(label_map_csv)
        lm = lm[lm["species"] != "other"].reset_index(drop=True)
        if not args.dry_run:
            lm.to_csv(label_map_csv, index=False)
        print("  label_map.csv から other を削除")
        if args.dry_run:
            print("[DRY-RUN] ファイル書き換えなし")
        return

    # 対象 OOD 種を tier1 / tier2 から抽出
    ood = taxonomy["ood_species"]
    tier1_specs = [s for s in ood["tier1"] if s["en"] in tier1_names] if tier1_names else ood["tier1"]
    tier2_specs = [s for s in ood["tier2"] if s["en"] in tier2_names] if tier2_names else ood["tier2"]

    print(f"[CFG] train録音/種={n_train_rec} / eval録音/種={n_eval_rec} / cap={cap_per_rec}/録音")
    print(f"[CFG] Tier1 対象: {[s['en'] for s in tier1_specs]}")
    print(f"[CFG] Tier2 対象: {[s['en'] for s in tier2_specs]}")

    print("\n[COLLECT] Tier1 —")
    t1_train, t1_val = collect_other_chunks(tier1_specs, 1, n_train_rec, n_eval_rec, cap_per_rec, rng)

    print("\n[COLLECT] Tier2 —")
    t2_train, t2_val = collect_other_chunks(tier2_specs, 2, n_train_rec, n_eval_rec, cap_per_rec, rng)

    all_train = t1_train + t2_train
    all_val = t1_val + t2_val

    print(f"\n[RESULT]")
    print(f"  other train: {len(all_train)} チャンク")
    print(f"  other val:   {len(all_val)} チャンク")

    if args.dry_run:
        print("[DRY-RUN] ファイル書き換えなし")
        return

    # 既存 other 行を除去してから追記（冪等性確保）
    for csv_path, new_rows in [(train_csv, all_train), (val_csv, all_val)]:
        df = pd.read_csv(csv_path)
        df = df[df["species"] != "other"].reset_index(drop=True)
        df_new = pd.DataFrame(new_rows)
        df_merged = pd.concat([df, df_new], ignore_index=True)
        df_merged.to_csv(csv_path, index=False)
        print(f"  {csv_path.name}: {len(df)} + {len(new_rows)} = {len(df_merged)} 行")

    # label_map に "other" を追加
    lm = pd.read_csv(label_map_csv)
    if "other" not in lm["species"].values:
        other_id = int(lm["label_id"].max()) + 1
        lm = pd.concat([lm, pd.DataFrame([{"label_id": other_id, "species": "other"}])], ignore_index=True)
        lm.to_csv(label_map_csv, index=False)
        print(f"  label_map.csv: 'other' → label_id={other_id} を追加")
    else:
        print("  label_map.csv: 'other' は既存")

    print("\n[OK] 完了")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
