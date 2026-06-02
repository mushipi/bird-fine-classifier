"""split CSV (train/val/test) に Xeno-canto 録音メタデータを付与し、
データキュレーション用のレポートを生成する。

背景:
    現行の data/splits/*.csv は species/xc_id/chunk_index/file_path/duration_sec/
    source_file しか持たず、Xeno-canto が返す type/stage/sex/q/length 等の構造化
    メタデータが捨てられていた。このため juvenile distribution shift（run09）や
    長尺録音 XC488112/113（45分/49分）の発見を「耳で聞く」に頼っていた。

    本スクリプトは data/raw/{Species}/metadata.csv（DL時に保存済み）を結合し、
    xc_id → XC番号 経由で各チャンクにメタを付与する。これにより
      - length_sec > 閾値      → 長尺録音を自動フラグ
      - stage == juvenile      → distribution shift 候補を自動フラグ
      - train/test の stage 構成比 → shift の定量化
    が「耳で聞かずに」可能になる。

結合の限界:
    metadata.csv は 454 録音分のみ。初期 DL 分など XC 番号が抽出できても
    metadata.csv に存在しない録音（未一致）は NaN になる。カバー率を報告する。
    未一致分を埋めるには XC API での再取得が別途必要（本スクリプトでは行わない）。

使い方:
    uv run python -m bird_fine.data.enrich_metadata               # enrich + レポート
    uv run python -m bird_fine.data.enrich_metadata --long-min 10 # 長尺閾値10分
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# split に付与するメタデータ列（録音単位の属性のみ）
META_COLS = ["type", "sex", "stage", "q", "length", "length_sec", "cnt", "date", "rec", "lat", "lon"]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_length(val: object) -> float | None:
    """Xeno-canto の length 表記を秒に変換。'2:09'→129, '44:55'→2695, '1:02:03'→3723。"""
    if pd.isna(val):
        return None
    parts = str(val).strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    sec = 0
    for n in nums:  # 左から順に時/分/秒として桁上げ
        sec = sec * 60 + n
    return float(sec)


def extract_xc_number(xc_id: object) -> int | None:
    """split の xc_id 文字列から XC 番号（整数）を抽出。'XC197026'→197026。
    非XC形式（属名等）の古い録音は None。"""
    m = re.search(r"XC?(\d+)", str(xc_id))
    return int(m.group(1)) if m else None


def normalize_filename(name: object) -> str:
    """ファイル名を照合用キーに正規化（拡張子・記号除去, 小文字化）。
    非XC形式の古い録音は xc_id に XC番号が無いため、source_file を
    metadata.csv の file-name 列と突き合わせて XC番号を復元する用途。"""
    s = re.sub(r"\.(mp3|wav)$", "", str(name).lower())
    return re.sub(r"[^a-z0-9]", "", s)


def build_metadata_index() -> tuple[pd.DataFrame, dict[str, int]]:
    """全種の metadata.csv を結合し、(XC番号indexのメタ表, file-name正規化キー→id) を返す。
    後者は非XC形式の古い録音を source_file 照合で復元するためのマップ。"""
    frames = []
    for f in sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "*" / "metadata.csv"))):
        m = pd.read_csv(f)
        m["species_dir"] = Path(f).parent.name
        frames.append(m)
    if not frames:
        raise FileNotFoundError("data/raw/*/metadata.csv が見つかりません")

    meta = pd.concat(frames, ignore_index=True)
    meta["id"] = meta["id"].astype(int)
    meta = meta.drop_duplicates(subset="id", keep="first")
    meta["length_sec"] = meta["length"].map(parse_length)

    fn_map: dict[str, int] = {}
    if "file-name" in meta.columns:
        fn_map = {normalize_filename(fn): int(i) for fn, i in zip(meta["file-name"], meta["id"])}

    keep = ["id"] + [c for c in META_COLS if c in meta.columns]
    return meta[keep].set_index("id"), fn_map


def enrich_split(split_csv: Path, meta_idx: pd.DataFrame, fn_map: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    """1つの split CSV にメタを左結合し、(enriched_df, カバー率統計) を返す。"""
    df = pd.read_csv(split_csv)
    df["xc_number"] = df["xc_id"].map(extract_xc_number)

    # フォールバック: XC番号が抽出できない古い録音は source_file を file-name 照合で復元
    no_id = df["xc_number"].isna()
    n_recovered = 0
    if no_id.any():
        recovered = df.loc[no_id, "source_file"].map(lambda s: fn_map.get(normalize_filename(s)))
        n_recovered = int(recovered.notna().sum())
        df.loc[no_id, "xc_number"] = recovered

    enriched = df.merge(
        meta_idx, how="left", left_on="xc_number", right_index=True, suffixes=("", "_meta")
    )

    n_chunks = len(df)
    n_id_ok = int(df["xc_number"].notna().sum())
    # length_sec はほぼ全 metadata 行に存在するため、結合成功（メタ一致）の判定に使う
    n_matched = int(enriched["length_sec"].notna().sum())

    stats = {
        "split": split_csv.stem,
        "chunks": n_chunks,
        "recordings": int(df["xc_id"].nunique()),
        "id_extracted": n_id_ok,
        "recovered": n_recovered,
        "meta_matched": n_matched,
        "coverage_pct": round(n_matched / n_chunks * 100, 1) if n_chunks else 0.0,
    }
    return enriched, stats


def recording_table(enriched_all: pd.DataFrame) -> pd.DataFrame:
    """全 split の enriched を録音単位に集約（chunk 数・メタ属性）。"""
    agg = (
        enriched_all.groupby(["split", "species", "xc_id"], dropna=False)
        .agg(
            chunks=("chunk_index", "size"),
            length_sec=("length_sec", "first"),
            stage=("stage", "first"),
            q=("q", "first"),
            type=("type", "first"),
            cnt=("cnt", "first"),
        )
        .reset_index()
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long-min", type=float, default=10.0, help="長尺録音フラグの閾値（分）。既定10分")
    parser.add_argument("--no-write", action="store_true", help="*_enriched.csv を書かずレポートのみ")
    args = parser.parse_args()

    config = load_config()
    splits_dir = PROJECT_ROOT / config["preprocessing"]["splits_dir"]
    out_dir = PROJECT_ROOT / "outputs" / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_idx, fn_map = build_metadata_index()
    print(f"[INFO] metadata 索引: {len(meta_idx)} 録音 / file-name照合キー {len(fn_map)} 件\n")

    # ── 各 split を enrich ──────────────────────────────────────────────
    enriched_frames = []
    print(f"{'split':8s} {'chunks':>7s} {'録音':>5s} {'ID抽出':>7s} {'fn復元':>7s} {'メタ一致':>8s} {'カバー率':>8s}")
    print("-" * 60)
    for name in ["train", "val", "test"]:
        csv = splits_dir / f"{name}.csv"
        if not csv.exists():
            print(f"[WARN] {csv} が無いのでスキップ")
            continue
        enriched, st = enrich_split(csv, meta_idx, fn_map)
        enriched["split"] = name
        enriched_frames.append(enriched)
        print(f"{st['split']:8s} {st['chunks']:7d} {st['recordings']:5d} "
              f"{st['id_extracted']:7d} {st['recovered']:7d} {st['meta_matched']:8d} {st['coverage_pct']:7.1f}%")
        if not args.no_write:
            enriched.drop(columns=["xc_number"]).to_csv(splits_dir / f"{name}_enriched.csv", index=False)

    if not enriched_frames:
        print("[ERROR] enrich 対象 split がありません")
        return
    if not args.no_write:
        print(f"\n[OK] data/splits/*_enriched.csv を書き出し")

    all_enr = pd.concat(enriched_frames, ignore_index=True)
    rec = recording_table(all_enr)
    rec.to_csv(out_dir / "recordings.csv", index=False)

    long_thresh = args.long_min * 60

    # ── レポート1: 長尺録音フラグ ───────────────────────────────────────
    print(f"\n{'='*64}\n[レポート1] 長尺録音（length > {args.long_min:.0f}分）— 冗長チャンクの温床\n{'='*64}")
    long_rec = rec[rec["length_sec"] > long_thresh].sort_values("length_sec", ascending=False)
    if len(long_rec):
        print(f"{'split':6s} {'species':20s} {'長さ':>7s} {'chunks':>7s}  xc_id")
        for _, r in long_rec.iterrows():
            mm = f"{int(r['length_sec'])//60}:{int(r['length_sec'])%60:02d}"
            print(f"{r['split']:6s} {str(r['species']):20s} {mm:>7s} {int(r['chunks']):7d}  {str(r['xc_id'])[:40]}")
    else:
        print("該当なし")

    # ── レポート2: 種 × stage クロス集計（録音数）─────────────────────────
    print(f"\n{'='*64}\n[レポート2] 種 × stage クロス集計（録音数）\n{'='*64}")
    rec_uniq = rec.drop_duplicates(subset=["species", "xc_id"])
    ct = pd.crosstab(rec_uniq["species"], rec_uniq["stage"].fillna("(未記載)"))
    print(ct.to_string())

    # ── レポート3: train/test の stage 構成比（distribution shift）─────────
    print(f"\n{'='*64}\n[レポート3] train vs test の stage 構成比（録音単位）— shift 検出\n{'='*64}")
    rec_tt = rec[rec["split"].isin(["train", "test"])].drop_duplicates(subset=["split", "species", "xc_id"])
    shift = pd.crosstab(
        [rec_tt["species"], rec_tt["stage"].fillna("(未記載)")], rec_tt["split"]
    )
    print(shift.to_string())

    print(f"\n[OK] 録音単位テーブル: {out_dir / 'recordings.csv'}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
