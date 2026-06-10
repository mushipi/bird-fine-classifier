"""Perch 2.0 (perch_v2_cpu) 埋め込み抽出器。

隔離環境（tools/perch_embed/.venv: tensorflow-cpu + perch-hoplite）で実行する。
split CSV を駆動し、各チャンクを Perch 埋め込み(1536次元, 窓平均)に変換して
data/embeddings/perch/{split}.npz にキャッシュする。

モデルは perch_hoplite が CPU 機では自動で perch_v2_cpu を Kaggle(public, 認証不要)から
取得し KAGGLEHUB_CACHE にキャッシュする。GPU専用エクスポート(HFミラー)はCPUで動かないため使わない。

実行例:
  cd tools/perch_embed
  .venv/bin/python extract_perch.py --splits train val test --source raw
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 既定の kagglehub キャッシュ（references 配下に固定）
os.environ.setdefault(
    "KAGGLEHUB_CACHE",
    str(Path(__file__).resolve().parents[2] / "references" / "weights" / "kagglehub_cache"),
)

# repo の src を import パスへ（bird_fine.embeddings.io_utils を使う）
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import librosa  # noqa: E402
from bird_fine.embeddings import io_utils  # noqa: E402

PERCH_SR = 32_000


def load_perch():
    from perch_hoplite.zoo import model_configs as mc
    print("[load] perch_v2 (CPU variant, auto-download if needed)...", flush=True)
    model = mc.load_model_by_name("perch_v2")
    print(f"[ok] sample_rate={model.sample_rate} window_s={model.window_size_s}", flush=True)
    return model


def embed_chunk(model, chunk_32k: np.ndarray) -> np.ndarray:
    """10秒チャンク(32kHz)→ Perch埋め込み(1536,)。内部で5秒窓に分割→窓平均。"""
    out = model.embed(chunk_32k.astype(np.float32))
    emb = np.asarray(out.embeddings)  # (num_windows, channels, 1536)
    return emb.reshape(-1, emb.shape[-1]).mean(axis=0)


def extract_split(model, df, label_map, raw_dir: Path, source: str) -> dict:
    """1 split を抽出。録音単位で生音源を1回ロードし、各 chunk_index を埋め込む。"""
    feats: list[np.ndarray] = []
    ys, xcs, cidx, specs = [], [], [], []
    missing = 0

    # 録音単位（species, source_file）でグループ化して I/O を最小化
    for (species, source_file), grp in df.groupby(["species", "source_file"], sort=False):
        if source == "raw":
            audio_path = io_utils.raw_audio_path(species, source_file, raw_dir)
            if not audio_path.exists():
                missing += len(grp)
                continue
            try:
                audio, _ = librosa.load(str(audio_path), sr=PERCH_SR, mono=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] {audio_path.name}: {e}", flush=True)
                missing += len(grp)
                continue
            chunks = io_utils.chunk_audio(audio, PERCH_SR, 10.0, 3.0, 0.0)
            for _, row in grp.iterrows():
                i = int(row["chunk_index"])
                if i >= len(chunks):
                    missing += 1
                    continue
                feats.append(embed_chunk(model, chunks[i]))
                ys.append(label_map[species]); xcs.append(row["xc_id"])
                cidx.append(i); specs.append(species)
        else:  # source == "processed": 既存16kHzチャンクを32kHzへ resample
            for _, row in grp.iterrows():
                p = io_utils.normalize_rel_path(row["file_path"])
                if not p.exists():
                    missing += 1
                    continue
                audio, _ = librosa.load(str(p), sr=PERCH_SR, mono=True)
                feats.append(embed_chunk(model, audio))
                ys.append(label_map[species]); xcs.append(row["xc_id"])
                cidx.append(int(row["chunk_index"])); specs.append(species)

    return dict(
        X=np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 1536), np.float32),
        y=np.asarray(ys, dtype=np.int64),
        xc_id=np.asarray(xcs, dtype=object),
        chunk_index=np.asarray(cidx, dtype=np.int64),
        species=np.asarray(specs, dtype=object),
        missing=missing,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", default=str(REPO_ROOT / "data" / "splits"))
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "embeddings" / "perch"))
    ap.add_argument("--source", choices=["raw", "processed"], default="raw",
                    help="raw=生mp3を32kHz再デコード(高忠実) / processed=既存16kHzを32kHz resample")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    label_map = io_utils.load_label_map(splits_dir)
    label_names = io_utils.label_names_from_map(label_map)
    model = load_perch()

    for split in args.splits:
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            print(f"[SKIP] {csv} なし", flush=True)
            continue
        df = io_utils.read_split(csv)
        print(f"[{split}] {len(df)} chunks 抽出中 (source={args.source})...", flush=True)
        res = extract_split(model, df, label_map, Path(args.raw_dir), args.source)
        out_path = Path(args.out_dir) / f"{split}.npz"
        io_utils.save_embeddings(
            out_path, res["X"], res["y"], res["xc_id"], res["chunk_index"],
            res["species"], label_names, "perch_v2", X_seq=None,
        )
        print(f"  [OK] {res['X'].shape} -> {out_path}  (missing={res['missing']})", flush=True)


if __name__ == "__main__":
    main()
