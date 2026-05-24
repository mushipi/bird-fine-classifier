"""誤分類チャンクと正解チャンクのメルスペクトログラムを並べて視覚比較。

run08 で「Tufted_Duck と予測されて Eurasian_Teal だった13件のうち12件が同じ録音
XC197026 から」「Wigeon 4件は全て XC349677 から」と判明。「音響的類似」か
「録音の汚染/誤ラベル」かを目視で切り分けるために使う。

使い方:
    uv run python -m bird_fine.analysis.confusion_audio --eval-dir outputs/eval_20260524_174742
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 16000
N_MELS = 128


def load_merged(eval_dir: Path) -> pd.DataFrame:
    """predictions.csv と test.csv を順序ベースで結合し、true/pred ラベル名を付与。"""
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv").reset_index(drop=True)
    preds = pd.read_csv(eval_dir / "predictions.csv").reset_index(drop=True)
    label_map = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "label_map.csv")
    id2sp = dict(zip(label_map["label_id"], label_map["species"]))

    if len(test_df) != len(preds):
        raise ValueError(f"test.csv ({len(test_df)}) と predictions.csv ({len(preds)}) の長さが不一致")

    merged = pd.concat([test_df, preds], axis=1)
    merged["true"] = merged["y_true"].map(id2sp)
    merged["pred"] = merged["y_pred"].map(id2sp)
    return merged


def mel_spec(audio_path: Path) -> np.ndarray:
    """10秒チャンク wav からメルスペクトログラム (dB) を作る。"""
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=sr // 2)
    return librosa.power_to_db(mel, ref=np.max)


def plot_group(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    n_cols: int = 4,
    max_samples: int = 8,
) -> None:
    """グループ内のチャンクを最大 max_samples 件並べる。"""
    sub = df.head(max_samples).reset_index(drop=True)
    n = len(sub)
    if n == 0:
        print(f"  [SKIP] {title}: 該当チャンクなし")
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i >= n:
            ax.axis("off")
            continue
        row = sub.iloc[i]
        audio_path = PROJECT_ROOT / row["file_path"]
        try:
            mel = mel_spec(audio_path)
        except Exception as e:
            ax.set_title(f"ERR: {e}")
            ax.axis("off")
            continue
        librosa.display.specshow(
            mel,
            sr=SAMPLE_RATE,
            x_axis="time",
            y_axis="mel",
            fmax=SAMPLE_RATE // 2,
            ax=ax,
            cmap="magma",
        )
        xc_short = row["xc_id"][:30]
        ax.set_title(
            f"{xc_short}#{row['chunk_index']}\ntrue={row['true']} pred={row['pred']}",
            fontsize=8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] {title} -> {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="evaluate.py が生成した outputs/eval_YYYYMMDD_HHMMSS ディレクトリ",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先（省略時は --eval-dir/confusion_audio）",
    )
    parser.add_argument("--max-samples", type=int, default=8, help="グループあたり最大何件並べるか")
    args = parser.parse_args()

    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else (PROJECT_ROOT / args.eval_dir)
    out_dir = args.out_dir if args.out_dir else (eval_dir / "confusion_audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged(eval_dir)
    print(f"[INFO] merged: {len(df)} rows")

    # 主要な分析対象グループ
    groups = [
        ("A_tufted_TP", "Tufted_Duck 正解 (TP)", df[(df["pred"] == "Tufted_Duck") & (df["true"] == "Tufted_Duck")]),
        ("B_teal_FP_as_tufted", "Eurasian_Teal が Tufted_Duck と誤分類", df[(df["pred"] == "Tufted_Duck") & (df["true"] == "Eurasian_Teal")]),
        ("C_wigeon_FP_as_tufted", "Eurasian_Wigeon が Tufted_Duck と誤分類", df[(df["pred"] == "Tufted_Duck") & (df["true"] == "Eurasian_Wigeon")]),
        ("D_teal_TP", "Eurasian_Teal 正解 (TP)", df[(df["pred"] == "Eurasian_Teal") & (df["true"] == "Eurasian_Teal")]),
        ("E_wigeon_TP", "Eurasian_Wigeon 正解 (TP)", df[(df["pred"] == "Eurasian_Wigeon") & (df["true"] == "Eurasian_Wigeon")]),
    ]

    summary_rows = []
    for tag, title, sub in groups:
        print(f"[GROUP] {tag}: {len(sub)} chunks")
        if len(sub) > 0:
            print(f"  recordings: {sub['xc_id'].nunique()} unique")
            print(f"  top recordings:")
            for xid, n in sub["xc_id"].value_counts().head(3).items():
                print(f"    {n}x {xid[:60]}")
        plot_group(sub, f"{tag}: {title}", out_dir / f"{tag}.png", max_samples=args.max_samples)
        summary_rows.append({
            "group": tag,
            "title": title,
            "n_chunks": len(sub),
            "n_recordings": int(sub["xc_id"].nunique()) if len(sub) else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[OK] 出力: {out_dir}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
