"""学習済み AST モデルを OOD テストセットで評価する。

in-distribution (test.csv) と OOD (data/ood_processed/) の confidence 分布を比較し、
誤吸引先・ROC 曲線・推奨しきい値を出力する。

使い方:
    uv run python -m bird_fine.analysis.ood_eval
    uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-v10
    uv run python -m bird_fine.analysis.ood_eval --tiers 1 2
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import ASTFeatureExtractor, ASTForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TAXONOMY_PATH = PROJECT_ROOT / "species_taxonomy.yaml"
OOD_PROCESSED_DIR = PROJECT_ROOT / "data" / "ood_processed"


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_dir: Path, device: torch.device):
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    label_df = pd.read_csv(model_dir / "label_map.csv")
    id2label = {int(row["label_id"]): row["species"] for _, row in label_df.iterrows()}
    return model, id2label


def infer_chunks(
    wav_files: list[Path],
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    id2label: dict[int, str],
    device: torch.device,
) -> list[dict]:
    """チャンク単位で推論し、各チャンクのmax_conf・pred_species・全probs を返す。"""
    import soundfile as sf

    results = []
    with torch.no_grad():
        for wav in wav_files:
            try:
                audio, sr = sf.read(str(wav), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
            except Exception:
                continue

            inputs = feature_extractor(
                audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
            )
            logits = model(input_values=inputs["input_values"].to(device)).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            top_idx = int(np.argmax(probs))
            results.append({
                "file": wav.name,
                "max_conf": float(probs[top_idx]),
                "pred_species": id2label[top_idx],
                "probs": probs,
            })
    return results


def collect_indist(
    splits_dir: Path,
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    id2label: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    """in-distribution test セットの推論結果を DataFrame で返す。"""
    test_csv = splits_dir / "test.csv"
    df = pd.read_csv(test_csv)

    rows = []
    for _, row in df.iterrows():
        wav = PROJECT_ROOT / Path(row["file_path"].replace("\\", "/"))
        res = infer_chunks([wav], model, feature_extractor, id2label, device)
        if not res:
            continue
        r = res[0]
        rows.append({
            "source": "in_dist",
            "tier": "in_dist",
            "true_species": row["species"],
            "pred_species": r["pred_species"],
            "max_conf": r["max_conf"],
            "correct": row["species"] == r["pred_species"],
        })

    return pd.DataFrame(rows)


def collect_ood(
    tiers: list[int],
    taxonomy: dict,
    model: ASTForAudioClassification,
    feature_extractor: ASTFeatureExtractor,
    id2label: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    """OOD チャンクの推論結果を DataFrame で返す。"""
    tier_map = {
        1: taxonomy["ood_species"]["tier1"],
        2: taxonomy["ood_species"]["tier2"],
        3: taxonomy["ood_species"]["tier3"],
    }
    rows = []
    for tier in tiers:
        for sp in tier_map[tier]:
            en = sp["en"]
            species_dir = OOD_PROCESSED_DIR / f"tier{tier}" / en.replace(" ", "_")
            if not species_dir.exists():
                print(f"  [SKIP] Tier{tier}/{en}: チャンクなし")
                continue
            wavs = sorted(species_dir.glob("*.wav"))
            if not wavs:
                continue
            res = infer_chunks(wavs, model, feature_extractor, id2label, device)
            for r in res:
                rows.append({
                    "source": "ood",
                    "tier": f"tier{tier}",
                    "true_species": en,
                    "pred_species": r["pred_species"],
                    "max_conf": r["max_conf"],
                    "correct": False,
                })
            print(f"  Tier{tier} / {en}: {len(res)} チャンク")

    return pd.DataFrame(rows)


def plot_confidence_dist(df: pd.DataFrame, out_dir: Path) -> None:
    """Tier 別の max_conf 分布をバイオリンプロットで可視化。"""
    groups = ["in_dist", "tier1", "tier2", "tier3"]
    labels = ["In-dist\n(test)", "OOD Tier1\n(他カモ科)", "OOD Tier2\n(水辺非カモ)", "OOD Tier3\n(コントロール)"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    data = [df[df["tier"] == g]["max_conf"].values for g in groups]
    data = [d for d in data if len(d) > 0]
    active_labels = [l for l, d in zip(labels, data) if len(d) > 0]
    active_colors = [c for c, d in zip(colors, data) if len(d) > 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], active_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(active_labels)))
    ax.set_xticklabels(active_labels, fontsize=10)
    ax.set_ylabel("Max Softmax Confidence")
    ax.set_title("Confidence Distribution: In-distribution vs OOD")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="θ=0.7 (暫定)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "confidence_dist.png", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] confidence_dist.png")


def plot_roc(df_indist: pd.DataFrame, df_ood: pd.DataFrame, out_dir: Path) -> float:
    """TPR (in-dist 正解保持率) vs FPR (OOD 誤受理率) の ROC 曲線を描き、推奨 θ を返す。"""
    thresholds = np.linspace(0.0, 1.0, 200)
    tprs, fprs = [], []

    indist_conf = df_indist["max_conf"].values
    indist_correct = df_indist["correct"].values.astype(bool)
    ood_conf = df_ood["max_conf"].values if len(df_ood) > 0 else np.array([])

    for theta in thresholds:
        # TPR: 正解 in-dist チャンクのうち confidence >= θ の割合
        tpr = float(np.mean(indist_conf[indist_correct] >= theta)) if indist_correct.sum() > 0 else 0.0
        # FPR: OOD チャンクのうち confidence >= θ の割合（誤受理）
        fpr = float(np.mean(ood_conf >= theta)) if len(ood_conf) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tprs, fprs = np.array(tprs), np.array(fprs)

    # 推奨 θ: FPR <= 0.05 を満たしながら TPR を最大化
    mask = fprs <= 0.05
    if mask.any():
        best_idx = int(np.argmax(tprs[mask]))
        rec_theta = float(thresholds[mask][best_idx])
        rec_tpr = float(tprs[mask][best_idx])
        rec_fpr = float(fprs[mask][best_idx])
    else:
        best_idx = int(np.argmin(fprs))
        rec_theta = float(thresholds[best_idx])
        rec_tpr = float(tprs[best_idx])
        rec_fpr = float(fprs[best_idx])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, color="#4c72b0", linewidth=2, label="ROC")
    ax.scatter([rec_fpr], [rec_tpr], color="red", zorder=5,
               label=f"推奨 θ={rec_theta:.2f} (TPR={rec_tpr:.3f}, FPR={rec_fpr:.3f})")
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.8, label="FPR=0.05 基準")
    ax.set_xlabel("FPR (OOD 誤受理率)")
    ax.set_ylabel("TPR (in-dist 正解保持率)")
    ax.set_title("ROC: Confidence Threshold vs OOD Rejection")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] roc_curve.png")

    return rec_theta


def report_top_misclassified(df_ood: pd.DataFrame, out_dir: Path) -> None:
    """OOD 種ごとの誤吸引先 Top3 を表形式で出力・保存。"""
    if df_ood.empty:
        return

    rows = []
    for (tier, sp), grp in df_ood.groupby(["tier", "true_species"]):
        top3 = grp["pred_species"].value_counts().head(3)
        absorbed = {f"top{i+1}": f"{k} ({v})" for i, (k, v) in enumerate(top3.items())}
        rows.append({
            "tier": tier,
            "ood_species": sp,
            "n_chunks": len(grp),
            "mean_conf": round(grp["max_conf"].mean(), 3),
            **absorbed,
        })

    result_df = pd.DataFrame(rows).sort_values(["tier", "mean_conf"], ascending=[True, False])
    result_df.to_csv(out_dir / "ood_misclassified.csv", index=False)

    print("\n[OOD 誤吸引先 Top3]")
    print(result_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None, help="モデルディレクトリ。省略時は config.training.output_dir")
    parser.add_argument("--tiers", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    args = parser.parse_args()

    config = load_config()
    taxonomy = load_taxonomy()
    model_cfg = config["model"]
    train_cfg = config["training"]
    pp = config["preprocessing"]

    model_dir = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / train_cfg["output_dir"]
    if not model_dir.exists():
        print(f"[ERROR] {model_dir} が見つからない。")
        return

    splits_dir = PROJECT_ROOT / pp["splits_dir"]
    max_length = int(model_cfg.get("feature_extractor_max_length", 1024))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / f"ood_eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] {device}")
    print(f"[MODEL] {model_dir}")

    print("[LOAD] モデル・Feature Extractor...")
    model, id2label = load_model(model_dir, device)
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        model_cfg["pretrained"], max_length=max_length
    )

    print("\n[IN-DIST] test セット推論中...")
    df_indist = collect_indist(splits_dir, model, feature_extractor, id2label, device)
    indist_acc = df_indist["correct"].mean()
    indist_mean_conf = df_indist["max_conf"].mean()
    print(f"  in-dist: {len(df_indist)} チャンク / acc={indist_acc:.3f} / mean_conf={indist_mean_conf:.3f}")

    print("\n[OOD] OOD チャンク推論中...")
    df_ood = collect_ood(args.tiers, taxonomy, model, feature_extractor, id2label, device)
    if df_ood.empty:
        print("  [WARN] OOD チャンクが見つからない。先に download_ood.py を実行して。")
    else:
        print(f"  OOD: {len(df_ood)} チャンク / mean_conf={df_ood['max_conf'].mean():.3f}")

    # 全データ統合
    df_all = pd.concat([df_indist, df_ood], ignore_index=True)
    df_all.drop(columns=["probs"] if "probs" in df_all.columns else [], errors="ignore")
    df_all.to_csv(out_dir / "ood_results.csv", index=False)

    print("\n[PLOT] 可視化...")
    plot_confidence_dist(df_all, out_dir)

    rec_theta = None
    if not df_ood.empty:
        rec_theta = plot_roc(df_indist, df_ood, out_dir)
        report_top_misclassified(df_ood, out_dir)

    # サマリ
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print(f"  in-dist test: {len(df_indist)} chunks / acc={indist_acc:.3f} / mean_conf={indist_mean_conf:.3f}")
    if not df_ood.empty:
        for tier in args.tiers:
            sub = df_ood[df_ood["tier"] == f"tier{tier}"]
            if not sub.empty:
                print(f"  OOD Tier{tier}: {len(sub)} chunks / mean_conf={sub['max_conf'].mean():.3f}")
    if rec_theta is not None:
        print(f"\n  推奨 confidence_threshold: {rec_theta:.2f}")
        print(f"  → species_taxonomy.yaml の pipeline.confidence_threshold に設定してください")
    print(f"\n  出力先: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
