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
import sys
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
    """チャンク単位で推論し、softmax max_conf・energy_score・pred_species を返す。"""
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
            logits_np = logits.cpu().numpy()[0]          # (num_classes,)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            top_idx = int(np.argmax(probs))
            # Energy score: logsumexp(logits)。in-dist ほど高い
            a = logits_np.max()
            energy = float(a + np.log(np.sum(np.exp(logits_np - a))))

            results.append({
                "file": wav.name,
                "max_conf": float(probs[top_idx]),
                "energy_score": energy,
                "pred_species": id2label[top_idx],
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
            "energy_score": r["energy_score"],
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
    # OOD 種は species_master.csv の status から引く（taxonomy の ood_species は
    # グループ構造移行で廃止。run13 で target 昇格した4種は status が target になり
    # ここから自動的に除外される）。
    master = pd.read_csv(PROJECT_ROOT / "data" / "species_master.csv")

    def _tier_species(status: str) -> list[dict]:
        sub = master[master["status"] == status]
        return [{"en": en} for en in sub["en_inat"].tolist()]

    tier_map = {
        1: _tier_species("ood_tier1"),
        2: _tier_species("ood_tier2"),
        3: _tier_species("ood_tier3"),
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
                    "energy_score": r["energy_score"],
                    "correct": False,
                })
            print(f"  Tier{tier} / {en}: {len(res)} チャンク")

    return pd.DataFrame(rows)


def plot_score_dist(df: pd.DataFrame, out_dir: Path, score_col: str, filename: str, title: str, ylabel: str) -> None:
    """Tier 別のスコア分布をバイオリンプロットで可視化。"""
    groups = ["in_dist", "tier1", "tier2", "tier3"]
    labels = ["In-dist\n(test)", "OOD Tier1\n(duck-adj)", "OOD Tier2\n(waterbird)", "OOD Tier3\n(control)"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    data = [df[df["tier"] == g][score_col].values for g in groups]
    valid = [(d, l, c) for d, l, c in zip(data, labels, colors) if len(d) > 0]
    data_v, labels_v, colors_v = zip(*valid) if valid else ([], [], [])

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(list(data_v), positions=range(len(data_v)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors_v):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels_v)))
    ax.set_xticklabels(list(labels_v), fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] {filename}")


def plot_roc(
    df_indist: pd.DataFrame,
    df_ood: pd.DataFrame,
    out_dir: Path,
    score_col: str,
    filename: str,
    label: str,
    color: str,
) -> tuple[float, float]:
    """TPR vs FPR の ROC 曲線を描き、(推奨閾値, AUROC) を返す。"""
    scores_in = df_indist[score_col].values
    correct_in = df_indist["correct"].values.astype(bool)
    scores_ood = df_ood[score_col].values if len(df_ood) > 0 else np.array([])

    all_scores = np.concatenate([scores_in, scores_ood]) if len(scores_ood) > 0 else scores_in
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 300)

    tprs, fprs = [], []
    for theta in thresholds:
        tpr = float(np.mean(scores_in[correct_in] >= theta)) if correct_in.sum() > 0 else 0.0
        fpr = float(np.mean(scores_ood >= theta)) if len(scores_ood) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tprs, fprs = np.array(tprs), np.array(fprs)
    auroc = float(np.trapezoid(tprs[::-1], fprs[::-1]))

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
    ax.plot(fprs, tprs, color=color, linewidth=2, label=f"{label} (AUROC={auroc:.3f})")
    ax.scatter([rec_fpr], [rec_tpr], color="red", zorder=5,
               label=f"rec theta={rec_theta:.3f} (TPR={rec_tpr:.3f}, FPR={rec_fpr:.3f})")
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.8, label="FPR=0.05")
    ax.set_xlabel("FPR (OOD false acceptance rate)")
    ax.set_ylabel("TPR (in-dist correct retention rate)")
    ax.set_title(f"ROC [{label}]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] {filename}  AUROC={auroc:.3f}  rec_theta={rec_theta:.3f}")

    return rec_theta, auroc


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
        print(f"  OOD: {len(df_ood)} チャンク / mean_conf={df_ood['max_conf'].mean():.3f} / mean_energy={df_ood['energy_score'].mean():.3f}")

    df_all = pd.concat([df_indist, df_ood], ignore_index=True)
    df_all.to_csv(out_dir / "ood_results.csv", index=False)

    print("\n[PLOT] 可視化...")
    # softmax 分布
    plot_score_dist(
        df_all, out_dir,
        score_col="max_conf", filename="dist_softmax.png",
        title="Softmax Confidence: In-dist vs OOD",
        ylabel="Max Softmax Confidence",
    )
    # energy スコア分布
    plot_score_dist(
        df_all, out_dir,
        score_col="energy_score", filename="dist_energy.png",
        title="Energy Score: In-dist vs OOD",
        ylabel="Energy Score (logsumexp of logits)",
    )

    rec_softmax, rec_energy = None, None
    auroc_softmax, auroc_energy = None, None
    if not df_ood.empty:
        rec_softmax, auroc_softmax = plot_roc(
            df_indist, df_ood, out_dir,
            score_col="max_conf", filename="roc_softmax.png",
            label="Max Softmax", color="#4c72b0",
        )
        rec_energy, auroc_energy = plot_roc(
            df_indist, df_ood, out_dir,
            score_col="energy_score", filename="roc_energy.png",
            label="Energy Score", color="#dd8452",
        )
        report_top_misclassified(df_ood, out_dir)

    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print(f"  in-dist: {len(df_indist)} chunks / acc={indist_acc:.3f}")
    print(f"    softmax mean_conf={df_indist['max_conf'].mean():.3f}  energy_mean={df_indist['energy_score'].mean():.3f}")
    if not df_ood.empty:
        for tier in args.tiers:
            sub = df_ood[df_ood["tier"] == f"tier{tier}"]
            if not sub.empty:
                print(f"  OOD Tier{tier}: {len(sub)} chunks / softmax={sub['max_conf'].mean():.3f} / energy={sub['energy_score'].mean():.3f}")
    if auroc_softmax is not None:
        print(f"\n  AUROC  softmax={auroc_softmax:.3f}  energy={auroc_energy:.3f}")
        better = "Energy" if auroc_energy > auroc_softmax else "Softmax"
        print(f"  → {better} の方が OOD 分離性能が高い")
    if rec_energy is not None:
        print(f"\n  推奨閾値 (energy, FPR<=5%): {rec_energy:.3f}")
        print(f"  → species_taxonomy.yaml の pipeline.confidence_threshold に設定してください")
    print(f"\n  出力先: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
