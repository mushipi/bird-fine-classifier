"""GitHub Pages 記事（生態学者向け）の図を生成する。
レポート docs/perch_kd_report.md の数値から要約図を起こす（重い実データ不要・再現可能）。
出力: site_build/figs/*.png
スタイル: クリーン / 日本語ラベル / Noto Sans CJK JP
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

rcParams["font.family"] = "Noto Sans CJK JP"
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 140
rcParams["savefig.dpi"] = 140
rcParams["savefig.bbox"] = "tight"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25
rcParams["grid.linewidth"] = 0.6
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["font.size"] = 12.5

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

C_BASE = "#9aa0a6"     # 蒸留なし（灰）
C_KD = "#3a7bd5"       # 蒸留あり（青・強調）
C_TEACHER = "#C44E52"  # 教師（赤）
C_GREEN = "#55a868"
C_ORANGE = "#DD8452"
TEACHER = 0.8304


def fig1_headline():
    """3人の生徒で base vs KD を録音単位F1 + 95%CI で並べる（主役）。"""
    groups = ["素のAI\n(小型・伸びしろ大)", "高性能AI\n(単発)", "高性能AI\n(3つ平均)"]
    base = [0.4498, 0.8259, 0.8133]
    base_lo = [0.383, 0.755, 0.746]
    base_hi = [0.507, 0.879, 0.867]
    kd = [0.5977, 0.8510, 0.8714]
    kd_lo = [0.521, 0.786, 0.812]
    kd_hi = [0.662, 0.899, 0.918]
    sig = ["有意", "（埋もれ）", "有意"]

    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    be = [[b - lo for b, lo in zip(base, base_lo)], [hi - b for b, hi in zip(base, base_hi)]]
    ke = [[k - lo for k, lo in zip(kd, kd_lo)], [hi - k for k, hi in zip(kd, kd_hi)]]

    ax.bar(x - w / 2, base, w, yerr=be, capsize=4, color=C_BASE, label="蒸留なし",
           error_kw=dict(ecolor="#555", lw=1.2))
    ax.bar(x + w / 2, kd, w, yerr=ke, capsize=4, color=C_KD, label="蒸留あり (Perchが先生)",
           error_kw=dict(ecolor="#1f3d6b", lw=1.2))

    ax.axhline(TEACHER, ls="--", lw=1.3, color=C_TEACHER, alpha=0.9)
    ax.text(2.45, TEACHER + 0.008, "先生(Perch) 0.83", color=C_TEACHER, fontsize=10.5,
            ha="right", va="bottom")

    for xi, (b, k, s) in enumerate(zip(base, kd, sig)):
        d = k - b
        ax.annotate(f"+{d:.3f}\n{s}", xy=(xi, max(kd_hi[xi], TEACHER) + 0.03),
                    ha="center", va="bottom", fontsize=10.5,
                    color=C_KD if s == "有意" else "#888",
                    fontweight="bold" if s == "有意" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("識別の正確さ（録音1本ごと・10種平均）")
    ax.set_ylim(0, 1.04)
    ax.set_title("先生から学ばせると、生徒AIの正確さが上がる", fontsize=14, pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(0.5, -0.02, "棒の上下のヒゲ＝95%信頼区間。ヒゲが重ならない差が「偶然でない」と言える。",
             ha="center", fontsize=9.5, color="#666")
    fig.savefig(OUT / "fig1_headline.png")
    plt.close(fig)


def fig2_foundation():
    """なぜ Perch を先生にしたか：土台AIの比較。"""
    names = ["Perch\n(鳥の声で学習)", "BirdAVES\n(別の土台AI)"]
    vals = [0.8308, 0.6459]
    colors = [C_GREEN, C_ORANGE]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(names, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("識別の正確さ（録音1本ごと）")
    ax.set_ylim(0, 1.0)
    ax.set_title("先生役には「鳥の声で鍛えた土台AI(Perch)」が圧勝", fontsize=12.5, pad=10)
    ax.annotate("", xy=(0, 0.79), xytext=(1, 0.79),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
    ax.text(0.5, 0.81, "差 0.18", ha="center", color="#444", fontsize=11)
    fig.savefig(OUT / "fig2_foundation.png")
    plt.close(fig)


def fig3_seeds():
    """3回学習し直しても KD が常に勝つ（再現性）。"""
    seeds = ["1回目", "2回目", "3回目"]
    base = [0.826, 0.825, 0.801]
    kd = [0.851, 0.858, 0.825]
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for i in range(len(seeds)):
        ax.plot([x[i] - 0.12, x[i] + 0.12], [base[i], kd[i]], color="#bbb", lw=1.4, zorder=1)
    ax.scatter(x - 0.12, base, s=80, color=C_BASE, zorder=2, label="蒸留なし")
    ax.scatter(x + 0.12, kd, s=80, color=C_KD, zorder=2, label="蒸留あり")
    for i in range(len(seeds)):
        ax.annotate(f"+{kd[i]-base[i]:.3f}", xy=(x[i], kd[i] + 0.004), ha="center",
                    fontsize=10, color=C_KD)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel("識別の正確さ（録音1本ごと）")
    ax.set_ylim(0.78, 0.88)
    ax.set_title("学習をやり直しても、蒸留は毎回プラス（運任せでない）", fontsize=12.5, pad=10)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.savefig(OUT / "fig3_seeds.png")
    plt.close(fig)


def fig4_concept():
    """知識蒸留の概念図：先生の「自信の配分」を生徒が真似る。"""
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5)

    box(0.3, 1.3, 2.3, 1.4, "鳴き声\n（3秒の録音）", "#eef3fb", C_KD)
    box(3.4, 2.3, 2.6, 1.2, "先生 Perch\n（鳥の声の達人）", "#fdeeee", C_TEACHER)
    box(3.4, 0.3, 2.6, 1.2, "生徒 AI\n（これから育てる）", "#eef7f0", C_GREEN)
    box(7.0, 1.3, 2.6, 1.4, "「マガモ 7割\nコガモ 2割…」\nという“迷い方”を共有", "#f4f4f4", "#999")

    arr = lambda x1, y1, x2, y2, c: ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=c, lw=1.8))
    arr(2.6, 2.3, 3.4, 2.7, "#888")
    arr(2.6, 1.7, 3.4, 1.0, "#888")
    arr(6.0, 2.9, 7.0, 2.3, C_TEACHER)
    arr(7.0, 1.6, 6.0, 1.0, C_GREEN)
    ax.text(6.55, 1.05, "真似る", color=C_GREEN, fontsize=10.5, rotation=-18)

    ax.set_title("知識蒸留：先生の“答え方”ごと生徒に写す", fontsize=13.5)
    fig.savefig(OUT / "fig4_concept.png")
    plt.close(fig)


def fig5_flow():
    """全体フロー：録音 → 教師ソフトラベル → 蒸留学習 → soup → 評価 → 昇格。"""
    fig, ax = plt.subplots(figsize=(7.4, 8.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    steps = [
        ("① データ", "録音231本を3秒チャンクに分割\n（カモ10種・録音単位で評価）", "#eef3fb", C_KD),
        ("② 教師ソフトラベル", "先生Perchの複数プローブ平均を、\n録音単位5-fold OOFで作成（過信回避）", "#fdeeee", C_TEACHER),
        ("③ 蒸留学習", "生徒を  L = CE + λT²·KL  で学習\n（λ≈1, T≈2 が最適域）", "#eef7f0", C_GREEN),
        ("④ model soup", "seed 42/1/2 で3回学習し、\n重みを平均（θ_soup = mean θ_s）", "#fff6e9", C_ORANGE),
        ("⑤ 評価", "録音単位 macro-F1 ＋\nブートストラップ95%信頼区間で判定", "#eef3fb", C_KD),
        ("⑥ 昇格", "信頼区間が0を除外 → 本物の改善\nとして運用モデルへ採用", "#eef7f0", C_GREEN),
    ]
    n = len(steps)
    bh, gap = 1.35, 0.42
    top = 11.4
    cx, bw = 5.0, 7.6
    for i, (title, body, fc, ec) in enumerate(steps):
        y = top - i * (bh + gap) - bh
        ax.add_patch(FancyBboxPatch((cx - bw / 2, y), bw, bh,
                     boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=1.7))
        ax.text(cx - bw / 2 + 0.28, y + bh - 0.34, title, ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=ec)
        ax.text(cx, y + 0.46, body, ha="center", va="center", fontsize=10.8, color="#333")
        if i < n - 1:
            ay = y - 0.02
            ax.add_patch(FancyArrowPatch((cx, ay), (cx, ay - gap + 0.04),
                         arrowstyle="-|>", mutation_scale=18, color="#9aa0a6", lw=2.0))

    ax.set_title("全体フロー：先生の知識を蒸留して、確かな改善だけ採用する", fontsize=13.5, pad=14)
    fig.savefig(OUT / "fig5_flow.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_headline()
    fig2_foundation()
    fig3_seeds()
    fig4_concept()
    fig5_flow()
    print("done:", sorted(p.name for p in OUT.glob("*.png")))
