"""Qiita Part2 用の図を生成する（クリーンテイスト / 日本語ラベル）。
出力先: docs/qiita_figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path

# --- 共通スタイル（Qiita 向けクリーン）---
rcParams["font.family"] = "Noto Sans CJK JP"
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["savefig.dpi"] = 130
rcParams["savefig.bbox"] = "tight"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["grid.linewidth"] = 0.6
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["font.size"] = 12

OUT = Path(__file__).parent / "qiita_figures"
OUT.mkdir(exist_ok=True)

C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GRAY = "#9aa0a6"
C_RED = "#C44E52"
C_GREEN = "#55A868"

# =====================================================================
# ① run 推移ラインチャート（記事の背骨）
# =====================================================================
# 方針: 図中タイトル・注釈は入れない（説明は本文キャプションが担う）。
# 軸ラベル・凡例・データ値のみに絞る（Part1 のスタイルに合わせる）。
from matplotlib.patches import Patch

runs = ["run05", "run06", "run07", "run08", "run09", "run10", "run11"]
f1 = [0.838, 0.810, 0.827, 0.820, 0.810, 0.782, 0.793]
x = np.arange(len(runs))

fig, ax = plt.subplots(figsize=(9, 4.6))
# 95%CI 帯（録音単位bootstrap, run10実測）と 10s/3s 領域は色だけで示す
ax.axhspan(0.679, 0.843, color=C_RED, alpha=0.07, zorder=0)
ax.axvspan(-0.5, 4.5, color=C_BLUE, alpha=0.05)
ax.axvspan(4.5, 6.5, color=C_ORANGE, alpha=0.07)
ax.axhline(0.838, color=C_GRAY, ls=":", lw=1)

ax.plot(x[:5], f1[:5], "-o", color=C_BLUE, lw=2, ms=8, zorder=3, label="10秒チャンク")
ax.plot(x[4:], f1[4:], "-o", color=C_ORANGE, lw=2, ms=8, zorder=3, label="3秒チャンク")
for xi, yi in zip(x, f1):
    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9)

handles = [
    plt.Line2D([], [], color=C_BLUE, marker="o", lw=2, label="10秒チャンク"),
    plt.Line2D([], [], color=C_ORANGE, marker="o", lw=2, label="3秒チャンク"),
    plt.Line2D([], [], color=C_GRAY, ls=":", label="Part1到達点 0.838"),
    Patch(color=C_RED, alpha=0.18, label="1 run分の95%CI"),
]
ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left", ncol=2)
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_ylabel("test f1_macro（8種）")
ax.set_ylim(0.66, 0.86)
fig.savefig(OUT / "fig1_run_trajectory.png")
plt.close(fig)

# =====================================================================
# ② run08 仮説崩壊（chunk半減 vs precision不変）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax = axes[0]
bars = ax.bar(["run05", "run08"], [728, 361], color=[C_BLUE, C_ORANGE], width=0.55)
ax.set_ylabel("train チャンク数（キンクロ）")
for b, v in zip(bars, [728, 361]):
    ax.text(b.get_x()+b.get_width()/2, v+12, str(v), ha="center", fontsize=10)
ax.set_ylim(0, 820)

ax = axes[1]
bars = ax.bar(["run05", "run08"], [0.269, 0.269], color=[C_BLUE, C_ORANGE], width=0.55)
ax.set_ylabel("precision（キンクロ）")
for b, v in zip(bars, [0.269, 0.269]):
    ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 0.36)
fig.tight_layout()
fig.savefig(OUT / "fig2_run08_collapse.png")
plt.close(fig)

# =====================================================================
# ③ 誤判定の録音集中（XC197026 / XC349677、run09 前後）
# =====================================================================
labels = ["XC197026\n(コガモ 幼鳥)", "XC349677\n(ヒドリガモ 掛け合い)"]
before = [13, 5]
after = [4, 0]
xpos = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(7.5, 4.3))
b1 = ax.bar(xpos - w/2, before, w, label="長尺2録音 あり", color=C_RED)
b2 = ax.bar(xpos + w/2, after, w, label="長尺2録音 削除", color=C_GREEN)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                str(int(b.get_height())), ha="center", fontsize=10)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("キンクロと誤判定された件数")
ax.set_ylim(0, 15)
ax.legend(frameon=False, fontsize=10)
fig.savefig(OUT / "fig3_recording_concentration.png")
plt.close(fig)

# =====================================================================
# ④ 種別 F1 before/after（run05 10s vs run10 3s）
# =====================================================================
sp = ["マガモ", "コガモ", "オナガガモ", "ハシビロガモ",
      "ヒドリガモ", "キンクロ\nハジロ", "ホシハジロ", "ホオジロ\nガモ"]
run05 = [0.918, 0.844, 0.949, 0.923, 0.809, 0.389, 0.945, 0.930]
run10 = [0.843, 0.772, 0.722, 0.693, 0.813, 0.738, 0.947, 0.729]
xpos = np.arange(len(sp))
w = 0.38

fig, ax = plt.subplots(figsize=(10, 4.6))
b1 = ax.bar(xpos - w/2, run05, w, label="run05（10秒チャンク）", color=C_BLUE)
b2 = ax.bar(xpos + w/2, run10, w, label="run10（3秒チャンク）", color=C_ORANGE)
ax.bar(5 + w/2, run10[5], w, color=C_GREEN, zorder=3)  # キンクロを色で強調
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(sp, fontsize=10)
ax.set_ylabel("test F1")
ax.set_ylim(0, 1.1)
ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.savefig(OUT / "fig4_per_species.png")
plt.close(fig)

print("生成完了:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.relative_to(Path(__file__).parent.parent))
