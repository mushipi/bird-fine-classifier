"""Qiita 記事用の図を生成する（クリーンテイスト / 日本語ラベル）。Part2 / Part3…。
出力先: docs/qiita/part<N>/figures/（記事ごと）
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

BASE = Path(__file__).parent
P2 = BASE / "part2" / "figures"; P2.mkdir(parents=True, exist_ok=True)
P3 = BASE / "part3" / "figures"; P3.mkdir(parents=True, exist_ok=True)
P4 = BASE / "part4" / "figures"; P4.mkdir(parents=True, exist_ok=True)
P5 = BASE / "part5" / "figures"; P5.mkdir(parents=True, exist_ok=True)

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
fig.savefig(P2 / "fig1_run_trajectory.png")
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
fig.savefig(P2 / "fig2_run08_collapse.png")
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
fig.savefig(P2 / "fig3_recording_concentration.png")
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
fig.savefig(P2 / "fig4_per_species.png")
plt.close(fig)

# =====================================================================
# Part3 ①｜効果量は「生徒の伸びしろ」で決まる（弱CNN は有意 / 強AST は非有意）
# =====================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.6))
groups = ["素のCNN\n(弱い生徒)", "AST\n(強い生徒)"]
base_v = [0.4498, 0.8259]
kd_v = [0.5977, 0.8510]
x = np.arange(len(groups)); w = 0.34
ax.bar(x - w / 2, base_v, w, color=C_GRAY, label="蒸留なし base")
ax.bar(x + w / 2, kd_v, w, color=C_BLUE, label="蒸留 KD")
for i in range(len(groups)):
    ax.annotate(f"{base_v[i]:.3f}", (x[i] - w / 2, base_v[i]), ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{kd_v[i]:.3f}", (x[i] + w / 2, kd_v[i]), ha="center", va="bottom", fontsize=10)
delta = ["+0.148\n★有意 [+0.084,+0.211]", "+0.025\n非有意 [-0.022,+0.079]"]
dcol = [C_GREEN, C_RED]
for i in range(len(groups)):
    top = max(base_v[i], kd_v[i])
    ax.annotate(delta[i], (x[i], top + 0.07), ha="center", va="bottom",
                fontsize=10.5, color=dcol[i], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("録音単位 macro-F1"); ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", framealpha=0.9)
fig.savefig(P3 / "fig_p3_1_effect_by_student.png")
plt.close(fig)

# =====================================================================
# Part3 ②｜リーク: 運用モデルのテスト289録音の45%が「学習済み」だった
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 2.6))
clean, leak = 159, 130
ax.barh([0], [clean], color=C_GREEN, label=f"未学習＝honest評価に使える {clean}録音")
ax.barh([0], [leak], left=[clean], color=C_RED, label=f"旧trainからリーク {leak}録音 (45.0%)")
ax.annotate(f"{clean}", (clean / 2, 0), ha="center", va="center", color="white", fontweight="bold")
ax.annotate(f"{leak}\n(45%)", (clean + leak / 2, 0), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax.set_xlim(0, clean + leak); ax.set_yticks([])
ax.set_xlabel("Cv2-test の録音数（計 289）")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.95), ncol=1, framealpha=0.9)
ax.grid(False)
fig.savefig(P3 / "fig_p3_2_leak.png")
plt.close(fig)

# =====================================================================
# Part3 ③｜評価をhonestにすると順位が反転する（運用モデルが最下位へ）
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
models = ["Cprod\n(運用中)", "Cv2BASE\n(蒸留なし)", "Cv2KD\n(蒸留)"]
full = [0.910, 0.908, 0.894]
honest = [0.787, 0.897, 0.822]
cols = [C_RED, C_GREEN, C_BLUE]
for ax, vals, title in [(axes[0], full, "全体評価（リーク込み・289録音）"),
                        (axes[1], honest, "honest 評価（未学習159録音）")]:
    ax.bar(models, vals, color=cols)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=10.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("録音単位 f1")
fig.savefig(P3 / "fig_p3_3_rank_flip.png")
plt.close(fig)

# =====================================================================
# Part4 ①｜CPU 推論レイテンシ（推論は軽い・前処理が律速）
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.2))
items = ["AST 推論", "前処理\n(特徴抽出)"]
ms = [109, 611]
cols = [C_BLUE, C_ORANGE]
b = ax.bar(items, ms, color=cols, width=0.55)
for bi, v in zip(b, ms):
    ax.annotate(f"{v} ms", (bi.get_x()+bi.get_width()/2, v), ha="center", va="bottom", fontsize=11)
ax.set_ylabel("CPU 処理時間 (ms / 3秒チャンク)")
ax.set_ylim(0, 700)
ax.set_title("", fontsize=1)
fig.savefig(P4 / "fig_p4_1_cpu_latency.png")
plt.close(fig)

# =====================================================================
# Part4 ②｜カルガモ壁: Perch2.0本体でも カルガモは自種に当たらない（非対称崩壊）
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.3))
labels = ["カルガモ録音", "マガモ録音"]
self_hit = [0, 18]   # 全クラス argmax = 自種 (/30)
ax.bar(labels, self_hit, color=[C_RED, C_GREEN], width=0.5)
for i, v in enumerate(self_hit):
    ax.annotate(f"{v}/30", (i, v), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Perch本体が「自種」と判定した数 (/30)")
ax.set_ylim(0, 30)
ax.annotate("カルガモ専用クラスを持つ\nPerch2.0でも 0/30", (0, 1.0), ha="center", va="bottom",
            fontsize=10, color=C_RED)
fig.savefig(P4 / "fig_p4_2_kalgamo_wall.png")
plt.close(fig)

# =====================================================================
# Part5 ①｜gull 録音単位 混同行列（大型白頭3種は互いに崩壊しない）
# =====================================================================
g_labels = ["ユリ", "ウミネコ", "カモメ", "ズグロ", "オオセグロ", "セグロ"]
cm = np.array([
    [16, 0, 0, 0, 0, 0],
    [0, 3, 1, 0, 1, 0],
    [2, 0, 14, 0, 0, 0],
    [1, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 4, 0],
    [2, 0, 0, 0, 0, 8],
])
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6)); ax.set_xticklabels(g_labels, rotation=30, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(g_labels)
ax.set_xlabel("予測"); ax.set_ylabel("真")
for i in range(6):
    for j in range(6):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 8 else "black", fontsize=11)
ax.grid(False)
fig.savefig(P5 / "fig_p5_1_gull_confusion.png")
plt.close(fig)

print("生成完了:")
for d in (P2, P3, P4, P5):
    for p in sorted(d.glob("*.png")):
        print(" ", p.relative_to(BASE))
