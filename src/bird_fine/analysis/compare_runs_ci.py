"""複数 run の test 予測を録音単位クラスタブートストラップで比較する。

なぜ必要か:
    test は録音数が少なく（既存8種で 69 録音）、同一録音内の chunk は強く相関する。
    chunk 単位の macro-F1 / 再標本化は実効サンプル数を過大評価し、CI を過小評価
    （→ docs/bootstrap_ci.py）＋ 録音内の一部誤分類をペナルティして性能を過小評価する。
    run 間の精度差を主張する前に、本スクリプトで録音単位 macro-F1 の 95%CI と
    ペア差の CI を確認し、**CI が 0 を含む差は「差なし（ノイズ）」**として扱う。

    2026-06-05 の検証では run11/13/15 の既存8種差はすべて有意でなかった
    （CI 幅 ±0.1 / 全ペア差 CI が 0 を含む）。test 69 録音では run 間の微差を
    測定する解像度が無い。微差比較は参考値に留め、大きな構造的差のみ意思決定に使う。

各 run は (name, eval_dir, test の git ref, label_map の git ref) で指定。
現行 split を使う run は ref="cur"。新しい run を足すときは RUNS に追記する。

使い方:
  uv run python -m bird_fine.analysis.compare_runs_ci
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# run11 と公平比較できる「既存8種」（種拡張前から共通の target）
EXIST8 = [
    "Common_Goldeneye", "Common_Pochard", "Eurasian_Teal", "Eurasian_Wigeon",
    "Mallard", "Northern_Pintail", "Northern_Shoveler", "Tufted_Duck",
]

# (name, eval_dir, test の git ref, label_map の git ref)。ref="cur" は現行 split。
RUNS = [
    ("run10", "outputs/eval_20260531_190005", "08380cc", "08380cc"),
    ("run11", "outputs/eval_20260531_231427", "038b443", "038b443"),
    ("run13", "outputs/eval_20260604_221425", "d23574b", "d23574b"),
    ("run14", "outputs/eval_20260604_231559", "d23574b", "d23574b"),
    ("run15", "outputs/eval_20260605_054712", "cur", "cur"),
]
B = 3000
SEED = 42


def _gitread(ref: str, path: str) -> pd.DataFrame:
    if ref == "cur":
        return pd.read_csv(PROJECT_ROOT / path)
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, cwd=PROJECT_ROOT)
    return pd.read_csv(io.StringIO(out.stdout))


def _load(eval_dir: str, test_ref: str, lm_ref: str) -> pd.DataFrame:
    pred = pd.read_csv(PROJECT_ROOT / eval_dir / "predictions.csv")
    test = _gitread(test_ref, "data/splits/test.csv").reset_index(drop=True)
    lm = _gitread(lm_ref, "data/splits/label_map.csv")
    id2sp = {int(r.label_id): r.species for _, r in lm.iterrows()}
    test["true"] = pred["y_true"].map(id2sp)
    test["pred"] = pred["y_pred"].map(id2sp)
    return test[test["true"].isin(EXIST8)]


def main() -> None:
    runs = {name: _load(ev, tr, lr) for name, ev, tr, lr in RUNS}
    common = sorted(set.intersection(*[set(d["xc_id"]) for d in runs.values()]))
    print(f"共通の既存8種録音数: {len(common)}")

    # 録音単位（chunk 多数決）テーブル
    def rec_table(d: pd.DataFrame) -> pd.DataFrame:
        d = d[d["xc_id"].isin(common)]
        return d.groupby("xc_id").agg(
            true=("true", "first"),
            pred=("pred", lambda s: s.value_counts().index[0]),
        )

    tabs = {k: rec_table(d) for k, d in runs.items()}

    def f1_of(tab: pd.DataFrame, sample: list[str]) -> float:
        return f1_score(tab.loc[sample, "true"], tab.loc[sample, "pred"],
                        labels=EXIST8, average="macro", zero_division=0)

    rng = np.random.default_rng(SEED)
    print(f"\n=== 録音単位 macro-F1（既存8種, n={len(common)}）点推定 + 95%CI ===")
    for k, tab in tabs.items():
        pt = f1_of(tab, common)
        bs = np.array([f1_of(tab, list(rng.choice(common, len(common), replace=True))) for _ in range(B)])
        print(f"  {k}: {pt:.3f}  95%CI[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")

    print(f"\n=== run 間差のペア bootstrap（同一録音 resample）95%CI ===")
    rng2 = np.random.default_rng(SEED)
    names = list(tabs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[j], names[i]
            diffs = []
            for _ in range(B):
                s = list(rng2.choice(common, len(common), replace=True))
                diffs.append(f1_of(tabs[a], s) - f1_of(tabs[b], s))
            lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            sig = "有意" if (lo > 0 or hi < 0) else "★差なし(CIが0を含む)"
            print(f"  {a}-{b}: 点差={f1_of(tabs[a],common)-f1_of(tabs[b],common):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
