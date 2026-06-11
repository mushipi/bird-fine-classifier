"""記事中の信頼区間を再現するスクリプト。
録音単位（クラスタ）ブートストラップで macro-F1 の 95%CI を算出する。
チャンクは同一録音内で相関するため、チャンク単位の再標本化は CI を過小評価する。
正しくは録音(xc_id)を塊ごと再標本化する。

使い方:
  uv run python docs/bootstrap_ci.py
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

EVAL_DIR = "outputs/eval_20260531_190005"   # run10 の test 予測
SPLIT = "data/splits/test.csv"
B = 3000
SEED = 42

preds = pd.read_csv(f"{EVAL_DIR}/predictions.csv")
test = pd.read_csv(SPLIT).reset_index(drop=True)
assert len(preds) == len(test), (len(preds), len(test))

df = test.copy()
df["y_true"] = preds["y_true"].values
df["y_pred"] = preds["y_pred"].values

point = f1_score(df.y_true, df.y_pred, average="macro")
print(f"point macro-F1 = {point:.4f} / 録音数={df.xc_id.nunique()} / チャンク数={len(df)}")

rng = np.random.default_rng(SEED)
yt, yp = df.y_true.values, df.y_pred.values


def boot_chunk():
    out = []
    idx = np.arange(len(df))
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        out.append(f1_score(yt[s], yp[s], average="macro"))
    return np.array(out)


def boot_recording():
    groups = {k: g.index.values for k, g in df.groupby("xc_id")}
    recs = list(groups)
    out = []
    for _ in range(B):
        chosen = rng.choice(recs, size=len(recs), replace=True)
        rows = np.concatenate([groups[r] for r in chosen])
        out.append(f1_score(yt[rows], yp[rows], average="macro"))
    return np.array(out)


def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


c = boot_chunk()
r = boot_recording()
cl, ch = ci(c)
rl, rh = ci(r)
print(f"チャンク単位 95%CI: [{cl:.3f}, {ch:.3f}]  幅={ch-cl:.3f}  (相関無視→過小評価)")
print(f"録音単位   95%CI: [{rl:.3f}, {rh:.3f}]  幅={rh-rl:.3f}  (クラスタ再標本化→正しい)")
