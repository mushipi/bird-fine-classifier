"""Perch2.0 ネイティブ eBird 分類ヘッドで カルガモ vs マガモ の分離力を直接測る。

問い: カルガモ壁は音響的に本質的か（誰がやっても無理）, それとも我々のプローブ/データの限界か。
Perch2.0本体(eBird ~14796クラス)を実物の カルガモ/マガモ 録音に当て、
spbduc(カルガモ) と mallar3(マガモ) の logit 大小・全体argmax を見る。
"""
import glob
import os
import sys

import librosa
import numpy as np
from perch_hoplite.zoo import model_configs as mc

CSV = "references/weights/kagglehub_cache/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2/assets/perch_v2_ebird_classes.csv"
classes = [l.strip() for l in open(CSV)][1:]
I_MAL, I_SPB = classes.index("mallar3"), classes.index("spbduc")
print(f"classes={len(classes)} mallar3(マガモ)={I_MAL} spbduc(カルガモ)={I_SPB}", flush=True)

m = mc.load_model_by_name("perch_v2")


def logits_of(wav):
    a, _ = librosa.load(wav, sr=int(m.sample_rate), mono=True)
    out = m.embed(a.astype(np.float32))
    L = out.logits
    if isinstance(L, dict):
        L = list(L.values())[0]
    L = np.asarray(L)
    L = L.reshape(-1, L.shape[-1])  # (windows, classes)
    return L.mean(0)  # 録音(チャンク)平均


def run(name, wavs, true_idx, other_idx):
    n = 0; correct_pair = 0; argmax_self = 0; tops = {}
    for w in wavs:
        try:
            lg = logits_of(w)
        except Exception as e:  # noqa
            continue
        n += 1
        if lg[true_idx] > lg[other_idx]:
            correct_pair += 1            # 2値(自種 vs 相手)で自種が勝つ
        am = int(lg.argmax())
        if am == true_idx:
            argmax_self += 1             # 全14796クラスで自種がトップ
        tops[classes[am]] = tops.get(classes[am], 0) + 1
    top5 = sorted(tops.items(), key=lambda x: -x[1])[:5]
    print(f"\n[{name}] n={n}", flush=True)
    print(f"  自種 vs 相手の2値勝率: {correct_pair}/{n} = {correct_pair/max(n,1):.3f}", flush=True)
    print(f"  全クラスargmaxが自種: {argmax_self}/{n} = {argmax_self/max(n,1):.3f}", flush=True)
    print(f"  argmax先 top5: {top5}", flush=True)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
kal = glob.glob("data/ood_processed/tier1/Eastern_Spot-billed_Duck/*.wav")[:N]
mal = glob.glob("data/processed/Mallard/*.wav")[:N]
print(f"カルガモ {len(kal)}件 / マガモ {len(mal)}件で評価", flush=True)
run("カルガモ(spbduc)", kal, I_SPB, I_MAL)
run("マガモ(mallar3)", mal, I_MAL, I_SPB)
