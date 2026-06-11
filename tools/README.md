# tools/ — Foundation Model 埋め込み＋プローブ（AST頭打ち打開の別系統）

AST fine-tune が test f1_macro ≈ 0.838 で頭打ち（真因＝音響的類似性/モデル容量、docs/experiments.md run01-09）。
鳥特化 Foundation Model の埋め込みを凍結利用し、軽量プローブで 0.838 超えを狙う系統。
**フレームワーク非互換（TF↔torch）を避けるため、抽出は隔離環境で .npz に出して受け渡す。**

## 構成

| 環境 | 中身 | 用途 |
|---|---|---|
| `tools/perch_embed/.venv` | tensorflow-cpu + perch-hoplite | Perch 2.0 (perch_v2_cpu) 抽出 |
| `tools/aves_embed/.venv` | CPU torch + esp-aves + sklearn/matplotlib | BirdAVES 抽出 ＋ プローブ学習/評価 |

- 重み: Perch=Kaggle public を perch-hoplite が自動取得（認証不要、`references/weights/kagglehub_cache/`）。
  BirdAVES=`references/weights/birdaves/`（取得済）。
- **注意**: HFミラー `references/weights/perch_v2/`(392M) は **GPU専用エクスポートでCPU不可**。使わない（削除可）。

## 実行手順（mushipi-pc から data/raw・data/processed を取得後）

### 1. 埋め込み抽出 → data/embeddings/{model}/{split}.npz

```bash
# Perch 2.0（生mp3を32kHz再デコード, 高忠実）
cd tools/perch_embed
.venv/bin/python extract_perch.py --splits train val test --source raw

# BirdAVES-biox-large（既存16kHzチャンク）
cd ../aves_embed
.venv/bin/python extract_birdaves.py --splits train val test
```

### 2. プローブ学習（torch環境で。論文準拠ハイパラは config.yaml: probe）

```bash
cd tools/aves_embed
export PYTHONPATH=$(git rev-parse --show-toplevel)/src
PY=.venv/bin/python
$PY -m bird_fine.training.train_probe --model perch    --probe linear
$PY -m bird_fine.training.train_probe --model perch    --probe mlp
$PY -m bird_fine.training.train_probe --model birdaves --probe linear
$PY -m bird_fine.training.train_probe --model birdaves --probe attentive   # transformer系のみ可
```
→ `models/probe_{model}_{probe}/`（probe.pt, scaler.npz, meta.json, history.json）

### 3. 評価（test, chunk＋録音単位）→ AST 0.838 と比較

```bash
$PY -m bird_fine.training.eval_probe --run probe_perch_linear
# 他 run も同様
```
→ `outputs/{run}_{ts}/`（report.json, confusion_matrix*, predictions.csv）
出力に `vs AST baseline (0.838): UP/DOWN` を表示。

## 検証済み（2026-06-10, データ非依存部分）

- Perch 2.0 CPU ロード＋埋め込み: 5s→(1,1,1536) on CPU（kagglehub匿名DL、認証不要）
- BirdAVES ロード＋埋め込み: mean(1024) + seq(32,1024)
- 抽出器 CSV→音声→npz グルー（raw 32k / processed 16k 両経路）
- プローブ学習/評価 全経路（linear/mlp/attentive, chunk＋録音集約, f1/AUROC/top1/混同行列）

**未検証＝実データ依存**: 実 439 録音での抽出・プローブ学習・0.838 との実比較（mushipi-pc 復帰後）。

## 運用メモ（repo CLAUDE.md 準拠）

- 新 run は `docs/experiments.md` に事前登録（`run<NN>(pre)`）してから学習、結果は別コミット。
- プローブ実験も「1 run 1 変更」。まず Perch-linear をベースラインに据える。
- 頭打ち診断: `outputs/*/predictions.csv` を xc_id 単位で見て、誤分類が特定録音に集中していないか確認
  （run09 の XC197026 juvenile / XC349677 のような録音単位集中の再点検）。
