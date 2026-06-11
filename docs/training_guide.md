# Training Guide

学習・評価ステップの詳細解説。ハイパラの意味・ログの読み方・Outlier Exposure の運用。

---

## 1. 学習の流れ（通常: 8 種）

```bash
# 動作確認（dry-run: 各種 6 チャンクのサブセットで 1 epoch）
uv run python -m bird_fine.training.train --dry-run

# 本学習
uv run python -m bird_fine.training.train

# ハイパラ上書き
uv run python -m bird_fine.training.train --epochs 5 --batch-size 2
```

`train.py` の処理フロー:

```
1. config.yaml 読み込み
2. Dataset 構築（train/val/test）
3. モデルロード
   - init_from 指定なし → 事前学習済み AST から初期化
   - init_from 指定あり → 既存チェックポイントから分類ヘッドを拡張して初期化
4. 位置埋め込みを 3s 用にリサイズ（1214 → 352 次元）
5. DuckTrainer で学習（WeightedRandomSampler + カスタム CE Loss）
6. best model を models/{output_dir}/ に保存
```

---

## 2. Outlier Exposure 学習（9 クラス）

OOD 検知精度を上げるために "other" クラスを追加する場合の手順。

### 前提: "other" データの準備

```bash
# 学習データに "other" チャンクを追記
uv run python -m bird_fine.data.prepare_other_class --dry-run  # 件数確認
uv run python -m bird_fine.data.prepare_other_class
```

### config.yaml の設定

```yaml
model:
  num_labels: 9
  init_from: "models/ast-duck-v10"  # 既存モデルから分類ヘッドを拡張

training:
  metric_for_best_model: "f1_macro_8class"  # "other" に引きずられない

other_class:
  loss_alpha: 1.5  # "other" の CE loss 重み（固定値のみ）
```

### Double Dipping 禁止

WeightedRandomSampler と Loss 重みの**二者択一**が鉄則。

```
NG: Sampler で均等化 + Loss にデータ数ベース重みを掛ける
    → "other" 勾配が二重に爆発して 8 種の決定境界が破壊される

OK: Sampler で均等化 + Loss は固定値 α のみ
```

### モデル選択の基準

| 指標 | 用途 |
|---|---|
| `eval_f1_macro_8class` | **モデル選択基準**（8 種のみ）|
| `eval_f1_macro` | 参考（9 クラス全体）|
| `eval_other_recall` | 参考（"other" の検知率）|

"other" の recall が低くても問題ない。目的は energy 空間のキャリブレーションであり、
softmax 空間での "other" 分類精度は副産物（→ `docs/ood_detection.md` 参照）。

---

## 3. ハイパーパラメータ

| パラメータ | 現行値 | 意味 |
|---|---|---|
| `learning_rate` | 2.0e-5 | run03 で確立。Transformer fine-tune の標準域 |
| `weight_decay` | 0.03 | run05 で最適点を確認 |
| `per_device_train_batch_size` | 4 | RTX 3060 Ti (8GB) の上限付近 |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `early_stopping_patience` | 4 | run03 以降固定 |
| `fp16` | true | RTX 3060 Ti で有効 |
| `gradient_checkpointing` | true | VRAM 節約 |

**1 run 1 変更の原則**: 複数パラメータを同時変更すると切り分け不能になる（run02 の失敗）。

---

## 4. メトリクス

| 指標 | 意味 |
|---|---|
| `f1_macro` | 全クラスの F1 単純平均。クラス不均衡に強い |
| `f1_macro_8class` | 8 種のみの F1 平均（OE 学習時のモデル選択基準）|
| `other_recall` | "other" クラスの再現率（OE 学習時のみ出力）|
| `accuracy` | 全体正解率（参考値）|

**val → test の乖離に注意:**

run03〜05 で「val 最良モデルが test で最低」という逆転が発生した（選択バイアス）。
最終評価は必ず test セットで行い、val だけで判断しない。

---

## 5. 評価

```bash
uv run python -m bird_fine.training.evaluate
uv run python -m bird_fine.training.evaluate --model-dir models/ast-duck-v11
```

出力:

```
outputs/eval_{timestamp}/
├── confusion_matrix_norm.png
├── confusion_matrix_raw.png
├── report.json
└── predictions.csv
```

**混同行列の読み方**: 行=正解ラベル、列=予測ラベル。対角線が太いほど正確。

---

## 6. OOD 評価

```bash
uv run python -m bird_fine.analysis.ood_eval
uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-v11
```

AUROC と推奨 energy_threshold を算出して `species_taxonomy.yaml` に反映する。
詳細は `docs/ood_detection.md` 参照。

---

## 7. TensorBoard

```bash
uv run tensorboard --logdir models/ast-duck-v11/runs
# http://localhost:6006
```

| グラフ | 見るポイント |
|---|---|
| `train/loss` | 単調減少しているか |
| `eval/loss` | train との乖離が大きい → 過学習 |
| `eval/f1_macro_8class` | 上昇傾向か（OE 学習時はこちらを見る）|
| `train/grad_norm` | 急上昇 = 不安定 |

---

## 8. 実験管理ルール

CLAUDE.md の規則に従い、各 run を記録する:

1. **学習前（事前登録）**: `docs/experiments.md` に run セクションを先行作成
   - ハイパラ差分・仮説・予測値・検証後の分岐を記載
   - `run<NN>(pre): 条件と仮説を学習前に記録` でコミット
2. **学習後**: 結果を埋めて `run<NN>: <結果>` でコミット
3. **設計判断**: `docs/journal.md` に追記

コミットメッセージ例:
```
run11: "other"クラス追加 AUROC energy 0.894→0.932 / test f1_8class +0.011
```
