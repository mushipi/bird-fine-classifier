# Training Guide

学習・評価ステップの詳細解説。ハイパラの意味、ログの読み方、評価指標の解釈。

---

## 1. 学習スクリプト

### 構造

`src/bird_fine/training/train.py` の流れ：

```
1. config.yaml 読み込み
2. Dataset構築（train/val/test）
3. ASTモデルロード（事前学習重み + 分類ヘッド初期化）
4. TrainingArguments構築
5. HuggingFace Trainerで学習
6. best modelを models/ast-duck/ に保存
7. label_map.csv をモデルディレクトリにコピー
```

HuggingFace Trainerを使う理由：
- 学習ループ・mixed precision・grad accumulation・早期停止が引数1つで切り替えられる
- TensorBoard連携が標準
- 自前のループより堅牢

### コマンド

```powershell
# 通常学習
uv run python -m bird_fine.training.train

# サブセット50件 + 1epochで動作確認（OOMチェック、バグ確認）
uv run python -m bird_fine.training.train --dry-run

# ハイパラ上書き
uv run python -m bird_fine.training.train --epochs 5 --batch-size 2
```

---

## 2. ハイパラの意味

### 基本（config.yaml `training:` 配下）

| パラメータ | デフォルト | 意味 | 調整の目安 |
|---|---|---|---|
| `num_train_epochs` | 15 | 学習エポック数 | EarlyStoppingで早期終了するので多めでOK |
| `per_device_train_batch_size` | 4 | 1GPUのバッチサイズ | OOMなら下げる、余裕あれば上げる |
| `per_device_eval_batch_size` | 8 | 評価時のバッチ | 推論はメモリ少なめ、訓練の2倍OK |
| `gradient_accumulation_steps` | 4 | 何ステップ分の勾配を貯めて1回更新 | effective batch = train_bs × this |
| `learning_rate` | 5e-5 | 学習率 | Transformer fine-tune標準。下げるならbatch小さい時 |
| `warmup_ratio` | 0.1 | 全ステップの何%をwarmupに | 序盤の発散防止、10%固定でOK |
| `weight_decay` | 0.01 | L2正則化の強さ | 過学習対策。データ少ない時は0.05まで上げる選択肢 |
| `fp16` | true | 混合精度学習 | RTX系GPUで2倍速、メモリ半減 |
| `gradient_checkpointing` | true | 活性化を再計算してメモリ節約 | VRAM 8GBには必須 |
| `early_stopping_patience` | 3 | val改善しない連続epoch数 | 0で無効。3〜5が標準 |

### 「effective batch size」の感覚

```
effective batch = per_device_train_batch_size × gradient_accumulation_steps
                = 4 × 4 = 16
```

つまり実質バッチ16で学習している。学習率は effective batch に比例させるのが定石（**linear scaling rule**）。

**8GBのVRAMで限界**：fp16 + grad_checkpointing でも `batch_size=8` は厳しい可能性。
最初は4でスタート、安定したら6〜8に挑戦。

### 学習率の調整指針

| 状況 | 学習率の目安 |
|---|---|
| 標準（事前学習を活かす） | 3e-5 〜 5e-5 |
| データ多くて速く動かしたい | 1e-4 |
| 学習が振動する | 1e-5 まで下げる |
| 早期に過学習する | lr据え置き + weight_decay引き上げ |

---

## 3. メトリクス

学習中・評価で計算される指標：

| 指標 | 意味 | 解釈 |
|---|---|---|
| **loss** | クロスエントロピー損失 | 小さいほど良い。trainとvalの乖離 = 過学習 |
| **accuracy** | 全体正解率 | クラス不均衡では多数派に引きずられる |
| **precision_macro** | 種ごとの精度の単純平均 | 「Mallardって予測したうち実際にMallardだった割合」の平均 |
| **recall_macro** | 種ごとの再現率の単純平均 | 「実際のMallardのうち拾えた割合」の平均 |
| **f1_macro** | 各種F1の単純平均 | **最重要指標** 。少数派の性能も反映 |

### best modelの選び方

`config.yaml`:
```yaml
metric_for_best_model: "f1_macro"
greater_is_better: true
```

各エポックでval F1を見て、最良のものを保存。EarlyStoppingも同じ指標を基準。

### 期待値の目安

データ件数次第だが、ベースライン目標：

| F1 macro | 評価 |
|---|---|
| < 0.5 | データ不足 or 学習失敗。dry-runで何が起きているか確認 |
| 0.5 〜 0.7 | 動いている。改善の余地多い |
| **0.7 〜 0.85** | 実用ライン。本PJのテスト目標 |
| > 0.85 | 高精度。データ品質か、リークの可能性も確認 |

ランダム予測なら8クラスで `1/8 = 0.125`。0.5でも「学習している」と言える。

---

## 4. TensorBoardの見方

学習中にTensorBoardログが `models/ast-duck/runs/` 配下に出る。

```powershell
# 別ターミナルで起動
uv run tensorboard --logdir models/ast-duck/runs
# ブラウザで http://localhost:6006 を開く
```

### 見るべきチャート

| グラフ名 | チェックポイント |
|---|---|
| `train/loss` | 単調減少しているか？振動するならlr高すぎ |
| `eval/loss` | trainより遅れて減少。乖離が大きい → 過学習 |
| `eval/f1_macro` | 上昇しているか？頭打ちならEarlyStoppingが効く |
| `train/learning_rate` | warmupで増加→cosineで減少（typical schedule） |
| `train/grad_norm` | 急上昇 = 不安定。clip_grad対象 |

### 過学習のサイン

```
train_loss: ↓↓↓ どんどん下がる
eval_loss:  ↓↗ 途中から上がる
eval_f1:    ↑↘ 途中から下がる
```

→ `load_best_model_at_end=true` でベストエポックの重みが保存されるので、最終モデルは大丈夫だが、
   そもそも過学習が深刻ならweight_decay引き上げ・augmentation追加を検討。

---

## 5. 評価スクリプト

### 何をするか

`src/bird_fine/training/evaluate.py`:

1. 学習済みモデルをロード
2. テストセットで全推論
3. メトリクス計算 + 混同行列 + Attention可視化
4. 結果を `outputs/eval_{timestamp}/` に保存

### コマンド

```powershell
uv run python -m bird_fine.training.evaluate

# モデル指定
uv run python -m bird_fine.training.evaluate --model-dir models/ast-duck

# attention可視化スキップ（高速）
uv run python -m bird_fine.training.evaluate --no-attention
```

### 出力ファイル

```
outputs/eval_20260520_220000/
├── confusion_matrix_norm.png  # 正規化版（行ごとに％）
├── confusion_matrix_raw.png   # 件数版
├── confusion_matrix.csv       # 数値データ
├── report.json                # 全メトリクス
├── predictions.csv            # y_true, y_pred の対応
└── attention/
    └── attention_sample_*.png  # CLSトークンの注目度可視化
```

### 混同行列の読み方

行=正解、列=予測。対角線が太いほど高精度。

```
        Mallard  Teal  Pintail  ...
Mallard   0.85   0.05    0.10   ...  ← 正解Mallard、85%正答、Teal/Pintailに誤分類10%/5%
Teal      0.03   0.92    ...
...
```

**注目するパターン**：

| パターン | 意味・対処 |
|---|---|
| 対角線が極端に薄い種 | データ不足 or 鳴き声が独特すぎ → 追加DL検討 |
| AとBで相互誤分類が多い | 音響的に類似 → augmentation強化、データ追加 |
| 1種だけ全部多数派に流れる | クラス不均衡で吸収されている → 重み付け検討 |

### Attention可視化の見方

`attention/attention_sample_*.png`:

- 上段: CLSトークンから各時間-周波数パッチへの注目度（横軸=パッチ番号）
- 下段: 最終層のフルattention行列

**着目すべき点**：
- 鳴き声の瞬間にattentionが集中しているか？
- 無音区間（ゼロパディング）に無駄に注目していないか？
- 誤分類サンプルでattentionがどこを見ているか？

---

## 6. よくある改善手法

データを増やせない前提での改善策：

### A. データ拡張（augmentation）

```python
# Dataset.__getitem__ に追加
# - SpecAugment: 時間/周波数のマスキング
# - RandomTimeShift: ±0.5秒のずらし
# - GaussianNoise: SN比を変える
```

実装するなら`src/bird_fine/data/dataset.py` の `__getitem__` 内、feature_extractor呼び出し前後。

### B. クラス重み

```python
# train.py で計算
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.arange(8), y=train_labels)
# Trainerにcustom lossで渡す
```

### C. レイヤーごとの学習率（discriminative LR）

```python
# 後段ほど大きく、前段は小さくfine-tune
# transformersのoptimizer設定をカスタムで構築
```

### D. ハイパラチューニング

`optuna` 等でlr/weight_decayをサーチ。データ少ない時はk-fold CVと組み合わせ。
