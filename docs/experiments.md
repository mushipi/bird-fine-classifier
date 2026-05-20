# 学習実験ログ

カモ8種AST分類器の学習run記録。新規runを回したら下に追記すること。

---

## run01 — baseline (2026-05-19)

**出力:** `models/ast-duck/`
**コミット:** （未コミット / 初回学習）

### ハイパラ

| 項目 | 値 |
|---|---|
| pretrained | MIT/ast-finetuned-audioset-10-10-0.4593 |
| num_train_epochs | 15 |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 4 (effective batch = 16) |
| learning_rate | 5.0e-5 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| fp16 | true |
| gradient_checkpointing | true |
| early_stopping_patience | 3 |
| metric_for_best_model | f1_macro |
| データ拡張 | **なし** |

### データ規模

- train: 約2016件 / val: 約313件 / test: 約313件（10秒チャンク）
- 対象8種: マガモ / コガモ / オナガガモ / ハシビロガモ / ヒドリガモ / キンクロハジロ / ホオジロガモ / ホシハジロ

### 結果（best = checkpoint-504, epoch 4）

| metric | value |
|---|---|
| eval_accuracy | 0.888 |
| eval_f1_macro | **0.848** |
| eval_precision_macro | 0.905 |
| eval_recall_macro | 0.837 |
| eval_loss | 0.622 |

### 経過（主要マイルストーン）

| epoch | step | train_loss | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | 0.33 | 0.495 | 0.809 |
| 2 | 252 | 0.07 | 0.627 | 0.798 |
| 3 | 378 | 0.07 | 0.624 | 0.774 |
| **4** | **504** | **0.02** | **0.622** | **0.848** ← best |
| 5 | 630 | 0.002 | 0.684 | 0.833 |
| 6 | 756 | 0.001 | 0.630 | 0.819 |
| 7 | 882 | 0.0002 | 0.622 | 0.840 → EarlyStop |

### 所見

- **明確な overfit**: epoch 2 で既に train_loss が 0.07 まで急落、epoch 3 以降は 0.001 以下に張り付き
- eval_loss は epoch 1 (0.495) から悪化して epoch 4 で底打ち、改善せず
- f1_macro も epoch 4 がピーク、その後3エポック改善なしで EarlyStopping
- 原因仮説: (a) lr=5e-5 が大きすぎる, (b) weight_decay=0.01 では正則化が弱い, (c) データ拡張がない

---

## run02 — overfit対策 (2026-05-19)

**出力:** `models/ast-duck-v2/`
**変更点:** ハイパラ調整 + SpecAugment 追加
**学習実時間:** 23:32〜23:41（約9分、step 504 / max 1890 で EarlyStop）

### ハイパラ（run01 からの差分）

| 項目 | run01 | run02 | 意図 |
|---|---|---|---|
| learning_rate | 5.0e-5 | **2.0e-5** | 学習率を下げて細かく fit、ピーク後の急速 overfit を抑制 |
| weight_decay | 0.01 | **0.1** | L2 正則化を10倍に強化 |
| early_stopping_patience | 3 | **2** | 早めに止めて計算節約 |
| SpecAugment | なし | **freq_mask 24×2 + time_mask 80×2** | train特徴量にマスキング、汎化性能向上 |

### データ拡張仕様（SpecAugment）

- `dataset.py:DuckChunkDataset` に `spec_augment_cfg` 引数追加
- **train_ds のみ適用**（val/test は素の特徴量で評価）
- torchaudio.transforms の `FrequencyMasking` / `TimeMasking` を使用
- AST入力 (time, freq) → torchaudio想定の (freq, time) に転置してマスキング → 戻す

### 期待効果

- train_loss と eval_loss の乖離縮小
- ピーク epoch が後ろにずれる（4 → 6〜8 想定）
- f1_macro が 0.848 を上回ることが目標

### 結果（best = checkpoint-252, epoch 2）

| metric | value | run01 比較 |
|---|---|---|
| best_epoch | **2** | run01: 4 → 前倒し |
| eval_accuracy | 0.850 | 0.888 ↓ |
| eval_f1_macro | **0.826** | 0.848 ↓ (悪化) |
| eval_precision_macro | 0.882 | 0.905 ↓ |
| eval_recall_macro | 0.807 | 0.837 ↓ |
| eval_loss | 0.365 | 0.622 ↑ (改善) |

### 経過（主要マイルストーン）

| epoch | step | train_loss(末尾) | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | 0.51 | 0.591 | 0.758 |
| **2** | **252** | **0.22** | **0.365** | **0.826** ← best |
| 3 | 378 | 0.14 | 0.501 | 0.809 |
| 4 | 504 | 0.06 | 0.625 | 0.800 → EarlyStop |

### 所見

- **f1_macro は悪化（0.848 → 0.826）**。期待外れ。ただし eval_loss は明確に改善（0.622 → 0.365）しており、確信度の校正は良くなっている
- **train_loss の落下速度は鈍化を確認**（run01 ep2 で 0.07 → run02 ep2 で 0.22）。正則化自体は効いている
- ピーク epoch は後ろにずれず**前倒し**（4 → 2）になった。期待と逆の挙動
- 原因仮説:
  - (a) **`patience=2` が早すぎる**: lr を半減したのに評価機会も減らしたのは方針として噛み合っていない。`patience=3〜4` に戻すべき
  - (b) **SpecAugment のマスクが強い可能性**: freq_mask 24/128 ≈ 19%、time_mask 80/1024 ≈ 8% × 2 ≈ 16%。AST 事前学習時より強い設定の可能性
  - (c) **同時変更しすぎで切り分け不能**: 4つのパラメータを一気に変えたため、どれが寄与/阻害したか分離不可
  - (d) SpecAugment が eval にも漏れていないか実装確認が必要（`src/dataset.py`）

### 次のアクション候補

- `src/dataset.py` の SpecAugment 実装を確認（train_ds のみ適用か）
- run03 では**変更を1つに絞る**: 例えば「lr=2e-5 + weight_decay=0.05（やや控えめ）+ patience=4」で SpecAugment 一旦オフ
- もしくは run03 で SpecAugment 単独評価（lr/wd/patience を run01 と同じに戻す）

---

## メモ

- baseline (`models/ast-duck/`) は常に保護する。新規 run は必ず別 output_dir へ
- 学習中に `Win+Ctrl+Shift+B`（GPUドライバ再起動）は避ける — CUDA コンテキストが吹っ飛ぶ可能性
- TensorBoard: `uv run tensorboard --logdir models/ast-duck-v2/runs`
