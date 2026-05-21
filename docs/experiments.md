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

## run03 — lr 単独評価 (2026-05-20)

**出力:** `models/ast-duck-v3/`
**コミット:** 事前登録 `560a702` / 結果は本セクションのコミットで記録
**ステータス:** 学習完了（2026-05-21）。学習自体は正常完走したが直後にシステム異常終了し成果物が一部破損 → 無傷の checkpoint-252 から復旧済み（経緯は journal.md 2026-05-21）。

### ハイパラ（run01 / run02 からの差分）

| 項目 | run01 | run02 | run03 | run03 の意図 |
|---|---|---|---|---|
| learning_rate | 5.0e-5 | 2.0e-5 | **2.0e-5** | 主軸。lr↓ を単独評価 |
| weight_decay | 0.01 | 0.1 | **0.01** | run02 の 0.1 を baseline に戻し固定変数化 |
| early_stopping_patience | 3 | 2 | **4** | lr↓ への整合補正。早期打ち切り回避 |
| SpecAugment | なし | あり | **なし** | 補助を排除、主軸に集中 |

run01 baseline から見た実質変更は **lr↓ と patience↑ のみ**（patience は lr↓ の従属補正）。実質「lr 単独評価」。

### データ規模

- run01 / run02 と同一（train 約2016 / val 約313 / test 約313、10秒チャンク、8種）

### 仮説と予測（学習前に固定）

run02 から確定した事実:
- eval_loss は lr↓+wd↑ で 0.622 → 0.365 に改善（正則化は機能）
- f1_macro は 0.848 → 0.826 に悪化、best epoch は 4 → 2 に前倒し
- SpecAugment の eval 漏れは実装確認で否定（→ journal.md 2026-05-20）

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | f1 悪化の主因は patience=2 の早期打ち切りと wd=0.1 の過剰正則化であり、lr↓ 自体は f1 に有害でない | **f1_macro ≥ 0.848**（run01 と同等以上に回復） |
| **H2** | lr↓ で overfit 進行が緩み、ピークが後ろにずれる | **best_epoch ≥ 5**（run01 は 4） |
| **H3** | lr↓ 単独でも校正は改善する | **eval_loss < 0.622**（run01 比） |

### 検証後の分岐（学習前に固定）

- **H1 成立（f1 ≥ 0.848）**: 「run02 失敗の主因は patience/wd」が確定 → run04 で wd を単独評価
- **H1 不成立（f1 < 0.848）**: lr↓ 自体が f1 に不利な疑い → lr を 5e-5 系に戻す方向で再検討
- **H2 不成立（best_epoch < 5）**: lr は overfit タイミングに効きにくい → 別の正則化手段を検討

### 結果（best = checkpoint-252, epoch 2）

| metric | value | run01 比 | run02 比 |
|---|---|---|---|
| best_epoch | **2** | 4 → 前倒し | 2 → 同じ |
| eval_accuracy | 0.895 | 0.888 ↑ | 0.850 ↑ |
| eval_f1_macro | **0.875** | 0.848 ↑（改善 / run01〜03 で最高）| 0.826 ↑ |
| eval_precision_macro | 0.899 | 0.905 ↓ | 0.882 ↑ |
| eval_recall_macro | 0.865 | 0.837 ↑ | 0.807 ↑ |
| eval_loss | 0.317 | 0.622 ↓（改善）| 0.365 ↓（改善）|

> 結果値は TensorBoard ログ（`runs/`）と無傷の checkpoint-252 / 630 から復元。学習直後のシステム異常終了で root 成果物・checkpoint-756 の一部ファイルがゼロ埋め破損したため（→ journal.md 2026-05-21）。

### 経過（主要マイルストーン）

| epoch | step | train_loss | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | 0.36 | 0.535 | 0.749 |
| **2** | **252** | **0.12** | **0.317** | **0.875** ← best |
| 3 | 378 | 0.026 | 0.442 | 0.847 |
| 4 | 504 | 0.003 | 0.400 | 0.854 |
| 5 | 630 | 0.0006 | 0.513 | 0.819 |
| 6 | 756 | 0.0002 | （欠損）| （欠損）→ EarlyStop |

epoch 6 の eval 値はシステム異常終了で tfevents 未フラッシュ + `trainer_state.json` 破損のため復元不能。EarlyStopping は patience=4 到達で作動（best=ep2 に対し ep3〜6 が4回連続非改善）。

### 所見

- **H1 成立** — f1_macro 0.875 ≥ 0.848（予測どおり）。run01 baseline を上回り run01〜03 で最高。→ 検証分岐どおり「run02 失敗の主因は patience=2 / wd=0.1」が確定。**次は run04 で weight_decay を単独評価**。
- **H2 不成立** — best_epoch=2（予測は ≥5）。train_loss は ep2 末で 0.12、ep3 で 0.026、ep4 で 0.003 と急降下。lr↓（5e-5→2e-5）では overfit のタイミングは後ろにずれなかった。→ 検証分岐どおり「lr は overfit タイミングに効きにくい、別の正則化手段が必要」。
- **H3 成立** — eval_loss 0.317 < 0.622（予測どおり）。lr↓ 単独でも確信度の校正は改善した。
- **run03 のまとめ**: lr↓ 単独で f1 は回復（H1）、ただし overfit タイミングは不変（H2）。lr は f1 の絶対値には効くが overfit の進行そのものは止めない、という切り分けができた。run04 の wd 単独評価で、wd が f1・overfit タイミングの両面にどう効くかを見る。
- **補足（インシデント）**: 学習は正常完走したが直後にシステムが異常終了（Kernel-Power 41）し、root 成果物と checkpoint-756 の一部ファイルがゼロ埋め破損。best モデルは無傷の checkpoint-252 から復旧済み。再学習は不要。詳細・再発防止は journal.md 2026-05-21。

---

## run04 — weight_decay 単独評価 (2026-05-21)

**出力:** `models/ast-duck-v4/`
**コミット:** 事前登録 `501192a` / 結果は本セクションのコミットで記録
**ステータス:** 学習完了（2026-05-21）。GPU 150W 制限下で正常完走、システム異常終了の再発なし（約17分）。

### ハイパラ（run01 / run02 / run03 からの差分）

| 項目 | run01 | run02 | run03 | run04 | run04 の意図 |
|---|---|---|---|---|---|
| weight_decay | 0.01 | 0.1 | 0.01 | **0.05** | 主軸。wd↑ を単独評価 |
| learning_rate | 5.0e-5 | 2.0e-5 | 2.0e-5 | **2.0e-5** | run03 から据え置き（固定変数）|
| early_stopping_patience | 3 | 2 | 4 | **4** | run03 から据え置き（固定変数）|
| SpecAugment | なし | あり | なし | **なし** | 補助オフ継続、主軸に集中 |

run03 から見た実質変更は **weight_decay のみ**（output_dir 分離は学習ダイナミクスに無関係）。実質「wd 単独評価」。

### データ規模

- run01〜03 と同一（train 約2016 / val 約313 / test 約313、10秒チャンク、8種）

### 仮説と予測（学習前に固定）

run03 から確定した事実:
- lr↓ で f1_macro は 0.848 → 0.875 に回復（H1 成立）。run02 失敗の主因は patience=2 / wd=0.1 と確定
- best_epoch は 2 のまま（H2 不成立）。lr↓ では overfit タイミングは後ろにずれない
- train_loss は ep2 末 0.12 → ep3 0.026 → ep4 0.003 と急降下（overfit は速いまま）

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | wd 強化（0.01→0.05）で overfit 進行が緩み、best epoch が後ろにずれる | **best_epoch ≥ 3**（run03 は 2）|
| **H2** | wd=0.05 は過剰正則化ではなく、f1 は run03 と同等以上を維持 | **f1_macro ≥ 0.87**（run03 は 0.875）|
| **H3** | train_loss の急降下が run03 より鈍る | **epoch 4 時点の train_loss > 0.003**（run03 比で鈍化）|

### 検証後の分岐（学習前に固定）

- **H1 成立（best_epoch ≥ 3）**: wd は overfit タイミングに効く → run05 で wd をさらに強める、もしくは wd + SpecAugment 併用へ
- **H1 不成立（best_epoch = 2）**: wd も overfit タイミングに効かない → 主軸を SpecAugment 等のデータ拡張へ移す
- **H2 不成立（f1_macro < 0.87）**: wd=0.05 は過剰正則化 → run05 で wd=0.03 に弱めて再評価

### 結果（best = checkpoint-630, epoch 5）

| metric | value | run03 比 |
|---|---|---|
| best_epoch | **5** | 2 → +3（後退）|
| eval_accuracy | 0.888 | 0.895 ↓ |
| eval_f1_macro | **0.850** | 0.875 ↓（−0.025）|
| eval_precision_macro | 0.876 | 0.899 ↓ |
| eval_recall_macro | 0.846 | 0.865 ↓ |
| eval_loss | 0.435 | 0.317 ↑（悪化）|

### 経過（主要マイルストーン）

| epoch | step | train_loss | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | 0.38 | 0.548 | 0.771 |
| 2 | 252 | 0.15 | 0.345 | 0.836 |
| 3 | 378 | 0.027 | 0.565 | 0.823 |
| 4 | 504 | 0.002 | 0.532 | 0.819 |
| **5** | **630** | **0.001** | **0.435** | **0.850** ← best |
| 6 | 756 | 0.0002 | 0.466 | 0.845 |
| 7 | 882 | 0.0002 | 0.475 | 0.845 |
| 8 | 1008 | 0.0001 | 0.473 | 0.843 |
| 9 | 1134 | 0.0001 | 0.479 | 0.843 → EarlyStop |

EarlyStopping は patience=4 到達で作動（best=ep5 に対し ep6〜9 が4回連続非改善）。

### 所見

- **H1 成立** — best_epoch=5 ≥ 3（予測どおり）。run03（epoch 2）から overfit ピークが3エポック後退した。**wd は overfit タイミングに効く**と確定。run03 で残った「lr 以外の正則化手段が要る」という問いに、wd が答えになった。
- **H2 不成立** — eval_f1_macro 0.850 < 0.87（予測は ≥0.87）。run03 の 0.875 から約 2.5pt 低下。wd=0.05 は f1 にはコストがかかる。
- **H3 不成立** — epoch 4 の train_loss 0.002 で、run03（0.003）と同等以上に速い。wd は train_loss の急降下（訓練データの暗記速度）を鈍らせない。一方で eval 曲線は run03 と違いピーク後も 0.84〜0.85 で安定 → **wd は「eval ピークの位置と安定性」に効き、「train の暗記速度」には効かない**、と切り分けられた。
- **run04 のまとめ**: wd↑ は overfit タイミングを後ろにずらすが f1 を下げるトレードオフ。f1 の絶対値は run03（0.875）が依然最高。検証分岐に従い、run05 で **wd=0.03**（run03 0.01 と run04 0.05 の中間）を評価し、overfit 後退の利得を f1 コスト最小で取れる点を探す。
- **補足（PSU）**: GPU 電力上限を 150W に絞って学習したところ、システム異常終了は再発しなかった。run03 のクラッシュは電源容量不足の傍証（n=1。確証には継続観察が要る）。

---

## run05 — weight_decay 中間点評価 (2026-05-21)

**出力:** `models/ast-duck-v5/`
**コミット:** （学習後に記録）
**ステータス:** 学習前記録（事前登録）— 条件と仮説を学習前に固定。結果・経過・所見は学習完了後に追記。

### ハイパラ（run03 / run04 からの差分）

| 項目 | run03 | run04 | run05 | run05 の意図 |
|---|---|---|---|---|
| weight_decay | 0.01 | 0.05 | **0.03** | 主軸。wd の中間点を評価 |
| learning_rate | 2.0e-5 | 2.0e-5 | **2.0e-5** | 据え置き（固定変数）|
| early_stopping_patience | 4 | 4 | **4** | 据え置き（固定変数）|
| SpecAugment | なし | なし | **なし** | 補助オフ継続 |

run04 から見た実質変更は **weight_decay のみ**（0.05→0.03）。run03→04→05 で wd を 0.01 / 0.05 / 0.03 と振り、dose-response の中間点を埋める。

### データ規模

- run01〜04 と同一（train 2011 / val 313 / test 338、10秒チャンク、8種）

### 仮説と予測（学習前に固定）

run03 / run04 から確定した事実:
- wd=0.01（run03）: best_epoch 2 / f1_macro 0.875
- wd=0.05（run04）: best_epoch 5 / f1_macro 0.850
- wd↑ は overfit ピークを後退させる（H1 成立済み）が、f1 を下げる（トレードオフ）
- train_loss の急降下は wd では変わらない（wd は eval ピーク位置・安定性のみに効く）

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | wd=0.03 でも overfit ピークは run03 より後退する | **best_epoch ≥ 3** |
| **H2** | wd=0.03 の f1 は run04 を上回り、f1 コストを抑えられる | **f1_macro ≥ 0.86**（run04 0.850 を超え run03 0.875 寄りに回復）|
| **H3** | wd の効果は単調 — best_epoch・f1 とも run03 と run04 の中間に収まる | **best_epoch ∈ [3, 4]** かつ **f1_macro ∈ [0.855, 0.870]** |

### 検証後の分岐（学習前に固定）

- **H1・H2 ともに成立**: wd=0.03 が overfit 後退と f1 維持を両立 → 採用候補。run06 で SpecAugment 併用など別軸の上積みへ
- **H2 不成立（f1_macro < 0.86）**: wd=0.03 でも f1 コストが大きい → f1 最良は run03 の wd=0.01。overfit 対策は別軸（SpecAugment / dropout）へ切り替える
- **H3 不成立（中間に収まらない）**: wd の効果は非線形。dose-response が単純でないため wd の振り方を再設計する

### 結果（学習完了後に記入）

| metric | value | run03 比 | run04 比 |
|---|---|---|---|
| best_epoch | — | | |
| eval_accuracy | — | | |
| eval_f1_macro | — | | |
| eval_precision_macro | — | | |
| eval_recall_macro | — | | |
| eval_loss | — | | |

### 経過（学習完了後に記入）

| epoch | step | train_loss | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| — | | | | |

### 所見（学習完了後に記入）

— H1 / H2 / H3 の成否と、検証分岐に沿った次アクションを記入。

---

## メモ

- baseline (`models/ast-duck/`) は常に保護する。新規 run は必ず別 output_dir へ
- 学習中に `Win+Ctrl+Shift+B`（GPUドライバ再起動）は避ける — CUDA コンテキストが吹っ飛ぶ可能性
- TensorBoard: `uv run tensorboard --logdir models/ast-duck-v2/runs`
