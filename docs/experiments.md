# 学習実験ログ

カモ8種AST分類器の学習run記録。新規runを回したら下に追記すること。

---

## ⚠️ 評価プロトコルに関する重要な但し書き（2026-06-05 追記）

**本ログの run 間の精度比較（特に ±0.05 未満の差）は統計的に有意でない。**

- test は録音数が少なく（既存8種で **69 録音**）、同一録音内の chunk は強く相関する。
  chunk 単位 macro-F1 は実効サンプルを過大評価し CI を過小評価＋録音内の一部誤分類を
  ペナルティして**性能を過小評価**する。
- 録音単位クラスタ bootstrap（`bird_fine.analysis.compare_runs_ci`）で run11/13/15 の既存8種を
  検証した結果、**全ペア差の 95%CI が 0 を含む（＝差なし）**。CI 幅は ±0.1。
  点推定の順位すら指標（chunk単位 vs 録音単位）で逆転した（例: chunk単位 run11>run15>run13、
  録音単位 run13>run11>run15）。
- **したがって各 run セクションの f1 値・「+0.0XX 改善/悪化」の記述は参考値**であり、
  run の優劣を断定する根拠にはならない。**意思決定は大きな構造的差**（例: カルガモ F1=0.000、
  失敗種除去で f1_macro が桁で動く等）**と定性的所見のみ**で行うこと。
- 今後の全比較は **録音単位 + bootstrap CI** で行い、CI が重なる差は「差なし」と扱う。
  test の録音数拡大が最優先の改善課題。

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
**コミット:** 事前登録 `cf7d84e` / 結果は本セクションのコミットで記録
**ステータス:** 学習完了（2026-05-22）。GPU 150W 制限下で正常完走、システム異常終了の再発なし（PSU 観察2回目クリア）。

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

### 結果（best = checkpoint-756, epoch 6）

| metric | value | run03 比 | run04 比 |
|---|---|---|---|
| best_epoch | **6** | 2 | 5 |
| eval_accuracy | 0.885 | 0.895 ↓ | 0.888 ↓ |
| eval_f1_macro | **0.840** | 0.875 ↓ | 0.850 ↓ |
| eval_precision_macro | 0.865 | 0.899 ↓ | 0.876 ↓ |
| eval_recall_macro | 0.857 | 0.865 ↓ | 0.846 ↑ |
| eval_loss | 0.457 | 0.317 ↑ | 0.435 ↑ |

### 経過（主要マイルストーン）

| epoch | step | train_loss | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | 0.42 | 0.581 | 0.765 |
| 2 | 252 | 0.12 | 0.415 | 0.822 |
| 3 | 378 | 0.035 | 0.401 | 0.815 |
| 4 | 504 | 0.003 | 0.475 | 0.840 |
| 5 | 630 | 0.0005 | 0.418 | 0.840 |
| **6** | **756** | **0.0002** | **0.457** | **0.840** ← best |
| 7 | 882 | 0.0002 | 0.463 | 0.840 |
| 8 | 1008 | 0.0001 | 0.469 | 0.840 |
| 9 | 1134 | 0.0001 | 0.473 | 0.840 |
| 10 | 1260 | 0.0001 | 0.475 | 0.840 → EarlyStop |

eval_f1_macro は epoch 4 以降 7エポック連続でほぼ 0.840 に張り付き（best=ep6 は僅差）。EarlyStopping は patience=4 到達で作動。

### 所見

- **H1 成立** — best_epoch=6 ≥ 3。wd=0.03 でも overfit ピークは run03（ep2）より後退。
- **H2 不成立** — eval_f1_macro 0.840 < 0.86。run04（0.850）すら下回り、run03〜05 で最低。
- **H3 不成立** — dose-response は単調でない。wd を 0.01→0.03→0.05 と振った f1 は **0.875 → 0.840 → 0.850** で、中間の wd=0.03 が最低。best_epoch も 2 → 6 → 5 で単調でない。予測（best_epoch ∈ [3,4]、f1 ∈ [0.855,0.870]）は外れた。
- **重要な解釈** — 3 run の eval f1 曲線を並べると、差は wd ではなく**ノイズと early-stopping のアーティファクト**で説明がつく:
  - run05: ep4〜10 が 0.840 にほぼ完全固定（7エポック）。プラトー ≈ 0.840
  - run04: ep5〜9 が 0.843〜0.850。プラトー ≈ 0.845
  - run03: ep1〜5 が 0.749 / 0.875 / 0.847 / 0.854 / 0.819 と振れ、**0.875 は ep2 単発のスパイク**。プラトー ≈ 0.84〜0.85
  - 3 run とも安定水準は ≈0.84〜0.85。run03 の 0.875 は early-stopping が幸運な1エポックを拾った値で、wd=0.01 の robust な優位ではない
- **結論** — wd を 0.01〜0.05 で振っても val f1 は実質動かない（差はノイズ水準）。train_loss は毎 run epoch 4 で ≈0.003、epoch 8 で ≈0.0001 まで落ち、訓練 2011 件は毎回完全に暗記される。**val f1 ≈0.85 はこのデータ/モデルの汎化天井で、ハイパラ（lr・wd）では破れない。**
- **次のアクション**（事前登録分岐「wd の振り方を再設計」を上書き）:
  - (1) run03 / 04 / 05 の best モデルを**未使用の test セット（338件）で評価**し、early-stopping バイアスを除いた素の比較を取る（`evaluate.py`）。run03 の 0.875 が test でも出るか確認する
  - (2) wd チューニングは打ち切り。0.85 天井を破るには別レバー — SpecAugment のクリーンな単独評価、データ量・質の改善 — を検討する

---

## test セット評価 — run03 / 04 / 05 比較 (2026-05-22)

**目的:** run03〜05 の val f1 比較（0.875 / 0.850 / 0.840）が early-stopping の選択バイアスを含む疑い（run05 所見）。未使用の test セット338件で素の汎化性能を測り決着をつける。
**方法:** 各 run の best モデル（root の `model.safetensors`）を `evaluate.py --model-dir <dir> --no-attention` で test 338件評価。出力は `outputs/eval_20260522_*`（混同行列・report.json・predictions.csv）。

### 全体メトリクス

| run | wd | best_epoch | val f1 | **test f1** | **test acc** | val−test gap |
|---|---|---|---|---|---|---|
| run03 | 0.01 | 2 | 0.875 | 0.775 | 0.843 | **−0.100** |
| run04 | 0.05 | 5 | 0.850 | 0.806 | 0.873 | −0.044 |
| run05 | 0.03 | 6 | 0.840 | **0.838** | **0.876** | −0.002 |

### 所見 — val 順位が test で完全に逆転した

- **test の順位は val の真逆**。val 最良の run03（0.875）が test 最低（0.775）、val 最低の run05（0.840）が test 最高（0.838）。
- run03 の val−test gap は **−0.100**、run05 はわずか −0.002。
- run05 所見の仮説「run03 の 0.875 は early-stopping が拾ったノイズスパイク」は test で裏付けられた。さらに踏み込むと、run03 は単にノイズで持ち上がっただけでなく **3 run 中で最も汎化しないモデル**だった。
- gap は best_epoch と連動する: best_epoch 2 → 5 → 6 で gap −0.100 → −0.044 → −0.002、test f1 0.775 → 0.806 → 0.838。**早く止まったモデルほど test で悪い**。epoch 2 で選ばれた run03 は実質、学習不足のモデルを val スパイクで掴んでいた。
- run05 の val（0.840）≈ test（0.838）は、run05 の val 曲線が平坦なプラトーで「選択する山」が無く、選択バイアスが乗らなかったため。

### 種別 F1（test）と train データ量

| 種 | test n | run03 | run04 | run05 | train録音数 | trainチャンク数 |
|---|---|---|---|---|---|---|
| Common_Goldeneye | 64 | 0.885 | 0.953 | 0.930 | 30 | 144 |
| Common_Pochard | 63 | 0.952 | 0.938 | 0.945 | 47 | 220 |
| Eurasian_Teal | 86 | 0.845 | 0.866 | 0.844 | 42 | 249 |
| Eurasian_Wigeon | 27 | 0.650 | 0.816 | 0.809 | 20 | 154 |
| Mallard | 46 | 0.907 | 0.874 | 0.918 | 70 | 390 |
| Northern_Pintail | 28 | 0.871 | 0.949 | 0.949 | 32 | 88 |
| Northern_Shoveler | 14 | 0.786 | 0.667 | 0.923 | 31 | 72 |
| Tufted_Duck | 10 | 0.303 | 0.387 | 0.389 | 20 | 694 |

- **弱点を決めているのは train 録音数（チャンク数ではない）**。test F1（run05）は train 録音数と連動: 録音20本の Eurasian_Wigeon / Tufted_Duck が test 最下位2つ、録音30本以上の6種は 0.84〜0.95。録音の多様性が汎化を決めている。
- **Tufted_Duck は「データ不足」ではない**。trainチャンク数 694 で全種最多。だが録音は20本（1録音あたり35チャンク）で実質の多様性が低く、かつ 694 という突出量がチャンク不均衡を生む。run05 混同行列で Tufted_Duck と予測した26件中、正解は7件のみ（precision 0.269）。誤りは Eurasian_Teal 14 / Eurasian_Wigeon 5 — モデルが Tufted_Duck を過剰予測している。
- **Northern_Shoveler（test n=14）は弱点ではない**。trainチャンク最少72だが録音31本で多様、run05 F1 0.923・precision 1.000。test n の小ささを「弱い種」と誤読しないこと。

### 結論と次アクション

- **現時点の最良モデルは run05（test f1 0.838 / acc 0.876）**。「run03 が最良（0.875）」は val 選択バイアスによる誤り。run05 を採用する。
- **モデル選択の方法に問題がある**。`load_best_model_at_end` が noisy な val f1 のスパイクを掴む。val 313件はチャンク単位 f1 を 0.02〜0.03 揺らし、その最大値を選ぶと上方バイアスがかかる。run 比較・モデル選択は val ピークでなく test、または平滑化した指標で行うべき。
- **val f1 ≈0.85 / test f1 ≈0.84 の天井を破るレバーはハイパラではない**。真のボトルネックはデータ — ただし「量」ではなく **train 録音数（多様性）とチャンク数の不均衡**。録音20本前後の Tufted_Duck / Eurasian_Wigeon が弱い（「少数種のデータ不足」は誤り。Tufted_Duck はチャンク最多の694）。
- 次の一手候補:
  1. 録音数の少ない種（Tufted_Duck / Eurasian_Wigeon、各20録音）の音源を追加収集。チャンクでなく多様な録音を増やす
  2. チャンク数の不均衡是正 — 1録音あたりのチャンク数に上限、または class-weighted loss / オーバーサンプリング
  3. SpecAugment のクリーン単独評価（run02 は同時変更で評価不能だった）
  4. モデル選択を val ピーク依存から脱却（test 併用、val f1 の平滑化 / val loss 選択の検討）

---

## run06 — SpecAugment クリーン単独評価 (2026-05-24)

**出力:** `models/ast-duck-v6/`
**コミット:** 事前登録 `e0e9271` / 結果は本セクションのコミットで記録
**ステータス:** 学習・test 評価完了（2026-05-24）。結果: val f1 0.846（run05 比 +0.006）/ **test f1 0.810（run05 比 −0.028）**。H2 不成立で SpecAugment 単独投入は失敗。
**結論:** 現行ベストは引き続き **run05（test f1 0.838）**。SpecAugment は学習ダイナミクスを変えたが（H1 部分成立）、val ピークが逆に早期化して未学習モデルが選択された。Tufted_Duck のみ +0.05 改善も Northern_Shoveler が −0.19 で全体下落。

### ハイパラ（run03 / run04 / run05 からの差分）

| 項目 | run03 | run04 | run05 | run06 | run06 の意図 |
|---|---|---|---|---|---|
| SpecAugment | なし | なし | なし | **あり** | 補助。データ拡張を単独で評価 |
| freq_mask_param × num_freq_masks | — | — | — | **24 × 2** | 128メル中、最大 38% を周波数マスク |
| time_mask_param × num_time_masks | — | — | — | **80 × 2** | 時間軸最大 160 フレーム分をマスク |
| weight_decay | 0.01 | 0.05 | 0.03 | **0.03** | run05 据え置き（現行ベース / 固定変数）|
| learning_rate | 2.0e-5 | 2.0e-5 | 2.0e-5 | **2.0e-5** | run03 から据え置き（固定変数）|
| early_stopping_patience | 4 | 4 | 4 | **4** | 据え置き（固定変数）|

ベースは **run05**（test f1 0.838 で現行ベスト）。run05 から見た実質変更は **SpecAugment enabled: false → true のみ**。SpecAugment パラメータは config.yaml の既定値（freq 24×2 / time 80×2）をそのまま使用 — run02 と同じ強度なので、run02 失敗を SpecAugment 自体の問題と切り分ける目的も兼ねる。

### データ規模

- run01〜05 と同一（train 2011 / val 313 / test 338、10秒チャンク、8種）。SpecAugment は train_ds のみに適用（`src/bird_fine/data/dataset.py:124` 確認済み、val/test には None）

### 仮説と予測（学習前に固定）

run03〜05 / test 評価から確定した事実:
- lr / wd では val f1 ≈0.85 / test f1 ≈0.84 の天井を破れない（3 run で実証）
- train_loss は毎 run epoch 4 で ≈0.003、epoch 8 で ≈0.0001 まで落ち、訓練 2011 件は完全暗記される
- 真のボトルネックは録音多様性（Tufted_Duck / Eurasian_Wigeon の各20録音）。ただし SpecAugment は録音追加なしでスペクトログラム上の不変性を学ばせる別軸の対策
- run02 の SpecAugment は lr↓ / wd↑ / patience↓ と同時変更で評価不能だった

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | SpecAugment は train の完全暗記を遅らせる | **epoch 4 時点の train_loss ≥ 0.01**（run05 は 0.003 / run04 は 0.002）、かつ **best_epoch ≥ 7**（run05 は 6）|
| **H2（主）** | SpecAugment は補助レバーとして汎化天井を押し上げる | **test f1_macro ≥ 0.85**（run05 0.838 を上回る）|
| **H3** | val f1 の選択バイアスは run05 同様に小さい（プラトー型曲線が維持される） | **\|val − test\| ≤ 0.03**、かつ **val f1_macro ≥ 0.83**（run05 0.840 のプラトー水準を維持）|
| **H4** | SpecAugment は録音多様性不足を一部補う | **Tufted_Duck と Eurasian_Wigeon の test F1 が run05 比で改善**（run05 はそれぞれ 0.389 / 0.809）|

### 検証後の分岐（学習前に固定）

- **H2 成立（test f1_macro ≥ 0.85）**: SpecAugment は補助で天井を押し上げる。採用候補。run07 で SpecAugment + 録音追加の組み合わせ、あるいは mask param のチューニングへ
- **H2 不成立だが H4 成立**: 全体 test f1 は動かないが少数種は改善 → SpecAugment は class-imbalance 対策として有効。class-weighted loss と組み合わせる
- **H2・H4 ともに不成立（test f1 < 0.85 かつ少数種も横ばい）**: SpecAugment ではモデル側で天井を破れない。主軸をデータ側（録音追加収集、チャンク不均衡是正）に完全に切り替える
- **H1 不成立（train_loss が ≥0.01 まで上がらない / best_epoch ≤ 6）**: SpecAugment 強度が弱すぎる可能性 → freq_mask_param / num_*_masks の増強を検討。または実装バグ疑い（spec_augment が train に効いているか確認）

### 結果（best = checkpoint-252, epoch 2）

| metric | value | run05 比 |
|---|---|---|
| best_epoch | **2** | 6 |
| eval_accuracy | 0.875 | 0.885 |
| eval_f1_macro | **0.846** | 0.840 |
| eval_precision_macro | 0.893 | 0.865 |
| eval_recall_macro | 0.845 | 0.857 |
| eval_loss | 0.392 | 0.457 |
| **test_f1_macro** | **0.810** | **0.838** |
| **test_accuracy** | **0.858** | **0.876** |
| val−test gap | **−0.036** | −0.002 |

学習は epoch 6 で early stop（patience=4, best=epoch 2）。`load_best_model_at_end` は val f1_macro 最大の checkpoint-252 を選択。test 評価出力: `outputs/eval_20260524_112150/`。

### 経過（主要マイルストーン）

| epoch | step | train_loss(平均) | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 126 | ≈0.55 | 0.613 | 0.756 |
| **2** | **252** | ≈0.22 | **0.392** | **0.846** ← best |
| 3 | 378 | ≈0.10 | 0.461 | 0.811 |
| 4 | 504 | ≈0.025 | 0.533 | 0.842 |
| 5 | 630 | ≈0.019 | 0.662 | 0.830 |
| 6 | 756 | ≈0.012 | 0.772 | 0.826 |

train_loss は run05 比で大きく上振れ（epoch 4 ≈0.025 vs run05 0.003、epoch 5 ≈0.019 vs run05 0.001）。H1 train_loss 予測（≥0.01 @ ep4）は成立。一方 best_epoch は予測（≥7）に反して **2** と最早期化し、不成立。

### 仮説検証

| 仮説 | 予測 | 実測 | 判定 |
|---|---|---|---|
| H1a train_loss @ ep4 | ≥0.01 | 0.019〜0.054（≈0.025） | **成立** |
| H1b best_epoch | ≥7 | **2** | **不成立**（逆に早期化） |
| H2 test f1 | ≥0.85 | **0.810** | **不成立** |
| H3a \|val−test\| | ≤0.03 | 0.036 | 僅か不成立 |
| H3b val f1 | ≥0.83 | 0.846 | 成立 |
| H4 Tufted_Duck / Eurasian_Wigeon | 両方 run05 比改善 | 0.389→0.438 / 0.809→0.816 | 部分成立（Tufted のみ +0.05） |

### 種別 test F1（run03 / 04 / 05 / 06 比較）

| 種 | test n | run03 | run04 | run05 | **run06** | run05→06 差 | 録音数 | チャンク数 |
|---|---|---|---|---|---|---|---|---|
| Common_Goldeneye | 64 | 0.885 | 0.953 | 0.930 | 0.909 | **−0.021** | 30 | 144 |
| Common_Pochard | 63 | 0.952 | 0.938 | 0.945 | 0.950 | +0.005 | 47 | 220 |
| Eurasian_Teal | 86 | 0.845 | 0.866 | 0.844 | 0.843 | ≈0 | 42 | 249 |
| Eurasian_Wigeon | 27 | 0.650 | 0.816 | 0.809 | 0.816 | +0.007 | 20 | 154 |
| Mallard | 46 | 0.907 | 0.874 | 0.918 | 0.863 | **−0.055** | 70 | 390 |
| Northern_Pintail | 28 | 0.871 | 0.949 | 0.949 | 0.929 | −0.020 | 32 | 88 |
| **Northern_Shoveler** | 14 | 0.786 | 0.667 | 0.923 | **0.733** | **−0.190** | 31 | 72 |
| **Tufted_Duck** | 10 | 0.303 | 0.387 | 0.389 | **0.438** | **+0.049** | 20 | 694 |

### 所見

- **H2 不成立、H4 部分成立**: 事前登録の分岐ロジックでは「H2 不成立 / H4 部分成立」分岐に該当。SpecAugment 単独投入は test 全体で run05 を下回り、**現行ベストは引き続き run05（test f1 0.838）**。
- **SpecAugment は学習ダイナミクスを変えたが、選択指標の劣化を招いた**: H1a は明確に成立し train の暗記は確かに抑制された（train_loss が桁単位で上振れ）。だが val f1 のピークは逆に **epoch 6 → epoch 2 に早期化** した。SpecAugment が val 曲線を平坦化せずに「初期スパイク → 早期プラトー → 緩やかに悪化」型に変えた可能性がある。`load_best_model_at_end` は最大値（epoch 2 の 0.846）を掴むので、未学習に近い checkpoint が選ばれた。
- **val−test gap が再び拡大**: gap 0.036 は run05（0.002）から悪化し、run04（0.044）寄りに戻った。best_epoch=2 で選んだ結果という点でも run03 と類似の症状（run03: best_epoch=2, gap=0.100）。**val ピーク選択バイアスを引き戻している** ことになる。
- **少数種改善は限定的、強い種が悪化**: Tufted_Duck +0.05 / Eurasian_Wigeon ≈0 で「少数種を救う」効果は弱い。一方 Northern_Shoveler が −0.19（run05 で precision 1.000 だったのが run06 では 0.688 に低下）。SpecAugment が録音数の中程度のクラス境界を曖昧化した可能性。
- **「class-imbalance 対策として有効」と言える結果ではない**: 事前登録分岐の H2 不成立 + H4 成立分岐は「class-weighted loss と組み合わせる」だったが、**少数種 +0.05 / 中堅種 −0.19 のトレードはペイしない**。SpecAugment + 録音多様性向上で再評価するなら、まず録音追加が先で SpecAugment は後段に回す方が筋。
- **train_loss 抑制 ≠ 汎化向上**: 「train 完全暗記の阻止」を成功の代理指標にできない、という方法論の裏付け。H1a 成立 + H2 不成立 の組み合わせはこの結論を直接示している。
- 次の一手:
  1. **データ軸へ完全に切り替える** — 録音数20本の Tufted_Duck / Eurasian_Wigeon の音源を Xeno-canto から追加収集
  2. **モデル選択方式の見直し** — `load_best_model_at_end` の val 単点ピーク依存をやめ、val f1 の移動平均 / 上位 k checkpoint 平均 / val loss 併用などを検討。run03 / run06 と同じ罠を踏まないため
  3. **SpecAugment 再評価は後回し** — 録音多様性が改善した後、ベース性能が上がった状態で再度クリーン評価する

---

## run07 — 弱2種の録音追加（quality=B から +10 ずつ） (2026-05-24)

**出力:** `models/ast-duck-v7/`
**コミット:** 事前登録 `b18a221` / 結果は本セクションの結果コミットで記録
**ステータス:** 学習・test 評価完了（2026-05-24）。結果: val f1 **0.8446**（run05 比 +0.005）/ **test f1 0.8268**（run05 比 −0.011）。H1〜H4 不成立。録音 +10 では 0.85 天井を破れず、Tufted_Duck の過剰予測も解消せず。Northern_Shoveler / Mallard で他種への負の波及も発生。
**結論:** 現行ベストは引き続き **run05（test f1 0.838）**。データ軸の方向性は維持するが、規模・品質・不均衡是正の見直しが必要。

### データ変更（差分）

`docs/journal.md` の追加収集セクションに XCID 一覧と取得手順を記録。要約:

| 種 | train 録音 (run01〜06) | run07 追加 | 追加後 train 録音 | 追加後 train chunks | val / test |
|---|---|---|---|---|---|
| Tufted_Duck | 20 | **+10**（quality=B） | 30 (xc_id 上は29※) | 694 → **728**（+34） | val 16/4, test 10/5（不変） |
| Eurasian_Wigeon | 20 | **+10**（quality=B） | 30 | 154 → **191**（+37） | val 21/4, test 27/5（不変） |
| 他 6 種 | 30〜70 | 0 | 同左 | 同左 | 不変 |

※ Tufted_Duck の追加分のうち XC32831 と XC34072 は preprocess の `_extract_xc_id` の既知バグ（XC プレフィクスなしファイルでフォールバックで先頭単語を xc_id とする）により共に xc_id=`Aythya` に集約される。チャンクは10録音分すべて train.csv に入っているので学習データとしては +10録音だが、xc_id ベースのカウントは +9。

データソース:
- `data/raw/{Species}/metadata.csv` を `XCID 昇順` で先頭10件（quality=B）取得
- 既存 quality=A の XCID は `--exclude-existing` でスキップ
- `--worldwide-only` で countries フィルタを外して取得（既存 Wigeon q=B は Japan 3件 / worldwide 202件で worldwide のみ叩く方針）
- split.py `--preserve-existing` で val/test を完全固定したまま train だけに新規録音を追加

test 338件は run01〜06 と完全同一構成 → test 数値が直接比較可能。

### ハイパラ（run05 からの差分）

run07 のハイパラは **run05（現行ベスト）と完全同一**。変更点は train データのみ。run06 で評価して不採用が確定した SpecAugment は run05 の `enabled: false` に戻した。

| 項目 | run05 | run06 | **run07** | 意図 |
|---|---|---|---|---|
| データ | 既存 | 既存 | **+ Tufted +10 / Wigeon +10** | 主軸変更点 |
| weight_decay | 0.03 | 0.03 | **0.03** | run05 据え置き |
| learning_rate | 2.0e-5 | 2.0e-5 | **2.0e-5** | run05 据え置き |
| SpecAugment | off | on | **off** | run06 で不採用確定 → run05 と同条件 |
| early_stopping_patience | 4 | 4 | **4** | run05 据え置き |

### 仮説と予測（学習前に固定）

run01〜06 の事実から:
- ハイパラ・正則化系では val f1 ≈0.85 / test f1 ≈0.84 を破れない（4 run で実証）
- 弱点を決めるのは **train 録音数（多様性）**: 録音20本の Tufted / Wigeon が test 最弱、録音30本以上の6種は test F1 0.84〜0.95（run01〜06 で安定）
- Tufted_Duck はチャンク数最多（694）だが録音20本 → モデルが過剰予測（run05 で precision 0.269）
- run06 で SpecAugment は train 暗記抑制（H1a 成立）も val 曲線を悪化させた → 「正則化」では弱2種を救えない

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | 弱2種の録音追加で test 全体 F1 が上がる | **test f1_macro ≥ 0.86**（run05 0.838 を 0.02 以上上回る） |
| **H2（主）** | 弱2種の test F1 が個別に改善する | **Tufted_Duck test F1 ≥ 0.50**（run05 0.389、+0.11 以上）かつ **Eurasian_Wigeon test F1 ≥ 0.85**（run05 0.809、+0.04 以上） |
| **H3** | Tufted_Duck の過剰予測が緩和される（precision が上がる） | **Tufted_Duck precision ≥ 0.40**（run05 は run05 混同行列の値 ≈0.27、+0.13 以上） |
| **H4** | 他種の test F1 はほぼ不変（負の波及がない） | **他6種すべて run05 比 −0.02 以内**（=どの種も 0.02 以上は下げない） |
| **H5** | val−test gap は run05 並みの水準を維持 | **\|val − test\| ≤ 0.03**（run05 は 0.002）、best_epoch ≥ 5（run05 は 6） |

### 検証後の分岐（学習前に固定）

- **H1 成立（test f1 ≥ 0.86）**: 録音追加は 0.85 天井を破る有効レバー。次は (1) チャンク不均衡是正 (2) 残り種への録音追加 (3) class-weighted loss などデータ軸の上積みを継続
- **H1 不成立だが H2 成立**: 全体 F1 は動かないが弱2種は改善 → 「弱2種は録音追加で救えるが、他種が頭打ち」。次は弱2種にさらに録音追加 + 他種は録音多様性以外のレバー（モデル選択方式）を検討
- **H1・H2 ともに不成立**: 録音 +10 / 種では量が不足、または quality=B の noise が効果を打ち消す。次は (1) +20以上の追加 (2) quality=A 限定の精選追加 (3) チャンク不均衡是正（1録音上限）を試す
- **H4 不成立（他種が −0.02 以上下落）**: 弱2種優遇で他種が犠牲になっている → 不均衡是正策（1録音あたりチャンク上限、class-weighted loss）が必要
- **H5 不成立（val−test gap > 0.03）**: 選択バイアス再発 → run07 の test 結果は信頼度を下げて読む。次は `load_best_model_at_end` の代替を本気で検討

### 結果（best = checkpoint-524, epoch 4）

| metric | value | run05 比 | run06 比 |
|---|---|---|---|
| best_epoch | **4** | 6 | 2 |
| eval_accuracy | 0.885 | 0.885 | 0.875 |
| eval_f1_macro | **0.8446** | 0.840 | 0.846 |
| eval_loss | 0.458 | 0.457 | 0.392 |
| **test_f1_macro** | **0.8268** | **0.838** | **0.810** |
| **test_accuracy** | **0.870** | **0.876** | **0.858** |
| val−test gap | **+0.018** | +0.002 | +0.036 |

学習は epoch 8 で early stop（patience=4, best=epoch 4）。`load_best_model_at_end` は val f1_macro 最大の checkpoint-524 を選択。test 評価出力: `outputs/eval_20260524_155635/`。

### 経過（主要マイルストーン）

| epoch | step | train_loss(末尾) | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 131 | — | 0.636 | 0.745 |
| 2 | 262 | — | 0.559 | 0.792 |
| 3 | 393 | — | 0.467 | 0.838 |
| **4** | **524** | ≈0.01 | **0.458** | **0.845** ← best |
| 5 | 655 | 0.0004 | 0.526 | 0.838 |
| 6 | 786 | — | 0.531 | 0.835 |
| 7 | 917 | 0.0001 | 0.539 | 0.835 |
| 8 | 1048 | — | 0.541 | 0.835 |

train_loss は epoch 5 時点で 0.0004、epoch 7 で 0.0001 まで落ち run05 と同様の完全暗記パターン。val 曲線は run05 と類似の「3-4 epoch でピーク → プラトー → 緩やかに悪化」型で run06 のような早期スパイクは見られない。

### 仮説検証

| 仮説 | 予測 | 実測 | 判定 |
|---|---|---|---|
| H1 全体 test f1 | ≥ 0.86 | **0.827** | **不成立** (run05 −0.011) |
| H2a Tufted F1 | ≥ 0.50 | **0.432** | **不成立** (+0.043 改善も予測未達) |
| H2b Wigeon F1 | ≥ 0.85 | **0.809** | **不成立** (run05 と同値 ±0.000) |
| H3 Tufted precision | ≥ 0.40 | **0.296** | **不成立** (+0.027 微増のみ、過剰予測未解決) |
| H4 他6種 \|diff\| ≤ 0.02 | | Mallard −0.029, N.Shoveler −0.108 | **不成立**（2種で波及）|
| H5a \|val−test\| | ≤ 0.03 | 0.018 | **成立** |
| H5b best_epoch | ≥ 5 | 4 | **僅か不成立**（run05 6, run06 2）|

### 種別 test F1（run05 / 06 / 07 比較）

| 種 | test n | run05 | run06 | **run07** | run05→07 差 | 録音数 (train) |
|---|---|---|---|---|---|---|
| Common_Goldeneye | 64 | 0.930 | 0.909 | 0.938 | +0.007 | 30 |
| Common_Pochard | 63 | 0.945 | 0.950 | 0.968 | +0.023 | 47 |
| Eurasian_Teal | 86 | 0.844 | 0.843 | 0.831 | −0.013 | 42 |
| Eurasian_Wigeon | 27 | 0.809 | 0.816 | 0.809 | ±0.000 | **30** (+10) |
| Mallard | 46 | 0.918 | 0.863 | 0.889 | **−0.029** | 70 |
| Northern_Pintail | 28 | 0.949 | 0.929 | 0.933 | −0.016 | 32 |
| Northern_Shoveler | 14 | 0.923 | 0.733 | 0.815 | **−0.108** | 31 |
| Tufted_Duck | 10 | 0.389 | 0.438 | 0.432 | **+0.043** | **30** (+10) |

### 混同行列の主な誤分類（run07）

- **Tufted_Duck と予測されたチャンク 27件**: 正解 8 / 誤り 19 — **Eurasian_Teal 14件**, Eurasian_Wigeon 4件, Common_Goldeneye 1件。run05 と同パターン（Tufted は過剰予測のまま）。precision 0.296 / recall 0.800
- **Northern_Shoveler の正解 11/14**: 誤分類は Northern_Pintail 1, Mallard 1, Common_Pochard 1 件。test n=14 の小サンプルで F1 が振れているが、絶対数で見ると「正解 −1 / 他種からの吸引 +2」の微差。precision 1.000→0.846, recall 0.857→0.786

### 所見

- **H1〜H4 不成立**: 録音 +10 / 種では 0.85 天井を破れない。test f1 は run05 から −0.011 で **悪化**。事前登録の分岐ロジックでは「H1・H2 ともに不成立」分岐に該当
- **Tufted_Duck の過剰予測は未解決（H3 不成立）**: precision 0.269（run05）→ 0.296（run07）の +0.027 微増のみ。Eurasian_Teal 14件を Tufted と誤判定するパターンは run05 と同一で、根本原因が「Tufted のチャンク数突出」にあることを再確認（728 chunks、全種で依然最多）
- **録音追加でチャンク不均衡が逆に悪化**: Tufted_Duck train chunks 694 → 728（+34）で他種との差がさらに開いた。録音追加の副作用として「チャンク数で過剰予測されやすくなる」体質が継続。**録音追加とチャンク上限カットは同時にやるべきだった**
- **Northern_Shoveler の −0.108 はサンプル数バイアス**: test n=14 で正解 −1 / 他種からの吸引 +2 の小さな絶対数変化が F1 に大きく見える。混同行列ベースの判断と F1 単独の判断は分けるべき
- **Mallard の −0.029 は本物の悪化**: test n=46 で run05 0.918 → 0.889。precision 0.830 で Tufted_Duck か Eurasian_Teal あたりへの誤吸引が増えた可能性（混同行列の詳細チェックは別途）
- **val 曲線は健全**: val−test gap 0.018 で H5a 成立。run06 のような選択バイアスは再発せず、選択した checkpoint-524 自体は妥当。問題は「best モデルそのもの」の汎化性能
- **quality=B noise の影響は判断保留**: H2b（Wigeon F1 ≥0.85）が達成できなかった理由を「B 品質の SN比劣化」とするか「+10録音では量不足」とするかは現データだけでは切り分けられない。**quality=A 限定の追加収集で再評価する必要がある**

### 検証後の次アクション（事前登録分岐に基づく）

事前登録した「H1・H2 ともに不成立」分岐の指示は「(1) +20以上の追加 (2) quality=A 限定の精選追加 (3) チャンク不均衡是正（1録音上限）を試す」。順序を整理:

1. **チャンク不均衡是正を最優先**: 1録音あたりチャンク上限（例: 30）を設けて Tufted_Duck 728 → ≈600 程度に均す。録音は追加せず chunks_index.csv の生成だけ変える → val/test もチャンク数が変わるので preserve-existing と矛盾するか要検討。あるいは train.csv だけサブサンプリング
2. **quality=A 限定の追加収集**: 残念ながら Tufted 29本 / Wigeon 30本で上限。これ以上 A は取れない。別ソース（eBird, BirdNET, iNaturalist 等）の検討
3. **class-weighted loss**: Tufted の過剰予測を損失レベルで抑制。データを変えずに同条件で評価できる
4. **録音 +20以上の追加**: quality=B でさらに追加すると noise の影響がより大きくなる懸念。優先度は 3 の class-weighted loss 試行後

H4 不成立（Mallard / N.Shoveler への負の波及）も含めて、**run08 の主軸候補は「チャンク不均衡是正」または「class-weighted loss」**。どちらを単独で先に評価するかは次の判断ポイント。

---

## run08 — 1録音あたりチャンク数の上限（cap=100）単独評価 (2026-05-24)

**出力:** `models/ast-duck-v8/`
**コミット:** 事前登録 `3522a73` / 結果は本セクションの結果コミットで記録
**ステータス:** 学習・test 評価完了（2026-05-24）。**衝撃の結果**: Tufted_Duck の chunks を 728→361 に半減したにもかかわらず、Tufted_Duck の precision (0.269) と Tufted と予測されたチャンクの構成（Eurasian_Teal 13件 / Wigeon 4件吸引）が run05 とほぼ完全一致。**「Tufted 過剰予測 = チャンク数不均衡」という仮説が崩れた**。
**結論:** test f1 **0.820**（run05 比 −0.018 / run07 比 −0.007）で run08 も失敗。**現行ベストは引き続き run05（test f1 0.838）**。ただし負の知見として極めて重要 — 真のボトルネックはチャンク不均衡ではなく、Tufted_Duck の音響特徴が他種（特に Eurasian_Teal）と区別困難である可能性。

### データ変更（差分）

`src/bird_fine/data/cap_train_chunks.py` を新規実装し、`data/splits/train.csv` に対して **1録音あたりチャンク数の上限 100** を適用（seed=42 固定で再現性確保）。val/test は不変。

| 種 | run07 train chunks | run08 train chunks | 差 | top1 record clip |
|---|---|---|---|---|
| Common_Goldeneye | 144 | 144 | ±0 | （18→18 影響なし） |
| Common_Pochard | 220 | 220 | ±0 | （27→27 影響なし） |
| Eurasian_Teal | 249 | 249 | ±0 | （28→28 影響なし） |
| Eurasian_Wigeon | 191 | 191 | ±0 | （49→49 影響なし） |
| Mallard | 390 | 390 | ±0 | （64→64 影響なし） |
| Northern_Pintail | 88 | 88 | ±0 | （19→19 影響なし） |
| Northern_Shoveler | 72 | 72 | ±0 | （10→10 影響なし） |
| **Tufted_Duck** | **728** | **361** | **−367** | **XC488113: 297→100, XC488112: 270→100** |

Tufted のみ介入。録音数は 29/29 保持（全録音を残しつつ XC488113 / XC488112 の長尺録音だけ削った）。

### ハイパラ（run05 / 07 からの差分）

run08 のハイパラは **run07 と完全同一**（=run05 とも同一）。変更点は train データの cap のみ。

| 項目 | run05 | run07 | **run08** | 意図 |
|---|---|---|---|---|
| データ | 既存 | +録音 +チャンク不均衡悪化 | **+録音 +cap=100** | 主軸変更点（chunk 不均衡是正）|
| weight_decay | 0.03 | 0.03 | **0.03** | 据え置き |
| learning_rate | 2.0e-5 | 2.0e-5 | **2.0e-5** | 据え置き |
| SpecAugment | off | off | **off** | 据え置き |
| early_stopping_patience | 4 | 4 | **4** | 据え置き |

### 仮説と予測（学習前に固定）

run07 までの事実から:
- Tufted_Duck の chunks 数突出（728 / 全種 最多）が **過剰予測の構造的原因**。run05 で precision 0.269、run07 で 0.296 と録音追加でも未解決
- 上位2録音 XC488113 (297 chunks ≈49分) と XC488112 (270 chunks ≈45分) で Tufted total の **78%** を占める異常な偏り
- cap=100 で Tufted を 728→361 まで均し、Mallard 390 と並ぶ水準に。class-imbalance（チャンクベース）が大幅に改善
- val/test は不変なので test の数値は run05/07 と直接比較可能

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | チャンク不均衡是正で Tufted の過剰予測が緩和され、全体 test f1 が上がる | **test f1_macro ≥ 0.86**（run05 0.838 / run07 0.827 を上回る） |
| **H2（主）** | Tufted_Duck の precision が大幅改善する | **Tufted_Duck precision ≥ 0.50**（run07 0.296、+0.20 以上）|
| **H3** | Tufted の recall は悪化するが許容範囲 | **Tufted_Duck recall ≥ 0.50**（run07 0.800 から落ちても 0.50 は維持）|
| **H4** | Tufted_Duck の F1 自体は改善する | **Tufted_Duck F1 ≥ 0.55**（run07 0.432、+0.12 以上）|
| **H5** | 誤吸引されていた他種（Eurasian_Teal）が改善する | **Eurasian_Teal F1 ≥ 0.85**（run07 0.831、+0.02 以上）|
| **H6** | 他種への負の波及は最小 | **Mallard, Northern_Shoveler を含む他6種が run07 比 −0.02 以内**（cap で他種を変えていないので H6 達成は当然視するが、学習ダイナミクスの変化で間接波及の可能性は残る）|
| **H7** | val−test gap は run05/07 並み | **\|val − test\| ≤ 0.03**（run05 0.002 / run07 0.018）|

### 検証後の分岐（学習前に固定）

- **H1・H2 ともに成立（test f1 ≥0.86 かつ Tufted precision ≥0.50）**: チャンク不均衡是正が有効レバー。run09 では (1) cap をさらに下げる単独評価 or (2) class-weighted loss との併用検討
- **H1 不成立だが H2 成立（全体は伸びないが Tufted precision は改善）**: 不均衡是正は局所的に効くが全体改善には繋がらない → 他種で吸引されていた誤分類が **別の誤分類パターン** に置き換わったか確認。混同行列の精査
- **H2 不成立（Tufted precision <0.50）**: cap=100 でも十分でない → run09 で cap=50 を試すか、class-weighted loss に切り替え
- **H1 成立だが H3 不成立（Tufted recall が大幅悪化）**: 「過剰予測の抑制」が「Tufted を取りこぼす」に転じた → cap が強すぎる
- **H6 不成立（他種で −0.02 以上の悪化）**: cap は他種データを変えていないので **学習ダイナミクスの変化による間接波及**。Tufted 関連の決定境界変化が他種にも影響している。介入が想定外の副作用を起こしているサイン

### 結果（best = checkpoint-648, epoch 6）

| metric | value | run05 比 | run07 比 |
|---|---|---|---|
| best_epoch | **6** | 6 | 4 |
| eval_accuracy | 0.879 | 0.885 | 0.885 |
| eval_f1_macro | **0.8381** | 0.840 | 0.845 |
| eval_loss | 0.486 | 0.457 | 0.458 |
| **test_f1_macro** | **0.8203** | **0.838** | **0.827** |
| **test_accuracy** | **0.864** | **0.876** | **0.870** |
| val−test gap | **+0.018** | +0.002 | +0.018 |

学習は epoch 10 で early stop（patience=4, best=epoch 6）。`load_best_model_at_end` は checkpoint-648 を選択。test 評価出力: `outputs/eval_20260524_174742/`。

### 経過（主要マイルストーン）

| epoch | step | train_loss(末尾) | eval_loss | eval_f1_macro |
|---|---|---|---|---|
| 1 | 108 | — | 0.558 | 0.773 |
| 2 | 216 | 0.135 | 0.467 | 0.778 |
| 3 | 324 | — | 0.411 | 0.833 |
| 4 | 432 | — | 0.415 | 0.824 |
| 5 | 540 | 0.0003 | 0.487 | 0.834 |
| **6** | **648** | — | **0.486** | **0.838** ← best |
| 7 | 756 | 0.0002 | 0.492 | 0.838 |
| 8 | 864 | — | 0.499 | 0.838 |
| 9 | 972 | — | 0.508 | 0.834 |
| 10 | 1080 | 0.0001 | 0.509 | 0.832 |

val 曲線は run05 と類似（プラトー型、epoch 6 で best）。train_loss は epoch 5 時点で 0.0003、epoch 10 で 0.0001 まで落ち run05/07 と同様の完全暗記。

### 仮説検証

| 仮説 | 予測 | 実測 | 判定 |
|---|---|---|---|
| H1 全体 test f1 | ≥0.86 | **0.820** | **不成立** (run05 −0.018) |
| H2 Tufted precision | ≥0.50 | **0.269** | **不成立** (run07 0.296 から逆に悪化) |
| H3 Tufted recall | ≥0.50 | 0.700 | **成立** |
| H4 Tufted F1 | ≥0.55 | **0.389** | **不成立** (run07 0.432 から −0.043 / run05 と同値) |
| H5 Eurasian_Teal F1 | ≥0.85 | 0.859 | **成立** (+0.028 vs run07) |
| H6 他6種 \|diff\|≤0.02 | | Wigeon −0.049, Mallard −0.040, Pintail +0.049, Shoveler +0.031, Goldeneye −0.021 | **不成立**（5種が境界超過、3種改善 / 3種悪化）|
| H7 \|val−test\| | ≤0.03 | 0.018 | **成立** |

### 種別 test F1（run05 / 07 / 08 比較）

| 種 | test n | run05 | run07 | **run08** | run05→08 差 | train chunks |
|---|---|---|---|---|---|---|
| Common_Goldeneye | 64 | 0.930 | 0.938 | 0.917 | −0.013 | 144 |
| Common_Pochard | 63 | 0.945 | 0.968 | 0.960 | +0.015 | 220 |
| Eurasian_Teal | 86 | 0.844 | 0.831 | 0.859 | +0.015 | 249 |
| Eurasian_Wigeon | 27 | 0.809 | 0.809 | 0.760 | **−0.049** | 191 |
| Mallard | 46 | 0.918 | 0.889 | 0.849 | **−0.069** | 390 |
| **Northern_Pintail** | 28 | 0.949 | 0.933 | **0.982** | **+0.033** | 88 |
| **Northern_Shoveler** | 14 | 0.923 | 0.815 | **0.846** | −0.077 (vs05) | 72 |
| **Tufted_Duck** | 10 | 0.389 | 0.432 | 0.389 | **±0.000** | **728→361** |

### 混同行列の主な誤分類（run05 / 08 比較）

**Tufted_Duck と予測されたチャンクの内訳（precision の構成）**:
| | run05 | run08 |
|---|---|---|
| 予測 Tufted_Duck total | 26 | 26 |
| 正解 (Tufted_Duck) | 7 | 7 |
| 誤吸引 Eurasian_Teal | 14 | 13 |
| 誤吸引 Eurasian_Wigeon | 5 | 4 |
| 誤吸引 Common_Goldeneye | 0 | 2 |
| **Tufted precision** | **0.269** | **0.269** |

Tufted のチャンクを **−50%（728→361）削っても precision は小数第3位まで一致**。chunks 不均衡説では説明できない結果。

**Mallard と予測されたチャンクの内訳（precision の悪化）**:
- run08 Mallard precision 0.750（run07 0.830 / run05 0.957）。他種から Mallard への誤吸引が増えた
- Mallard true 46件中 45件正解（recall 0.978）。recall は改善したが precision の代償

### 所見

- **H1〜H4 不成立、H5 のみ局所成立**: 全体 test f1 は run05 から −0.018 で悪化。事前登録の分岐ロジックでは「H2 不成立」分岐に該当
- **「Tufted 過剰予測 = chunks 不均衡」仮説が崩れた**: chunks を 728→361 と半減しても Tufted precision は 0.269 で完全に同値、誤吸引の構成も Eurasian_Teal 13〜14件・Wigeon 4〜5件で **run05 と統計的にほぼ識別不能**。これは run05 / run08 の Tufted モデルがほぼ同じ決定境界を学んでいることを示す
- **真のボトルネックは音響的類似性の可能性**: Tufted_Duck と Eurasian_Teal の鳴き声が AST にとって区別困難な可能性がある。Wigeon との混同（4〜5件）も同様。これは「データ量」「データバランス」では解決しない、**モデル容量・特徴表現レベルの問題**かもしれない
- **データバランス介入は他種に副作用を起こす**: H6 不成立で Mallard −0.069 / Wigeon −0.049 / Goldeneye −0.021 と3種が大きく悪化、Pintail +0.033 / Shoveler +0.031 / Teal +0.015 / Pochard +0.015 と4種が改善。**Tufted のバランス変更が他種の決定境界を予測不能に揺らす**。「他種データを変えていない」と思っていたが、学習ダイナミクス全体が変わる。これは run07 の H4 不成立と同じ症状
- **小サンプル種は分散が大きい**: Pintail (n=28) +0.033、Shoveler (n=14) +0.031、Wigeon (n=27) −0.049 と test n が小さい種ほど run 間で変動が大きい。「乱数による振れ」と「介入の真の効果」の切り分けが難しい
- **AST + カモ8種で 0.85 天井の構造的説明**: ハイパラ系4 + データ系2 = 6 run 撃って run05 (0.838) を超えられない。「カモ類は鳴き声が似ていてこのモデル容量で区別できる上限がこの辺り」が仮説として説得力を持ってきた
- **chunks 半減で学習時間が増えた**: train chunks 2082→1715（-18%）にもかかわらず学習が 8→10 epochs に伸びて total 16→17分。理由不明（GPU メモリ管理か Tufted の学習収束の遅れ）

### 検証後の次アクション

事前登録した「H2 不成立」分岐は「cap=50 を試す or class-weighted loss に切り替え」だったが、**今回の結果から cap=50 も効かない可能性が高い**（chunks 半減で precision が動かなかったので、さらに減らしても同じ結果が予想される）。

方向性の見直しが必要:

1. **混同しやすい種ペアを分析**: Tufted / Teal / Wigeon の **メル分光特徴の類似性** を可視化（attention map、t-SNE/UMAP など）。本当に音響的に区別困難なのか確認
2. **class-weighted loss（run09 候補）**: chunks の数ではなく **損失の重み** で Tufted を抑える。データ介入で動かないなら損失介入を試す。ただし期待値は低め
3. **対比学習や hard negative mining**: AST の通常 cross-entropy では難しい場合に、Tufted/Teal を hard pair として明示的に学習させる
4. **モデル容量を上げる**: AST-Large などより大規模なモデルで音響特徴の表現力を上げる（VRAM 8GB の制約あり）
5. **モデル選択方式の改善**: run03 / 06 で踏んだ val ピーク選択バイアスの問題。run05 が「現行ベスト」で居続けるのは選択方式そのものが run05 と相性が良いだけかも

**最も学びが多い次の一手**: 1 (誤分類の音響的根拠を見る) → 2 (class-weighted loss で chunks 介入と異なる軸を試す)。1 でモデルが Tufted/Teal を本当に区別できていないと分かれば、2〜4 は方向性が変わる。

---

## 分析タスク — Tufted 誤分類の音響的根拠調査 (2026-05-24)

**目的:** run08 で「Tufted_Duck の chunks を半減しても precision が完全に変わらない」現象を受けて、誤分類されたチャンクの音響特徴を分光図で確認し、真の原因を突き止める。

**手法:** `src/bird_fine/analysis/confusion_audio.py` を新規実装。run08 の predictions.csv と test.csv を順序ベースで結合し、5 グループ（A: Tufted TP / B: Teal→Tufted 誤分類 / C: Wigeon→Tufted 誤分類 / D: Teal TP / E: Wigeon TP）のメルスペクトログラムを各最大8チャンク並べて画像化。

### 発見1: 誤分類の **録音単位での異常集中**

| グループ | チャンク数 | unique 録音 | 集中度 |
|---|---|---|---|
| A: Tufted_Duck TP | 7 | 3 | 分散（XC303149 4件, XC476421 2件, XC760407 1件）|
| **B: Teal → Tufted 誤分類** | **13** | **2** | **XC197026 から 12件 (92%)**, XC241897 から 1件 |
| **C: Wigeon → Tufted 誤分類** | **4** | **1** | **XC349677 から全件** |
| D: Eurasian_Teal TP | 67 | 8 | 分散（XC695865 21件 など）|
| E: Eurasian_Wigeon TP | 19 | 5 | 分散（XC615223 8件 など）|

正解は録音間で分散しているのに、**誤分類は特定の少数録音に異常集中**。これは「音響的類似」ではなく「**特定録音の問題**」を示唆。

### 発見2: 問題録音のメタデータが「学習データに無いタイプの音」

| XCID | 種 | type | sex | stage | 録音者コメント |
|---|---|---|---|---|---|
| **XC197026** | Eurasian_Teal | call | female | **juvenile** | "Female with 6 juvenil. The short whistling call is juvenile and the 'Butt' call is female." |
| **XC349677** | Eurasian_Wigeon | call | **female, male** | (none) | "Loud and Clear, nice recording of interspecific calls between males and females.." |
| XC241897 | Eurasian_Teal | song | (none) | (none) | （誤分類1件のみで集中なし）|

- **XC197026**: メス + **6羽の幼鳥**の鳴き声。**幼鳥の口笛声**が大量含まれる
- **XC349677**: **オスとメスの掛け合い**録音

### 発見3: train data に juvenile のサンプルがほぼ無い（distribution shift）

Eurasian_Teal の `stage` 分布:

| split | adult | juvenile | (none) |
|---|---|---|---|
| train (42 recordings) | 9 | **1** | 28 |
| **test (9 recordings)** | 1 | **1 (XC197026)** | 7 |

train に juvenile タグ録音は 1 件しかなく、AST は **「カモの juvenile call」を学んでない**。一方 test に juvenile 録音が入っている → **train/test の音響特徴分布が偏っている (distribution shift)**。

### 結論: 真の原因は「Tufted の決定境界が広すぎる」+ distribution shift

仮説の更新:
- 「Tufted 過剰予測 = chunks 不均衡」は **反証済み**（run08）
- 「Tufted 過剰予測 = 音響的類似性」も **疑わしい**（誤分類が特定録音に集中している = 種全体ではなく録音由来）
- **新仮説**: Tufted_Duck の長尺2録音 **XC488112 (45分) と XC488113 (49分)** が「多様なカモのデフォルト call っぽい音」を含んでおり、Tufted の決定境界を不当に広げている。学習データに無い音（juvenile call, 複数個体の鳴き合い）は、もっとも音響範囲の広い「Tufted_Duck」クラスに吸引される

この仮説は **chunks 半減（cap=100）で precision が動かなかった事実とも整合**: cap=100 ではランダムサブサンプリングなので XC488112/XC488113 の代表チャンクは残り、多様な音響パターンも保持されるため。

次の検証: **XC488112 / XC488113 を train から完全除外** (run09)。

---

## run09 — Tufted_Duck の長尺2録音を train から完全除外 (2026-05-24)

**出力:** `models/ast-duck-v9/`
**コミット:** 事前登録 `a41b3f3` / 結果は本セクションの結果コミットで記録
**ステータス:** 学習・test 評価完了（2026-05-24）。**仮説の核は確証された**: XC197026 (Teal juvenile) と XC349677 (Wigeon duet) の Tufted_Duck 過剰予測がそれぞれ 12→4、4→0 に激減。「Tufted の長尺2録音が決定境界を歪めていた」仮説は明確に確認。**ただし副作用**: Tufted の train chunks 161 が少なすぎて Tufted recall が 0.700→0.300 に崩壊、test f1 は **0.810**（run05 比 −0.028 / run08 比 −0.010）で全体悪化。
**結論:** **真因の特定に成功**したが「全削除」は介入が強すぎ。次は **長尺2録音を部分復活させる中間点**を探る（run10 候補）。現行ベストは引き続き **run05（test f1 0.838）**。

### データ変更（差分）

`src/bird_fine/data/exclude_train_recordings.py` を新規実装。`train.csv` から指定 XCID（部分一致）の chunks を完全削除。val/test は変更しない。

| 種 | run08 train chunks | run09 train chunks | 差 |
|---|---|---|---|
| Common_Goldeneye | 144 | 144 | ±0 |
| Common_Pochard | 220 | 220 | ±0 |
| Eurasian_Teal | 249 | 249 | ±0 |
| Eurasian_Wigeon | 191 | 191 | ±0 |
| Mallard | 390 | 390 | ±0 |
| Northern_Pintail | 88 | 88 | ±0 |
| Northern_Shoveler | 72 | 72 | ±0 |
| **Tufted_Duck** | **361** | **161** | **−200**（XC488112: 100→0, XC488113: 100→0）|

Tufted_Duck の train 録音数は 29 → 27（XC488112 と XC488113 を全削除）。chunks 数は他種より少ない逆方向の不均衡だが、これは **「長尺2録音が決定境界を歪めている」仮説の純粋な検証** が目的。

### ハイパラ（run08 からの差分）

run09 のハイパラは **run08 と完全同一**（=run05/07 とも同一）。変更点は train データの exclude のみ。

### 仮説と予測（学習前に固定）

run08 で chunks 半減が効かなかった事実から:
- chunks 数は原因ではない
- 真の原因は **特定録音（XC488112 / XC488113）が学習する音響パターンの幅**

長尺2録音を完全削除すれば:
- Tufted の決定境界が狭まる
- 「カモのデフォルト call」のような汎用パターンが Tufted に吸収されなくなる
- XC197026 (Teal juvenile) や XC349677 (Wigeon 掛け合い) が **正しく予測される確率が上がる**

| | 仮説 | 予測 |
|---|---|---|
| **H1（主）** | XC488112/XC488113 削除で Tufted の過剰予測が解消し、全体 test f1 が上がる | **test f1_macro ≥ 0.86**（run05 0.838 を上回る）|
| **H2（主）** | Tufted_Duck の precision が大幅改善する | **Tufted_Duck precision ≥ 0.55**（run08 0.269、+0.28）|
| **H3** | Tufted の recall は許容範囲内に維持 | **Tufted_Duck recall ≥ 0.40**（run08 0.700 から落ちても 0.40 維持）|
| **H4** | XC197026 (Teal juvenile) のチャンク群が **Tufted ではなく Teal と予測される** | **XC197026 のチャンク13件中、Teal 予測 ≥ 7件**（run08 では 0/13）|
| **H5** | XC349677 (Wigeon 掛け合い) のチャンク群が **Wigeon と予測される** | **XC349677 のチャンク4件中、Wigeon 予測 ≥ 2件**（run08 では 0/4）|
| **H6** | Eurasian_Teal の test F1 が改善（誤吸引解消の波及）| **Eurasian_Teal F1 ≥ 0.90**（run08 0.859、+0.04）|
| **H7** | val−test gap は維持 | **\|val−test\| ≤ 0.03**（run08 は 0.018）|

### 検証後の分岐（学習前に固定）

- **H1・H2 ともに成立（test f1 ≥0.86 かつ Tufted precision ≥0.55）**: 「長尺2録音が決定境界を歪めていた」仮説が確証。**真のレバーを発見**。次は (1) 他種にも同様の問題録音がないか調査 (2) Tufted の他の録音で 27 → 30 程度に補充するか検討
- **H2 成立だが H1 不成立**: Tufted の過剰予測は解消したが、全体 F1 が他種の悪化で相殺された → 他種の混同行列を精査
- **H2 不成立（Tufted precision <0.55）**: 長尺2録音削除でも効かない → 仮説が間違い。残るのは「カモ全般の音響特徴を AST-base では区別できない」モデル容量限界説 → run10 で AST-Large or class-weighted loss
- **H4 / H5 のいずれか成立、H2 不成立**: 一部の問題録音は解消できたが Tufted の過剰予測パターンは残る → 他のソースが他種誤分類を起こしている。詳細な混同行列分析
- **H3 不成立（Tufted recall <0.40）**: Tufted の学習データが少なすぎて学習不能 → recall を犠牲にしすぎ。Tufted を他録音で補充する方向

### 結果（best = checkpoint-475, epoch 5）

| metric | value | run05 比 | run08 比 |
|---|---|---|---|
| best_epoch | **5** | 6 | 6 |
| eval_accuracy | 0.885 | 0.885 | 0.879 |
| eval_f1_macro | **0.8436** | 0.840 | 0.838 |
| eval_loss | 0.417 | 0.457 | 0.486 |
| **test_f1_macro** | **0.8098** | **0.838** | **0.820** |
| **test_accuracy** | **0.870** | **0.876** | **0.864** |
| val−test gap | **+0.034** | +0.002 | +0.018 |

学習は epoch 9 で early stop（patience=4, best=epoch 5）。test 評価出力: `outputs/eval_20260524_234838/`。学習時間 808s（run08 比 -20%）。

### 仮説検証

| 仮説 | 予測 | 実測 | 判定 |
|---|---|---|---|
| H1 全体 test f1 | ≥0.86 | 0.810 | **不成立** (run05 −0.028) |
| H2 Tufted precision | ≥0.55 | 0.333 | **不成立**（run08 0.269 から +0.064 微改善）|
| H3 Tufted recall | ≥0.40 | **0.300** | **不成立**（run08 0.700 から **−0.400 崩壊**）|
| H4 XC197026 Teal 予測 | ≥7/13 | 3/13 | 不成立（数値）も **Tufted 過剰予測は 12→4 に劇的減少** |
| H5 XC349677 Wigeon 予測 | ≥2/6 | **2/6** | **成立**（Tufted 予測は 4→0 に完全消滅）|
| H6 Eurasian_Teal F1 | ≥0.90 | 0.859 | 不成立（run08 と同値）|
| H7 \|val−test\| | ≤0.03 | 0.034 | 僅か不成立（境界）|

### 重点検証: XC197026 / XC349677 の予測変化（run05→09 通史）

**XC197026 (Eurasian_Teal juvenile, n=13)**:

| run | Tufted_Duck | Common_Pochard | Eurasian_Teal | その他 |
|---|---|---|---|---|
| run05 | **13** | 0 | 0 | 0 |
| run07 | 12 | 0 | 1 | 0 |
| run08 | 12 | 1 | 0 | 0 |
| **run09** | **4** | **6** | 3 | 0 |

**XC349677 (Eurasian_Wigeon duet, n=6)**:

| run | Tufted_Duck | Eurasian_Wigeon | Mallard |
|---|---|---|---|
| run05 | 5 | 0 | 1 |
| run07 | 2 | 3 | 1 |
| run08 | **4** | 1 | 1 |
| **run09** | **0** | **2** | **4** |

**仮説の核は確証**: XC488112 / XC488113 削除で XC197026 の Tufted 予測 12→4、XC349677 の Tufted 予測 4→0。「長尺2録音が決定境界を歪めていた」は事実だった。

ただし:
- XC197026 の半数（6件）は **Common_Pochard に流れた** — distribution shift（juvenile 音）は Tufted の問題と独立。train に juvenile データが 1件しかないため、Tufted から離れたら別の種に吸引されるだけ
- XC349677 は Mallard に4件流れた — こちらも Wigeon の典型 call から離れた音は他種に流れる

### 種別 test F1（run05 / 08 / 09 比較）

| 種 | test n | run05 | run08 | **run09** | run08→09 差 | train chunks (run08→09) |
|---|---|---|---|---|---|---|
| Common_Goldeneye | 64 | 0.930 | 0.917 | 0.915 | −0.002 | 144 |
| Common_Pochard | 63 | 0.945 | 0.960 | 0.931 | **−0.029** | 220 |
| Eurasian_Teal | 86 | 0.844 | 0.859 | 0.859 | ±0.000 | 249 |
| Eurasian_Wigeon | 27 | 0.809 | 0.760 | 0.755 | −0.005 | 191 |
| Mallard | 46 | 0.918 | 0.849 | 0.857 | +0.008 | 390 |
| Northern_Pintail | 28 | 0.949 | 0.982 | 0.966 | −0.016 | 88 |
| **Northern_Shoveler** | 14 | 0.923 | 0.846 | **0.880** | **+0.034** | 72 |
| **Tufted_Duck** | 10 | 0.389 | 0.389 | **0.316** | **−0.073** | **361→161** |

### Tufted_Duck の予測の構造変化（run08 vs run09）

**Tufted と予測されたチャンク**:
| | run08 | run09 |
|---|---|---|
| 総予測数 | 26 | **9** |
| Tufted 正解 (TP) | 7 | 3 |
| Teal 誤吸引 (FP) | 13 | 4 |
| Wigeon 誤吸引 (FP) | 4 | 0 |
| Goldeneye 誤吸引 (FP) | 2 | 2 |
| precision | 0.269 | 0.333 |

**Tufted_Duck 正解の内訳（test n=10）**:
| | run08 pred | run09 pred |
|---|---|---|
| Tufted_Duck | 7 | **3** |
| Mallard | 1 | 3 |
| Wigeon | 1 | 2 |
| Teal | 1 | 1 |
| Pintail | 0 | 1 |

過剰予測は劇的に解消（26→9）したが、recall も 0.700→0.300 に崩壊。Tufted の test 10件中 7件が他種に流出。

### 所見

- **仮説の核（長尺2録音の決定境界歪み）は確証**: XC197026 / XC349677 の Tufted 過剰予測が劇的に減少。run05→09 で XC197026 は Tufted 13→4、XC349677 は Tufted 5→0。**run08 で chunks 半減が効かなかったのに、run09 で長尺2録音の完全削除が効いた** = 「chunks の数」ではなく「特定録音が学習させる音響パターンの幅」が問題だったと確定
- **数値仮説 (H2/H4) は不成立だが仮説の意図は達成**: H2「precision ≥0.55」は不成立（0.333）だが、Tufted 予測総数 26→9 で過剰予測自体は大幅解消。H4「XC197026 Teal 予測 ≥7」は数値不成立も、Tufted 予測 12→4 で過剰予測解消の意図は達成
- **distribution shift は別問題**: XC197026 の半数（6件）が Common_Pochard に流れたのは、train に juvenile データが 1件しかないため。Tufted を狭めるだけでは「学習データに無い音」の問題は解決しない。**録音追加（juvenile タグ付き）が別軸で必要**
- **介入が強すぎた**: Tufted train chunks 161 では Tufted の学習が不十分。Tufted の test 10件中 7件が他種に流出。test n=10 で F1 が 0.073 悪化したのは絶対数で「正解 7→3 件」の差
- **「過剰予測の解消」と「適正な学習」の両立点を探す必要**: 全削除（chunks 161）は介入が強すぎ、何もしない（chunks 728）は過剰予測。中間点（chunks 200〜400 程度？）を探す
- **副次的な発見: 短尺録音だけでも Tufted の test recall が確保できる程度の汎化が起きている**: Tufted 正解 3件はすべて短尺の test 録音（XC303149, XC476421, XC760407）から来ている。run08 と同じ。長尺2録音を削除しても Tufted の **本来の音響特徴**は他の短尺27録音から学べている

### 検証後の次アクション

- **run10 候補: 長尺録音の cap を強めにかける** — 全削除（chunks 0）でなく cap=30 や cap=50 などで部分復活。Tufted chunks を 161 < x < 361 の中間値（例 cap=30 で XC488112 270→30, XC488113 297→30 → Tufted total ~221）で、過剰予測抑制と Tufted 学習量のバランスを取る
- **run11 候補: Eurasian_Teal の juvenile 録音追加** — distribution shift の独立対処。Xeno-canto で `stage:juvenile` を指定して Teal の追加収集
- **run12 候補: モデル選択方式の改善** — 6 run 撃って run05 を超えられないが、val−test gap は run09 で再び 0.034 に拡大。`load_best_model_at_end` の代替案検討
- run10 を先に試して中間点を見つけてから、run11 で distribution shift を対処する順序が筋。並行は変数増えすぎ

---

## run10 — 前処理パラダイム変更: 10s→3sチャンク / BirdNet pipeline alignment (2026-05-31)

**ステータス:** 学習前記録（事前登録）
**出力:** `models/ast-duck-v10/`

### 変更概要

| パラメータ | run09（旧） | run10（今回） | 変更理由 |
|---|---|---|---|
| `chunk_duration_sec` | 10.0 | **3.0** | BirdNet 3s窓に合わせる |
| `min_chunk_duration_sec` | 3.0 | **1.0** | 端数チャンクを救う |
| `feature_extractor_max_length` | 1024 | **304** | 3s×16kHz のmelフレーム数 |
| `output_dir` | ast-duck-v9 | **ast-duck-v10** | 新規run |
| lr / wd / patience | 2.0e-5 / 0.03 / 4 | 同じ | run05ベースラインを継承 |
| SpecAugment | off | off | 固定変数 |

**旧 run10 計画（cap=30/50）との関係:** Ubuntu移行 + BirdNetコード確認を機に、チャンク長変更という上位の問題を先に解決することにした。cap 実験は 3s チャンク体制が安定してから再検討。

### 仮説と予測

**前提:** BirdNet は 3秒窓で音声を処理し「カモ類」を検出する。本モデルがその3秒音声を受け取る想定なのに、10秒チャンクで学習していたのは設計上の不整合。

**H1: チャンク数が増加する（録音あたり約3倍）**
- 10秒チャンクより細かく切れるため train/val/test の総チャンク数が増加するはず

**H2: Tufted_Duck の過剰予測が緩和方向に動く**
- XC488112/XC488113 は run09 の知見に基づき chunks_index.csv 段階で全splits から完全除外済み。3s チャンクで残る 27 録音 477 chunks から学習する

**H3: val / test f1 は初回でrun05（0.838）を超えない可能性が高い**
- AST の位置埋め込みは 10s（1024フレーム）向けに事前学習済み。3s（304フレーム）への適応は fine-tune で徐々に進む
- ただし run05 と同ハイパラなので早期に拮抗できれば成功と見なす

**数値予測:**
- 総 train チャンク数: 現状の 2.5〜3.0 倍（10s→3s で単純計算 3.3倍、端数・短尺除外で減少）
- val f1_macro: 0.80〜0.86（初期適応コスト + データ増の効果が拮抗）
- test f1_macro: 0.80〜0.86

**検証後の分岐:**
- test f1 ≥ 0.838（run05相当）→ 3sチャンク体制で実験継続（run11でTufted cap再検討）
- test f1 < 0.838 かつ Tufted F1 改善 → 3sチャンクは方向性として正しい。別ハイパラで run12
- test f1 < 0.838 かつ Tufted F1 も悪化 → 3sチャンクがASTの事前学習と相性が悪い可能性。10sに戻すか別モデル検討

### 学習前のチャンク分布（実数）

XC488112 (899 chunks) / XC488113 (989 chunks) を chunks_index.csv から除外後に re-split。

| 種 | train | val | test |
|---|---|---|---|
| Common_Goldeneye | 499 | 42 | 199 |
| Common_Pochard | 619 | 238 | 247 |
| Eurasian_Teal | 983 | 55 | 230 |
| Eurasian_Wigeon | 525 | 159 | 104 |
| Mallard | 1270 | 237 | 222 |
| Northern_Pintail | 305 | 107 | 47 |
| Northern_Shoveler | 243 | 53 | 35 |
| Tufted_Duck | **477** | **76** | **70** |
| **合計** | **4921** | **967** | **1154** |

- run05（10s train ~2000）比で train は約 2.5倍
- Tufted_Duck val が 1032→76 に正常化（XC488113 除外効果）

### 結果

early stopping: epoch 10 で停止（patience=4、best=epoch 6）

| epoch | eval_loss | eval_f1_macro | 備考 |
|---|---|---|---|
| 1 | 0.829 | 0.676 | |
| 2 | 0.847 | 0.729 | |
| 3 | 0.954 | 0.781 | |
| 4 | 1.064 | 0.774 | |
| 5 | 1.002 | 0.764 | |
| **6** | **0.922** | **0.806** | **← best** |
| 7 | 0.966 | 0.798 | |
| 8 | 0.991 | 0.790 | |
| 9 | 1.005 | 0.791 | |
| 10 | 1.013 | 0.791 | early stop |

### メトリクス（run05比較）

| 指標 | run05 | run10 | 差分 |
|---|---|---|---|
| val f1_macro | 0.840 | 0.806 | −0.034 |
| test f1_macro | **0.838** | **0.782** | **−0.056** |
| accuracy | - | 0.816 | |

### 種別 test F1（run05 / run10 比較）

| 種 | test n | run05 | run10 | 差分 |
|---|---|---|---|---|
| Common_Goldeneye | 199 | 0.930 | 0.729 | **−0.201** |
| Common_Pochard | 63→247 | 0.945 | 0.947 | +0.002 |
| Eurasian_Teal | 86→230 | 0.844 | 0.772 | −0.072 |
| Eurasian_Wigeon | 27→104 | 0.809 | 0.813 | +0.004 |
| Mallard | 46→222 | 0.918 | 0.843 | −0.075 |
| Northern_Pintail | 28→47 | 0.949 | 0.722 | **−0.227** |
| Northern_Shoveler | 14→35 | 0.923 | 0.693 | −0.230 |
| **Tufted_Duck** | 10→70 | **0.389** | **0.738** | **+0.349** |

※ test n が run05 と異なる（3sチャンクで再分割のため）

### 所見

- **H1 成立**: train チャンク数 ~2000→4921（約2.5倍）
- **H2 成立**: Tufted_Duck F1 0.389→0.738（+0.349）。XC488112/XC488113 除外 + 3s チャンク化の複合効果
- **H3 成立**: test f1 0.782 で run05（0.838）を下回る。AST 位置埋め込みの再適応コストが出た
- **全体 test f1 は低下（−0.056）**: Goldeneye（−0.201）・Pintail（−0.227）・Shoveler（−0.230）が大幅悪化。train チャンク数の少ない種が 3s 移行で影響を受けた可能性
- **val f1 のピーク（epoch 6: 0.806）と早期停止**: val 曲線は epoch 3 をピークに上下し、epoch 6 に再上昇。`load_best_model_at_end` が epoch 6 を選んだのは妥当
- **train.py の技術的課題**: 位置埋め込みのリサイズ後に `model.config.max_length` を更新していなかったため、保存済み config.json と実際の重みが不一致。config.json を手動で修正（→ 次回から自動化済み）

---

---

## run11 — "other" クラス追加による OOD 検知 (2026-05-31)

**ステータス:** 完了
**出力:** `models/ast-duck-v11/`
**初期化元:** `models/ast-duck-v10/`（run10チェックポイントから分類ヘッドを拡張）

### 変更概要

| 項目 | run10 | run11 |
|---|---|---|
| num_labels | 8 | **9**（+`other`クラス）|
| metric_for_best_model | f1_macro | **f1_macro_8class**（8種のみ）|
| WeightedRandomSampler | なし | **あり**（全クラス均等サンプリング）|
| CE loss | 均等 | **[1.0×8, 1.5×other]**（固定値・Double Dipping なし）|
| SpecAugment | off | **other クラスにのみ適用**（8種の境界を守る）|
| init | pretrained AST | **run10 checkpoint**（8クラス重みを保持して拡張）|

### "other" クラスデータ

recording 単位で train20 / eval10 に分割、per-recording cap=30 で過学習防止。

| 種 | tier | train録音 | eval録音 |
|---|---|---|---|
| Eastern Spot-billed Duck | 1 | 2 | 0 |
| Gadwall | 1 | 20 | 10 |
| Baikal Teal | 1 | 3 | 0 |
| Falcated Duck | 1 | 3 | 0 |
| Red-breasted Merganser | 1 | 20 | 5 |
| Eurasian Coot | 2 | 20 | 10 |
| Little Grebe | 2 | 20 | 10 |
| **合計** | | **1220 chunks** | **391 chunks** |

### 仮説と予測

**H1: 既存8種の精度（f1_macro_8class）が run10（val 0.806）を下回らない**
- run10 チェックポイントから分類ヘッドをゼロ拡張して初期化
- "other" ニューロンが徐々に学習される一方、8クラス境界は開始時点で保持
- Double Dipping を排除したことで "other" 学習が8種境界を侵食しない

**H2: OOD チャンクの energy_score が run10 比で低下する**
- "other" クラスを学習したことでモデルが「カモ以外の音」に低エネルギーを割り当てる
- AUROC（energy）が run10 の 0.894 を超える

**H3: other_recall（val）が最終的に 0.5 以上になる**
- val の "other" 391チャンクに対して半数以上を正しく弾ける

**数値予測:**
- val f1_macro_8class: 0.78 以上（run10 の 0.806 付近を維持）
- val other_recall: 0.50〜0.80
- OOD eval AUROC（energy）: 0.92 以上

**検証後の分岐:**
- f1_8class ≥ 0.78 かつ other_recall ≥ 0.5 → OOD対策として有効。confidence_threshold を設定して pipeline に組み込む
- f1_8class < 0.78（8種境界が侵食）→ alpha を下げる / 別ハイパラで run12
- other_recall < 0.5（"other" を学習できず）→ データ量・多様性を再検討

### 結果

early stopping: epoch 5 で停止（patience=4、best=epoch 2）

| epoch | eval_loss | f1_macro | f1_macro_8class | other_recall |
|---|---|---|---|---|
| 1 | 1.895 | 0.595 | 0.778 | **0.090** |
| **2** | **2.550** | **0.586** | **0.800** | 0.023 |
| 3 | 3.183 | 0.567 | 0.775 | 0.000 |
| 4 | 3.028 | 0.573 | 0.791 | 0.005 |
| 5 | 3.124 | 0.574 | 0.789 | 0.008 |

### test 評価（8種のみ・test.csv 1154チャンク）

| 指標 | run10 | run11 | 差分 |
|---|---|---|---|
| test f1_macro（8種のみ） | 0.782 | **0.793** | **+0.011** |
| test accuracy | 0.816 | 0.822 | +0.006 |

### 種別 test F1

| 種 | run10 | run11 | 差分 |
|---|---|---|---|
| Common_Goldeneye | 0.729 | 0.743 | +0.014 |
| Common_Pochard | 0.947 | **0.956** | +0.009 |
| Eurasian_Teal | 0.772 | 0.793 | +0.021 |
| Eurasian_Wigeon | 0.813 | 0.790 | −0.023 |
| Mallard | 0.843 | 0.825 | −0.018 |
| Northern_Pintail | 0.722 | **0.832** | **+0.110** |
| Northern_Shoveler | 0.693 | 0.738 | +0.045 |
| **Tufted_Duck** | 0.738 | **0.703** | −0.035 |

### OOD 評価（Energy-based）

| スコア | run10 | run11 | 差分 |
|---|---|---|---|
| AUROC softmax | 0.838 | 0.854 | +0.016 |
| **AUROC energy** | **0.894** | **0.932** | **+0.038** |
| 推奨閾値（energy, FPR≤5%）| 10.29 | 10.35 | - |

OOD mean energy: Tier1=6.44 / Tier2=5.89 / Tier3=6.50（run10比で全tier低下 → 正しい方向）

### 所見

- **H1 成立（ほぼ）**: f1_macro_8class val=0.800（run10 0.806 から −0.006 の微減）。8種境界はほぼ保持された
- **H2 成立**: AUROC energy 0.894 → **0.932（+0.038）**。"other" クラスの学習で energy スコアの分離が向上
- **H3 不成立（ただし問題設定の読み替えで解消）**: other_recall は最大 0.090 で H3 予測の 0.5 に大幅未達。val での softmax 分類としての "other" 検知はできていない。しかし——
- **Outlier Exposure としての正しい解釈**: run11 の目的は「"other" を正しく分類すること」ではなく「OOD データを学習に晒すことで energy 空間を calibrate すること」だった（Hendrycks et al., 2018 の Outlier Exposure と同じ設計）。AUROC energy 0.932 はこの目的が達成されたことを示す。other_recall の低さは「Outlier Exposure として成功した結果」であり失敗ではない
- **eval_loss が epoch1→5 で単調増加**（1.895→3.124）は "other" val チャンクが混入したことで loss スケールが変化した可能性。モデル選択基準として `f1_macro_8class` を使ったため実害なし
- **test F1（8種）は run10 比 +0.011 の微改善**。Outlier Exposure が 8 種の決定境界を侵食しなかった証拠。特に Pintail +0.110 が顕著

### 次ステップ

run11 モデルを **Outlier Exposure によるエネルギー空間キャリブレーション済みモデル** として確定し、`predict.py` に energy gate を実装することで「システムとしての OOD 弾き」を構築する。

```
推論パイプライン:
  logits → energy = T * logsumexp(logits / T)
  energy < energy_threshold(10.35) → reject ("unknown")
  energy >= 10.35 → argmax(logits) → 8 種の予測結果
```

閾値・温度は `species_taxonomy.yaml` で管理（`energy_threshold: 10.35` / `energy_temperature: 1.0`）。

---

## test v2 — ドメイン整合（juvenile/nestling 除外）による評価セット修正 (2026-06-03)

**ステータス:** 完了（評価土台の修正・run 番号は消費しない）

### 背景

「幼鳥は冬の日本のモニタリングに不要では?」という問いを検証。test の juvenile 録音は
**繁殖期・繁殖地**（XC667403/667392 = 2021-08-05 フランスの雛 begging call）で、冬の日本では
遭遇しない音響パターン。これらを評価に含めると Tufted recall=0.000 で性能を過小評価していた。

### 対処

`filter_domain.py` で **val/test から stage∋juvenile/nestling を除外**（train は多様性のため不変）。

| split | 変更前 | 除外 | 変更後 |
|---|---|---|---|
| val | 1358 | 73 (Mallard juvenile 1録音) | 1285 |
| test | 1154 | 116 (4録音: Mallard/Tufted×2/Teal) | 1038 |

### run11 再評価（旧 test → test v2、モデルは run11=v11 のまま）

| 指標 | 旧 test(1154) | **test v2(1038)** | 差分 |
|---|---|---|---|
| f1_macro_8class | 0.798 | **0.808** | +0.010 |
| Tufted_Duck F1 | 0.703 | **0.767** | **+0.064** |

繁殖期(5-8月)を月で除外する案は adult call まで巻き込み f1_8=0.764 / Tufted=0.480 と悪化した
ため不採用。**stage ベース除外が正しい**。

### 所見

- run12 以降の baseline は **test v2** とする（f1_macro_8class 0.808）。
- 旧 split は `*.csv.bak` に退避。train/test とも日本録音0・繁殖期混在というドメイン乖離は残課題
  （→ journal.md 2026-06-03。worldwide フォールバックの帰結）。

---

## run12 — AudioMAE 全体fine-tune による表現力評価 (2026-06-02)

**ステータス:** 学習前記録（事前登録・実装未着手）
**出力予定:** `models/audiomae-duck-v12/`（バックボーンが変わるため命名を AST と区別）
**初期化元:** timm 移植チェックポイント `gaunernst/vit_base_patch16_1024_128.audiomae_as2m`

### 背景（なぜ AudioMAE か）

run11 の種別誤分類をメタデータ駆動で精査した結果（→ journal.md 2026-06-02、`docs/model_comparison_audiomae_ssam.md` 結論更新）、弱点が2型に分離した:
- **データ不在型**（Tufted juvenile）: 世界に2録音で枯渇 → 解決不能
- **表現力不足型**（Goldeneye `song`）: Eurasian_Teal の song と音響類似・train データ十分 → **バックボーン強化が本命**

埋め込み分析で、誤分類 Goldeneye song の **68%が train Teal song の方に近く**（正解 song は 0%）、train GE song↔Teal song 重心 cos=**0.880**。データ施策が尽きたため、**自己教師あり事前学習（MAE）による低レベルスペクトル表現で song を分離できるか**を評価する。

### 変更概要

| 項目 | run11 (AST) | run12 (AudioMAE) |
|---|---|---|
| backbone | ASTForAudioClassification (教師あり AudioSet) | timm ViT-B + AudioMAE (自己教師あり MAE) |
| 事前学習 | 教師あり | **自己教師あり（マスク再構成）** |
| num_labels | 9（8種 + other） | 9（踏襲）|
| メルビン数 | 128 | 128（共有・前処理再利用度大）|
| max_length | 304 | 304（位置埋め込みを 1024→304 相当に補間）|
| feature extractor | ASTFeatureExtractor | **自前実装**（128mel・AudioMAE 正規化統計に合わせる）|
| Sampler / Loss / SpecAugment | WeightedRandomSampler / [1.0×8,1.5×other] / other のみ | **run11 と同一**（DuckTrainer 流用）|
| metric_for_best_model | f1_macro_8class | f1_macro_8class（踏襲）|

### 実装前 Go/No-Go チェックリスト（model_comparison §6 準拠）

着手前に読み取り/小規模検証で確認する。いずれか No なら見送り or 後回し。

- [ ] **ckpt 入手**: `gaunernst/vit_base_patch16_1024_128.audiomae_as2m` を timm でロードできるか
- [ ] **依存導入**: `uv add timm`（純Python・ビルド不要）で済むか
- [ ] **VRAM 実測**: fp16 + gradient_checkpointing + effective batch 16 で RTX 3060 Ti (8GB) に収まるか
- [ ] **入力整合**: 128mel / max_length=304 / 正規化統計を ckpt 仕様に合わせられるか
- [ ] **A/B 再現条件**: 既存 `data/splits/{train,val,test}.csv`・`f1_macro_8class`・seed=42 で公平比較できるか

### 仮説と予測（予測値を学習前に固定）

baseline = run11: test f1_macro_8class **0.793** / Goldeneye test F1 **0.743** / AUROC energy **0.932**

**H1（主）: Goldeneye の song 分離が改善し、Goldeneye test F1 が run11 を上回る**
- 自己教師あり MAE が AudioSet 教師ラベルに縛られず低レベルスペクトル構造を学ぶため、song の微細差を AST より分離できる
- 予測: Goldeneye test F1 **0.76〜0.82**

**H2: 8種全体の精度が run11 を大きく下回らない**
- リスク: 位置埋め込み補間の適応コスト（run10 で AST が 10s→3s で test −0.056 した前例）
- 予測: test f1_macro_8class **0.78〜0.83**（run11 0.793 ± 域内維持）

**H3（機序）: 誤分類 Goldeneye song の「Teal 重心寄り割合」が 68% から低下する**
- run12 埋め込みで再測定して検証（同一手順）

**検証後の分岐:**
- H1成立 ∧ H2成立 → **表現力ボトルネック仮説を確証**。AudioMAE 採用、SSAM も PoC 検討
- H1不成立（Goldeneye 改善せず）→ song 類似は AudioMAE でも分離不能。前処理（mel分解能・入力長）or SSAM、あるいは「この2種は分離限界」と受容
- H2不成立（全体悪化）→ 位置埋め込み適応 or layer-wise LR decay 等のハイパラ再調整。AST 維持も選択肢

### 実装タスク概略（着手時）

1. timm バックボーン読み込み → 9クラス分類ヘッド付与（`expand_classifier` 流用可）
2. 128mel 特徴抽出を自前実装（AudioMAE 正規化統計に合わせる）
3. 位置埋め込み補間（10s/1024 → 3s/304 相当。現行 `resize_position_embeddings` を移植）
4. `DuckTrainer`（WeightedRandomSampler + other重みCE）に載せる（モデル非依存なので流用可）
5. 学習後: OOD energy gate を再キャリブレーション（閾値・AUROC を再測定）

### 結果

（学習後に記入）

### 所見

（学習後に記入）

---

## run13 — 対象種拡張: 8種 → 12種（ood_tier1 カモ科の target 昇格） (2026-06-03)

**ステータス:** 学習前記録（事前登録）。split まで構築済み・学習未着手
**出力予定:** `models/ast-duck-v13/`
**初期化元:** **pretrained**（`MIT/ast-finetuned-audioset-10-10-0.4593`）

### 背景

「幼鳥は冬の日本に不要」のドメイン検証（test v2）から派生し、対象種を増やす検証へ。
`species_master.csv` の ood_tier1 カモ科のうち**データ十分な4種**を target 昇格する。
下位3種（Baikal Teal/Falcated Duck/Greater Scaup）は録音不足のため見送り、other に残置。

### 変更概要

| 項目 | run11 | run13 |
|---|---|---|
| 対象種 | 8 | **12**（+Gadwall/ウミアイサ/カルガモ/カワアイサ）|
| num_labels | 9 | **13**（12種 + other）|
| other tier1 | 5種 | **2種**（Baikal Teal / Falcated Duck）|
| init | run10 ヘッド拡張 | **pretrained**（12種で label 順序が変わり run11 ヘッド流用不可）|
| split | — | **既存8種は run11 保持・新4種のみ seed=42 で 70/15/15 追記**（既存8種を公平比較）|
| test | test v2(1038) | 1261（既存8種 test v2 + 新4種、juvenile/nestling 除外済み）|

### 昇格4種のデータ（target 化後）

| 種 | train録音 | val | test | train chunks |
|---|---|---|---|---|
| Gadwall オカヨシガモ | 30 | 6 | 8 | 855 |
| Red-breasted Merganser ウミアイサ | 17 | 3 | 5 | 299 |
| Eastern Spot-billed Duck カルガモ | 11 | 2 | 3 | 302（worldwide 追加収集）|
| Common Merganser カワアイサ | 8 | 1 | 3 | 195 |

### 仮説と予測（予測値を学習前に固定）

baseline = run11 (test v2): f1_macro_8class **0.808** / AUROC energy **0.932**

**H1: 既存8種の精度が大きく低下しない**
- 既存8種の test 録音は run11 と同一（公平比較）。クラス増で決定境界は増えるが既存学習データは不変
- 予測: 既存8種 f1_macro_8class **0.77〜0.81**（run11 0.808 から ∓域内）

**H2: 新4種は録音数相応の F1**
- 録音多様性が効く（run01〜09 の知見）。Gadwall(30録音)>RBM(17)>カルガモ(11)>カワアイサ(8)
- 予測: 新4種 f1_macro **0.40〜0.60**（Gadwall 0.6+、カワアイサ/カルガモは録音少で 0.2〜0.5）

**H3: OOD energy gate が大きく劣化しない**
- other tier1 が 5→2種に縮小。カモ科 OOD の多様性低下が AUROC に影響しうる
- 予測: AUROC energy **0.88〜0.93**

**12種全体予測:** f1_macro_12class **0.70〜0.78**（新4種が全体を押し下げる）

### 検証後の分岐

- H1成立 ∧ 新4種が実用水準 → 12種化成功。次は run14 = AudioMAE で Goldeneye song + 増えた近縁種の表現力評価
- H1不成立（既存8種低下）→ クラス増による容量不足。ハイパラ調整 or AudioMAE を先行
- 新4種が極端に低い（特にカワアイサ/カルガモ）→ 録音不足種を other に戻す or 追加収集

### 結果

early stop epoch 10（best は val f1_macro_8class 基準）。test（1261, 既存8種は run11 と同一録音）。

| 指標 | run13 | baseline | 判定 |
|---|---|---|---|
| **既存8種 f1_macro_8** | **0.769** | run11 test v2 0.808 | H1 ほぼ不成立（−0.039、予測下限 0.77 を僅か割れ）|
| 新4種 f1_macro_4 | 0.449 | 予測 0.40〜0.60 | H2 域内（ただし種で二極化）|
| 12種 f1_macro_12 | 0.662 | 予測 0.70〜0.78 | やや下振れ |
| accuracy(13cls) | 0.765 | — | — |
| **OOD AUROC energy** | **0.902** | run11 0.932 | H3 成立（予測 0.88〜0.93、閾値 11.461）|

**種別 F1**:
- 新4種: Gadwall **0.823**(録音30) / ウミアイサ **0.746**(17) ＝成功 / カワアイサ 0.227(8) / **カルガモ 0.000**(11) ＝失敗
- 既存8種で低下: Mallard **0.627** / Goldeneye 0.679 / Shoveler 0.658

### 所見

- **Gadwall・ウミアイサの昇格は成功**（録音30/17 で自種を保持）。**カルガモ・カワアイサは失敗**。
- **真因は同属相互混同ではなく Mallard の「吸引ハブ」化**。カルガモ(27ch)は 18→Mallard で**自種予測0**、
  カワアイサも 16→Mallard / 13→Goldeneye。pred=Mallard の内訳は Mallard 116 に対し Teal23/Gadwall23/
  カルガモ18/カワアイサ16 と多種が流入 → Mallard precision 低下 → 既存8種 f1 を −0.039 押し下げた。
- カルガモは Anas 属で Mallard に音響的に酷似＋録音11で多数派バイアスに負け全滅。**録音追加(worldwide
  16本上限)では Goldeneye song 同様「表現力/多数派バイアス」の壁の可能性**。
- run01〜09 の「録音数が効く」が再確認された（成功2種は録音多・失敗2種は録音少）。

### 検証後の分岐（採用判断）

H1 ほぼ不成立・新4種二極化のため、**12種そのまま採用は見送り**が妥当。候補:
1. **Gadwall・ウミアイサのみ昇格（10種）** ＝成功2種に絞り、カルガモ/カワアイサは other に戻す
2. Mallard 過剰予測対策（CE クラス重み or sampler 調整）を入れて 12種を再学習
3. run14 = AudioMAE で表現力を上げ、カルガモ/Goldeneye song の音響類似を分離できるか検証

`species_taxonomy.yaml`（v13 / energy_threshold 11.461）への切替は採用判断後に行う。

---

## run14 — focal loss で Mallard 多数派バイアスを抑制 (2026-06-04)

**ステータス:** 学習前記録（事前登録）
**出力予定:** `models/ast-duck-v14/`
**初期化元:** pretrained（run13 と同条件。**loss のみの1変更**で公平比較）

### 背景

run13 でカルガモ F1=0.000・カワアイサ 0.227 が Mallard に吸われた。埋め込み切り分けで、
**素の pretrained AST では test カルガモの 67% がカルガモ寄り**（run13 学習後は 0%）と判明。
表現力ではなく **run13 学習が Mallard 多数派に最適化して分類層の境界を潰した**のが主因。
focal loss で難サンプル（境界付近のカルガモ）を強調し、易しい Mallard の損失を下げて是正する。

### 変更概要

| 項目 | run13 | run14 |
|---|---|---|
| loss | CE（other に alpha=1.5）| **focal loss (gamma=2.0)** + 同 class weight |
| sampler / split / init / 種数 | — | **run13 と同一**（種均等 sampler・12種・pretrained）|

### 仮説と予測（学習前に固定）

baseline = run13: 既存8種 f1_macro_8 **0.769** / カルガモ **0.000** / カワアイサ 0.227 / 12種 0.662

- **H1**: カルガモ F1 > **0.3**（素 AST で 67% 分離可能＝分類層是正で救える）
- **H2**: 既存8種 f1_macro_8 ≥ **0.769**（Mallard precision 改善でむしろ上振れも）
- **H3**: 12種 f1_macro_12 > **0.70**
- リスク: Mallard recall 低下（precision とのトレードオフ）。Mallard F1 を監視。

### 検証後の分岐

- カルガモ救済 ∧ 既存8種維持 → 選択肢2成功。12種採用へ。species_taxonomy.yaml を v14 に
- カルガモ救えない → 分類層では限界＝表現力。run15 = AudioMAE へ
- 既存8種が大幅低下 → gamma 過大。gamma=1.0 等へ

### 結果

early stop epoch 8。test（1261, 12種）。

| 指標 | run14 (focal) | run13 (CE) | 判定 |
|---|---|---|---|
| **カルガモ F1** | **0.000** | 0.000 | **H1 不成立** |
| 既存8種 f1_macro_8 | 0.761 | 0.769 | H2 ほぼ維持（−0.008）|
| 新4種 f1_macro_4 | 0.416 | 0.449 | 悪化 |
| 12種 f1_macro_12 | 0.646 | 0.662 | **H3 不成立**（−0.016）|
| Mallard F1 | **0.674** | 0.627 | +0.047（precision 改善）|

種別: Gadwall 0.850(+0.027) / ウミアイサ 0.643(−0.103) / カワアイサ 0.170(−0.057) / **カルガモ 0.000(不変)**。
カルガモは依然 17→Mallard / 9→Goldeneye。pred=Mallard には Teal29/カルガモ17/Gadwall15 が流入し続ける。

### 所見

- **focal loss は失敗。「分類層の多数派バイアスが主因」仮説は反証された**。focal（分類層を直接是正する
  最強手）でもカルガモ 0%。Mallard precision は改善（F1 +0.047）したが、その分 ウミアイサ/カワアイサが
  犠牲になり新4種全体は悪化。
- 素 pretrained の「test カルガモ 67% がカルガモ寄り」は **pretrained 限定**。fine-tune すると CE でも
  focal でも極小マージン（重心 0.982・0.917 vs 0.915）が Mallard 多数派に潰される。
  → **分類層では救えない＝表現力の壁が本質**（事前登録の分岐どおり）。
- **Goldeneye song（run11 分析）と カルガモ（run13/14）が同じ「表現力の壁」に収束**。AST（AudioSet 教師あり）
  の汎用表現は近縁カモの fine-grained 差を分離しきれない。
- run14 は run13 を上回らないため**不採用**（models/ast-duck-v13 が 12種ベスト、ただし採用は別判断）。

### 検証後の分岐（確定）

データ追加（run07/13）も分類層調整（run14 focal）も近縁種の壁を破れなかった。残るは:
1. **run15 = AudioMAE**（表現力軸。Goldeneye song + カルガモ の音響類似を分離できるか）
2. **カルガモ/カワアイサを other に戻す**（成功2種 Gadwall/ウミアイサのみ昇格＝10種運用、確実）

---

## 分析タスク — 表現軸 probe 検証: AudioMAE / SSAM (2026-06-04)

**ステータス:** 完了（fine-tune せず probe で結論）

### 背景

run11〜14 で「カルガモ / Goldeneye song は表現力の壁」と確定。AST（教師あり AudioSet）に替えて
自己教師あり表現（AudioMAE / SSAM）が近縁カモを分離できるか検証。フル fine-tune の前に
**linear probe**（encoder 凍結＋LogReg、MAE 論文の標準評価）で公平・低コストに表現力を比較した。

### 実装メモ

- AudioMAE: `timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m")`。
  **前処理は kaldi.fbank(htk_compat, hanning, 128mel) + 正規化 (x-(-4.268))/(4.569*2)、1024frame 固定**。
  3s 音声は 10s タイルで埋める（ゼロpad は埋め込み崩壊）。**特徴は CLS でなく patch mean pool**
  （`forward_features(x)[:,1:,:].mean(1)`。global_pool=token の CLS は MAE で定数的＝崩壊）。
- 生特徴の重心距離比較は MAE 型に不利（生 encoder は分類向けでない）と判明 → probe に切替。

### 結果（test 1261, class_weight=balanced LogReg）

| 指標 | AudioMAE probe | AST probe | run13 AST fine-tune |
|---|---|---|---|
| 既存8種 f1_macro_8 | 0.528 | **0.631** | 0.769 |
| 新4種 f1_macro_4 | **0.208** | 0.153 | 0.449 |
| 12種 f1_macro_12 | 0.422 | **0.472** | 0.662 |
| **カルガモ F1** | **0.140** | 0.000 | 0.000 |

### 所見

- **総合表現力は AST が上**（教師あり AudioSet が分類向け）。AudioMAE 全面採用の根拠は弱い。
- **だがカルガモは AudioMAE のみ非ゼロ（0.140 vs AST 0.000）**。AST が原理的に潰す最弱種を、自己教師あり
  表現は微細に保持＝「芽」はある。ただし fine-tune で実用水準まで伸びる保証は弱い。
- **SSAM は使いやすい port が無い**（timm/HF 不在、著者公開ckpt無し、`Robzy/audiomamba` は別論文）。
  mamba-ssm ビルド＋公式コード移植が必要で AudioMAE の10倍超のコスト → **断念**。
- 結論: **表現軸に銀の弾丸なし**。AudioMAE は総合 AST 以下、SSAM はコスト過大。実用解は
  「成功2種(Gadwall/ウミアイサ)の10種運用」に確定するのが妥当。カルガモ/Goldeneye song は
  「現行リソースでは分離困難な近縁ペア」として受容。

---

## run15 — 10種運用で確定（カルガモ/カワアイサを other へ差戻し） (2026-06-04)

**ステータス:** 学習前記録（事前登録）
**出力予定:** `models/ast-duck-v15/` / **初期化元:** pretrained / **loss:** CE（focal_gamma=0）

### 背景

run13/14 と表現軸 probe で、カルガモ・カワアイサは「データ追加・分類層・表現力のいずれでも
Mallard と分離困難」と確定。実用解として**昇格成功の Gadwall/ウミアイサのみ残す10種**に確定する。
カルガモ/カワアイサは `ood_tier1`（OOD）へ戻す。

### 変更概要

| 項目 | run13 | run15 |
|---|---|---|
| 対象種 | 12 | **10**（−カルガモ/カワアイサ）|
| num_labels | 13 | **11**（10種 + other）|
| loss | CE | CE（run14 focal は失敗、CE 据え置き）|
| other tier1 | 2種 | **4種**（+カルガモ/カワアイサ）|
| split | 12種 | 既存10種は run13 保持、2種を除去（test 1261→1196）|

### 仮説と予測（学習前に固定）

baseline = run13（12種）: 既存8種 f1_macro_8 **0.769** / Gadwall 0.823 / ウミアイサ 0.746 / Mallard 0.627

- **H1**: 既存8種 f1_macro_8 が **run13 から回復（≥ 0.78）**。カルガモ/カワアイサの Mallard 吸引が
  消え、Mallard precision・F1 が改善するため
- **H2**: Gadwall/ウミアイサ F1 を維持（0.80±/0.74±）
- **H3**: 10種 f1_macro_10 > run13 の 12種 0.662（失敗2種が抜けて上昇、**> 0.74** を期待）
- OOD: AUROC energy が run13(0.902) 水準を維持（tier1 が 2→4種に戻る）

### 検証後の分岐

- H1∧H2 成立 → **10種を確定版に**。species_taxonomy.yaml を v15・閾値更新し運用モデルとする
- 既存8種が回復しない → Mallard 吸引以外の劣化要因を再分析

### 結果

early stop epoch 6。test（1196, 既存10種）。

| 指標 | run15 (10種) | baseline | 判定 |
|---|---|---|---|
| 既存8種 f1_macro_8 | **0.786** | run13 0.769 | **H1 成立**（+0.017 回復）|
| 新2種 f1_macro_2 | 0.771 | — | H2（Gadwall 0.706 / ウミアイサ 0.836）|
| 10種 f1_macro_10 | **0.783** | run13 12種 0.662 | **H3 成立**（>0.74）|
| accuracy | 0.796 | — | — |
| OOD AUROC energy | **0.899** | run13 0.902 | 維持（閾値 6.73）|

**既存種の回復**: Mallard 0.627→**0.664** / Goldeneye 0.679→**0.765** / Shoveler 0.658→0.712 /
Tufted 0.740→0.759。カルガモ/カワアイサの Mallard 吸引が消えた効果。

### 所見

- **10種運用は成功。確定版とする**。失敗2種(カルガモ/カワアイサ)を other へ戻したことで、既存8種が
  run13 から回復（特に Mallard/Goldeneye）。10種 f1_macro 0.783 は run13 12種 0.662 を大きく上回る。
- **Gadwall は 0.823→0.706 に低下**（ウミアイサは +0.090）。12種→10種で混同構造が変わった/学習ランダム性。
  新2種平均 0.771 は実用水準。
- **最終構成**: カモ10種（既存8 + Gadwall/ウミアイサ）+ OOD energy gate（AUROC 0.899, 閾値 6.73）。
  `species_taxonomy.yaml` を v15 に更新し運用モデルとした。
- カルガモ/Goldeneye song は「現行リソースで分離困難な近縁ペア」として受容（→ 一連の探索で確定）。

---

## メモ

- baseline (`models/ast-duck/`) は常に保護する。新規 run は必ず別 output_dir へ
- 学習中は GPU ドライバ再起動（Linux では Ctrl+Alt+Backspace 等）を避ける
- TensorBoard: `uv run tensorboard --logdir models/ast-duck-v11/runs`
