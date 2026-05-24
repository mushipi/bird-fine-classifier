# bird-fine-classifier

カモ8種の鳴き声を分類する AST (Audio Spectrogram Transformer) ベースの fine-grained classifier。Xeno-canto から音源収集し、HuggingFace の事前学習済み AST を fine-tune する。

## プロジェクト構成

- `config.yaml` — 対象種・前処理・モデル・学習・評価の全パラメータを一元管理
- `main.py` — メインエントリ
- `src/` — 実装本体（dataset, training, evaluation など）
- `data/raw/` — Xeno-canto 生音源（git ignore）
- `data/processed/` — 16kHz / 10秒チャンクに変換済み
- `data/splits/` — train/val/test = 0.70/0.15/0.15
- `models/<run-name>/` — 各 run の出力。checkpoint-*, model.safetensors, trainer_state.json, runs/（TensorBoard）
- `docs/experiments.md` — **実験 run の記録**（ハイパラ・結果・所見）。1 run 1 セクションで時系列追記
- `docs/journal.md` — **思考・失敗・気づきの時系列ログ**。判断の経緯や学びを残す
- `docs/architecture.md` / `data_guide.md` / `training_guide.md` / `troubleshooting.md` — 各種ガイド

## 環境

- パッケージ管理: **uv 必須**（`uv run`, `uv add` ベース）。pip / venv 直叩きは禁止
- GPU: RTX 3060 Ti (VRAM 8GB)。fp16 + gradient_checkpointing 前提
- ベースモデル: `MIT/ast-finetuned-audioset-10-10-0.4593`
- TensorBoard: `uv run tensorboard --logdir models/<run-name>/runs`

## 運用ルール

### 学習 run の流れ

1. **新規 run は必ず別の output_dir** に出す（例: `models/ast-duck-v3/`）。baseline (`models/ast-duck/`) は保護
2. `config.yaml` を編集する前に、現状からの差分意図をコメントで残す（例: `# v2: 5.0e-5 → overfit対策で半分以下`）
3. **学習前（事前登録）**: `docs/experiments.md` に `## run<NN> — <タイトル> (YYYY-MM-DD)` セクションを先行作成
   - ステータスに「学習前記録（事前登録）」と明記
   - ハイパラ差分表 / 仮説と予測（予測値を具体的に固定）/ 検証後の分岐
   - 結果・経過・所見はプレースホルダにし、`run<NN>(pre):` でコミット（→ コミット粒度参照）
4. **学習完了後**: 同セクションの結果欄を埋める
   - 主要マイルストーン表（epoch / train_loss / eval_loss / eval_f1_macro）
   - 結果メトリクス表（過去 run との比較カラム付き）
   - 所見（仮説の成否と検証分岐に沿った次のアクション）
5. 設計判断・失敗・気づきは `docs/journal.md` に追記（experiments.md には書ききれない経緯・学び）

### 思考と失敗の記録（journal）

- 日付ごとにセクション（`## YYYY-MM-DD <タイトル>`）
- 「やったこと / 結果 / なぜそうなったか / 学び / 次にやる」を意識して書く
- **失敗は隠さず記録する**。再発防止と切り分け学習のため
- 過度な装飾不要。事実と判断理由を残すことが目的

### ハイパラ実験の原則

- **1 run 1 変更が原則**。複数同時変更は切り分け不能になり学習効率が落ちる（→ journal.md 2026-05-19 参照）
- パラメータ間の整合性チェック必須：
  - lr を下げたら patience は上げる方向
  - 正則化を強めたら学習率も合わせて検討
- 変更パラメータは「主軸（lr, weight_decay, patience）」と「補助（augmentation）」に分け、主軸を先に評価

### コミット粒度

- 1 run = 1 commit を目安に（config.yaml + docs/experiments.md + journal.md + 関連コード変更）
- **事前登録する run は 2 commit**: 学習前に `run<NN>(pre): 条件と仮説を学習前に記録`、学習後に結果を `run<NN>: <結果>` で別コミット。「結果を見る前に予測を固定した」ことを git 履歴で担保するため
- メッセージ例: `run02: lr↓ wd↑ + SpecAugment で overfit対策（結果: f1 0.826 / 失敗）`

### 既知の落とし穴

- **学習中に `Win+Ctrl+Shift+B`（GPUドライバ再起動）を押さない** — CUDA コンテキストが吹っ飛ぶ
- `data/raw/` は容量大。git ignore 対象、コミットしない
- SpecAugment は **train_ds のみに適用**。val/test に漏れていないか実装確認すること

## 現状サマリ（更新: 2026-05-24）

- run01 baseline: f1_macro **0.848** (epoch 4)。明確な overfit
- run02: ハイパラ4点 + SpecAugment 同時変更で f1_macro **0.826** に悪化（失敗）
- run03: lr 単独評価。f1_macro **0.875** (epoch 2)。学習直後のシステム異常終了で成果物が一部破損 → checkpoint-252 から復旧済み（journal 2026-05-21）
- run04: weight_decay 単独評価（0.01→0.05）。f1_macro **0.850** (epoch 5)。wd↑ で overfit ピークが後退（H1 成立）も f1 低下
- run05: weight_decay 中間点評価（0.03）。val f1_macro **0.840** (epoch 6)。wd 0.01/0.03/0.05 → val f1 0.875/0.840/0.850 で dose-response 非単調
- **test 評価 (2026-05-22)**: run03/04/05 を未使用 test 338件で評価。test f1 は 0.775 / 0.806 / **0.838** で val 順位が逆転。val 最良の run03 が test 最低 = `load_best_model_at_end` が noisy な val f1 のスパイクを掴む選択バイアス。**現行ベストは run05（test f1 0.838）**
- 種別分析 (2026-05-22): test F1 は train 録音数と連動（録音20本の Tufted_Duck / Eurasian_Wigeon が最弱）。Tufted_Duck は trainチャンク最多694だが録音20本＝多様性不足＋チャンク不均衡。「少数種のデータ不足」は誤り
- **run06 (2026-05-24)**: SpecAugment クリーン単独評価。val f1 **0.846** (epoch 2) / **test f1 0.810**（run05 比 −0.028）。H1a 成立（train_loss が桁単位で上振れ）も H1b/H2 不成立 — best_epoch が逆に早期化し val ピーク選択バイアスを引き戻した。Tufted_Duck +0.05 も Northern_Shoveler −0.19 で全体下落。**現行ベストは引き続き run05**
- 学び: 「train 暗記の抑制」は汎化を保証しない。正則化系手法は val 曲線の形状を変えて `load_best_model_at_end` との相性が悪い場合がある
- **run07 (2026-05-24)**: 弱2種に quality=B 録音を +10 ずつ追加（test 固定）。val f1 **0.8446** (epoch 4) / **test f1 0.8268**（run05 比 −0.011）。H1〜H4 不成立。Tufted_Duck precision 0.269→0.296 で過剰予測未解決、Mallard −0.029 / Northern_Shoveler −0.108（小サンプルでバイアス）と他種へ負の波及。**現行ベストは引き続き run05**
- 学び: 録音追加でチャンクも増え（Tufted train 694→728）チャンク不均衡が悪化、録音多様性向上の効果を打ち消した。**録音追加とチャンク上限は同時にやるべき**
- **run08 (2026-05-24)**: cap=100 で Tufted train chunks 728→361（−50%）、他種不変。val f1 **0.8381** (epoch 6) / **test f1 0.8203**（run05 比 −0.018）。**Tufted precision が run05 と完全同値 0.269（小数第3位まで一致）**、誤吸引の構成（Teal 13/Wigeon 4 件）もほぼ run05 と同じ。**「chunks 不均衡 = Tufted 過剰予測」仮説が完全に崩れた**
- 学び: 真のボトルネックは音響的類似性 / モデル容量の可能性。**chunks 介入では永遠に解決しない**。介入していない他種（Mallard −0.069 / Wigeon −0.049）が大幅悪化、学習ダイナミクスは局所介入でも全体に波及
- 次: 6 run 撃って 0.85 天井を破れず構造説が強まる。**(1) 混同パターンの音響的根拠を見る**（Tufted/Teal の分光図比較） → (2) class-weighted loss → (3) AST-Large の順で検討。「データ・ハイパラ」ではなく「モデル容量・特徴表現」の軸へ
