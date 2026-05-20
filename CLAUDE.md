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

## 現状サマリ（更新: 2026-05-20）

- run01 baseline: f1_macro **0.848** (epoch 4)。明確な overfit
- run02: ハイパラ4点 + SpecAugment 同時変更で f1_macro **0.826** に悪化（失敗）
- run03: lr 単独評価（lr=2e-5 / wd=0.01 / patience=4 / SpecAugment オフ）を準備済み。条件・仮説は experiments.md に事前登録。学習待ち
