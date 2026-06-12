# bird-fine-classifier

カモ**10種**の鳴き声を分類する fine-grained classifier（**BirdNET後段・3秒固定チャンク**）。Xeno-canto 音源で AST を fine-tune。別系統で Perch2.0 凍結埋め込み＋軽量プローブも持ち、**Perch→AST 知識蒸留＋多seed soup** で運用モデル `models/ast-duck-C-kd-soup` を構築。OOD energy ゲート＋複合クラス出力（マガモ/カルガモ）を備える。**詳細・最新は `docs/perch_kd_report.md`**。

## プロジェクト構成

- `config.yaml` — 対象種・前処理・モデル・学習・評価の全パラメータを一元管理
- `main.py` — メインエントリ
- `src/` — 実装本体（dataset, training, evaluation など）
- `data/raw/` — Xeno-canto 生音源（git ignore）
- `data/processed/` — 16kHz / **3秒チャンク**（BirdNET 3s 窓に整合）
- `data/embeddings/` — Perch/BirdAVES 凍結埋め込み・教師proba（git ignore）
- `species_taxonomy.yaml` — **推論の単一の真実**: グループ別 stage2_model / OOD閾値 / 複合表示(display_groups)
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

### 評価の規律（2026-06-05 確立 / 必読）

- **run 間の精度差は「録音単位 + bootstrap CI」で判定する**。`bird_fine.analysis.compare_runs_ci` を使う。
- **chunk 単位 macro-F1 を run 比較に使わない**。test は録音数が少なく（既存8種で 69 録音）、同一録音内
  chunk は強く相関する。chunk 単位は実効サンプルを過大評価し CI を過小評価＋性能を過小評価する。
- **CI が重なる差（≈±0.05 未満）は「差なし＝ノイズ」**として扱い、run の優劣を断定しない。
  2026-06-05 の検証で run11/13/15 の既存8種差はすべて有意でなかった（→ journal.md 2026-06-05）。
- 意思決定は**大きな構造的差**（桁で動く・F1=0 等）と**定性的所見**のみで行う。
- **単一シードの点推定で「+0.0XX 改善」を効果と解釈しない**。必要なら複数シード。
- 根本課題: **test の録音数が少なすぎる**（解像度不足）。録音数拡大が精度比較の前提。

### コミット粒度

- 1 run = 1 commit を目安に（config.yaml + docs/experiments.md + journal.md + 関連コード変更）
- **事前登録する run は 2 commit**: 学習前に `run<NN>(pre): 条件と仮説を学習前に記録`、学習後に結果を `run<NN>: <結果>` で別コミット。「結果を見る前に予測を固定した」ことを git 履歴で担保するため
- メッセージ例: `run02: lr↓ wd↑ + SpecAugment で overfit対策（結果: f1 0.826 / 失敗）`

### 既知の落とし穴

- **学習中に `Win+Ctrl+Shift+B`（GPUドライバ再起動）を押さない** — CUDA コンテキストが吹っ飛ぶ
- `data/raw/` は容量大。git ignore 対象、コミットしない
- SpecAugment は **train_ds のみに適用**。val/test に漏れていないか実装確認すること

## 現状サマリ（更新: 2026-06-12）

**運用モデル** = `models/ast-duck-C-kd-soup`（Perch→KD蒸留 3seed soup）。test 録音単位 macro-f1 **0.871**。

- **系統2つ**: ①AST fine-tune（運用本線, `train.py` が KD/`--distill` 対応）②Perch2.0/BirdAVES 凍結埋め込み＋軽量プローブ（比較・**蒸留の教師**）。
- **蒸留**: Perch教師(複数arm平均, train OOF)を AST に温度付きKL蒸留→**多seed soup**で固める。素CNNで有効性を統計確認(+0.148 有意)、AST soupで有意化(base-soup0.813→**0.871**, +0.058 有意)。レシピ λ≈1/T≈2。
- **OOD energy ゲート**: `predict.py` が `species_taxonomy.yaml` の閾値で判定（**録音平均energy**）。現 **2.717**（録音単位再キャリブレ, 真カモ保持0.90）。正準ツール= `tools/ood_fp_audit.py`（`ood_eval.py` の chunk単位閾値は本番に使わない）。
- **複合クラス**: カルガモ↔マガモは **Perch本体でも分離不能(0/30)＝音響的に本質的**→ `display_groups` で Mallard を「マガモ/カルガモ」表示（再学習ゼロ・カルガモ受容）。`tools/perch_native_confusion.py` が実証。
- **評価規律**: 録音単位 macro-f1 ＋ 録音クラスタ bootstrap CI（`analysis/compare_runs_ci.py`）。test n=231。**CI が重なる差(≈±0.05)は「差なし」**。chunk単位の点推定で優劣を断定しない。
- **構造的事実**: 10種内に混同ペア無し（弱種=小標本振れ, ヒドリ n=7 等）。残る伸びしろは **test拡大とデータ**（モデル/蒸留は天井近く）。
- **環境/git**: 学習はmainPC(`tailscale ssh mushipi-mainpc-ubuntu`)。push は GT105 リレー（mainPC token無効）。
- 旧 AST run01-09（8種/10秒, ~2026-05）の詳細は `docs/journal.md`/`experiments.md`。**本系統の全経緯は `docs/perch_kd_report.md`**。
