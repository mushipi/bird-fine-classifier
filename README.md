# bird-fine-classifier

BirdNET の後段として動作する、**カモ類細分類**の音声分類プロジェクト。
2段階パイプラインの第2段（Stage2）として、カモ類10種を識別する。

```
音声 → BirdNET (CNN) → 「カモ類」検出 → 本モデル → マガモ/コガモ/… ＋ OOD棄却
```

BirdNET は近縁種の細分類が苦手なため、**上位カテゴリで絞り込んだ後の専門識別器**として
本モデルを位置づける。BirdNET の 3秒窓に合わせ、本モデルも **3秒固定チャンク**で運用する。

## 対象種（運用10種）

| 英名 | 学名 | 和名 |
|---|---|---|
| Mallard | Anas platyrhynchos | マガモ |
| Eurasian Teal | Anas crecca | コガモ |
| Northern Pintail | Anas acuta | オナガガモ |
| Northern Shoveler | Spatula clypeata | ハシビロガモ |
| Eurasian Wigeon | Mareca penelope | ヒドリガモ |
| Gadwall | Mareca strepera | オカヨシガモ |
| Tufted Duck | Aythya fuligula | キンクロハジロ |
| Common Pochard | Aythya ferina | ホシハジロ |
| Common Goldeneye | Bucephala clangula | ホオジロガモ |
| Red-breasted Merganser | Mergus serrator | ウミアイサ |

- 種選定は**北部九州の iNaturalist 観察頻度**を主軸に査定（`data/species_master.csv` が主マスタ）。
- 上記以外のカモ（カルガモ・スズガモ・カワアイサ等）や非カモ音は **OOD energy ゲート**で棄却する
  （学習対象に残すと既存種を巻き添えにするため、`status: ood_*` として除外）。

## 2つのモデル系統

| 系統 | 概要 | 位置づけ |
|---|---|---|
| **AST fine-tune** | `MIT/ast-finetuned-audioset` を10種にfine-tune | **運用本線**（Stage2モデル） |
| **Foundation埋め込み + 軽量プローブ** | Perch2.0 / BirdAVES の凍結埋め込み上に線形/MLP/sklearnプローブ | 比較・**蒸留の教師**として活用 |

両系統は同水準（録音単位f1 ≈ 0.83）。**Perch を教師に AST へ知識蒸留（KD）**し、多seed soup で
固めた `models/ast-duck-C-kd-soup` を運用モデルに昇格している。

### 現行モデルと性能（test 録音単位 macro-F1, n=231）

- **運用モデル**: `models/ast-duck-C-kd-soup`（Perch→KD蒸留 3seed soup）= **0.871**
- ベースライン（蒸留なし soup）= 0.813（差 +0.058, 95%CI[+0.018, +0.108]＝有意）
- OOD energy ゲート: AUROC 0.886 / 推奨閾値 2.948（FPR≤5%）
- 詳細・経緯 → **[docs/perch_kd_report.md](docs/perch_kd_report.md)**

評価は **録音単位 macro-F1 ＋ 録音クラスタ bootstrap CI** を規律とする
（チャンク単位の点推定は小サンプルで CI を過小評価するため使わない。`analysis/compare_runs_ci.py`）。

## クイックスタート

### セットアップ
```bash
uv sync                       # 本体（torch CUDA, transformers）
# 隔離環境（埋め込み抽出用、フレームワーク分離）
#   tools/perch_embed/.venv  : TensorFlow + perch-hoplite
#   tools/aves_embed/.venv   : torch + esp-aves
```
`.env` に Xeno-canto APIキー（`XENO_CANTO_API_KEY`）。

### データ整備
```bash
uv run python -m bird_fine.data.download        # Xeno-canto DL
uv run python -m bird_fine.data.preprocess      # 16kHz mono + 3秒チャンク
uv run python -m bird_fine.data.split           # 録音ID単位 train/val/test
uv run python -m bird_fine.data.download_ood    # OOD評価用音声（species_taxonomy 駆動）
```

### AST 学習（本線）
```bash
uv run python -m bird_fine.training.train --duck-order data/embeddings/teacher_proba/duck_order.csv \
    --output-dir models/ast-duck-XX --seed 42 --num-workers 8
# 蒸留する場合: --distill --kd-lambda 1.0 --kd-temp 2.0 --teacher-dir data/embeddings/teacher_proba
```

### Foundation 埋め込み + プローブ（教師の作成）
```bash
# Perch 埋め込み抽出（3秒チャンク, raw 32kHz）
tools/perch_embed/.venv/bin/python tools/perch_embed/extract_perch.py --source raw --min-chunk-sec 1.0
# プローブのスイープ / 教師ソフトラベル出力
.venv/bin/python tools/probe_sweep/run_sweep.py --emb-dir data/embeddings/perch --tag perch
.venv/bin/python tools/probe_sweep/export_teacher_proba.py     # 複数arm平均, train OOF
```

### 評価・蒸留・昇格
```bash
.venv/bin/python tools/ast_eval_proba.py --model-dir models/ast-duck-XX --tag XX  # test proba
.venv/bin/python tools/probe_sweep/kd_compare_ci.py --base <A> --kd <B>           # 録音単位CI
.venv/bin/python tools/soup_ast.py --out models/ast-duck-C-kd-soup <kd1> <kd2> <kd3>  # 重み平均
uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-C-kd-soup    # OOD閾値
```

### 推論（OODゲート込み）
```bash
uv run python -m bird_fine.inference.predict --audio path/to/duck.wav
# モデル・OOD閾値は species_taxonomy.yaml の <group>.pipeline から読む（推論の単一の真実）
```

## 推論設定（species_taxonomy.yaml）

グループ別の **Stage2 モデル・OOD energy閾値・温度** を保持する、推論/デプロイの単一の真実。
```yaml
duck:
  pipeline:
    stage2_model: "models/ast-duck-C-kd-soup"
    energy_threshold: 2.948      # energy < 閾値 → OOD棄却
    energy_temperature: 1.0
```
`predict.py` は `--group`（既定 duck）でこのブロックを参照する。

## ディレクトリ構成

```
bird-fine-classifier/
├── config.yaml                 # 種・前処理・学習ハイパラ
├── species_taxonomy.yaml       # 推論モデル/OOD params（グループ別）
├── docs/                       # ドキュメント（perch_kd_report.md 等）
├── data/                       # gitignore: raw/ processed/ ood/ embeddings/ + splits/
├── models/ outputs/            # gitignore
├── src/bird_fine/
│   ├── data/                   # download / preprocess(3s) / split / OOD / species_master
│   ├── models/                 # ast_classifier.py, probes.py(linear/mlp/attentive)
│   ├── embeddings/             # birdaves.py, io_utils.py（埋め込み入出力）
│   ├── training/               # train.py(KD対応), evaluate.py, train_probe.py, eval_probe.py
│   ├── inference/              # predict.py(OODゲート), export_onnx.py
│   └── analysis/               # ood_eval.py, compare_runs_ci.py（録音単位CI）
└── tools/                      # 隔離環境・実験スクリプト
    ├── perch_embed/  aves_embed/      # 埋め込み抽出（.venv 分離）
    ├── probe_sweep/                    # スイープ・教師export・CI比較・グリッド集計
    ├── train_cnn_kd.py  soup_ast.py    # 素CNN蒸留検証・重みsoup
    └── ast_eval_proba.py               # AST proba ダンプ
```

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/perch_kd_report.md](docs/perch_kd_report.md) | **Perch蒸留による強化の全記録**（手法・多seed soup・昇格） |
| [docs/architecture.md](docs/architecture.md) | パイプライン全体、AST の仕組み |
| [docs/data_guide.md](docs/data_guide.md) | DL戦略、チャンク化仕様、leakage対策split |
| [docs/training_guide.md](docs/training_guide.md) | ハイパラ・メトリクス解釈 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | OOM・データ不足等の対処 |

## 環境

- **Python** 3.12 / **uv** で依存管理（`uv.lock` 固定）
- **PyTorch** 2.6（CUDA 12.4）/ **Transformers** / 埋め込みは隔離 venv（TF / esp-aves）
- **想定GPU**: RTX 3060 Ti（VRAM 8GB）。学習は mainPC、抽出/評価も同機（`tailscale ssh`）
- DataLoader は `--num-workers` で並列化（Linux）

## ライセンス・注意

- Xeno-canto の録音は CC BY-NC-SA 等。**研究・個人実験用途のみ**。
- 学習済みモデルを公開する場合は再配布条件に注意。

## 関連リンク

- [Audio Spectrogram Transformer (AST)](https://arxiv.org/abs/2104.01778) ／ [HF AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [Perch 2.0](https://arxiv.org/abs/2508.04665) ／ [BirdAVES (earthspecies/aves)](https://github.com/earthspecies/aves)
- [Xeno-canto](https://xeno-canto.org/)
