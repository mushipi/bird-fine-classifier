# bird-fine-classifier

BirdNetの後段として動作する、**カモ類細分類用** のAudio Spectrogram Transformer (AST) fine-tuneプロジェクト。

## 概要

```
音声 → BirdNet (CNN) → 「カモ類」検出 → 本モデル (AST) → マガモ/コガモ/...
```

BirdNet（CNN）はカモ類・カモメ類・カラス類のような近縁種の細分類が苦手なため、
**上位カテゴリで絞り込んだ後の専門識別器** として本モデルを位置づける2段階パイプラインを構築する。

本PJは第一弾としてカモ類8種に特化したASTのfine-tuneを行う。将来的にはカモメ類・カラス類用も同じ構造で追加予定。

## 対象種

| 英名 | 学名 | 日本語名 |
|---|---|---|
| Mallard | Anas platyrhynchos | マガモ |
| Common Teal | Anas crecca | コガモ |
| Northern Pintail | Anas acuta | オナガガモ |
| Northern Shoveler | Spatula clypeata | ハシビロガモ |
| Eurasian Wigeon | Mareca penelope | ヒドリガモ |
| Tufted Duck | Aythya fuligula | キンクロハジロ |
| Greater Scaup | Aythya marila | スズガモ |
| Common Pochard | Aythya ferina | ホシハジロ |

## クイックスタート

### 1. セットアップ

```powershell
uv sync
```

`.env` に Xeno-canto APIキーを設定：

```env
XENO_CANTO_API_KEY=your_api_key_here
```

APIキーは https://xeno-canto.org/account/api で取得。

### 2. パイプライン実行

```powershell
# データダウンロード（メタデータ確認 → 本DL）
uv run python -m bird_fine.data.download --metadata-only   # 件数確認
uv run python -m bird_fine.data.download                   # 本DL

# 前処理 → 分割
uv run python -m bird_fine.data.preprocess
uv run python -m bird_fine.data.split

# 学習（dry-runでループ確認 → 本学習）
uv run python -m bird_fine.training.train --dry-run
uv run python -m bird_fine.training.train

# 評価
uv run python -m bird_fine.training.evaluate

# 推論
uv run python -m bird_fine.inference.predict --audio path/to/duck.wav
```

各コマンドの詳細は `docs/` を参照。

## ディレクトリ構成

```
bird-fine-classifier/
├── .env                        # XENO_CANTO_API_KEY（gitignore）
├── config.yaml                 # 種リスト・ハイパラ・前処理設定
├── pyproject.toml              # 依存（torch CUDA, transformers, xcapi）
├── main.py                     # 起動ヒント表示のみ
├── docs/                       # ドキュメント
│   ├── architecture.md         # 全体設計・ASTの仕組み
│   ├── data_guide.md           # データ収集・前処理仕様
│   ├── training_guide.md       # ハイパラ・メトリクス解説
│   └── troubleshooting.md      # OOM・データ不足等の対処
├── data/                       # gitignore
│   ├── raw/                    # xcapi DL生データ
│   ├── processed/              # 10秒チャンク化済み
│   └── splits/                 # train/val/test CSV + label_map
├── models/                     # gitignore（学習チェックポイント）
├── outputs/                    # gitignore（評価結果・可視化）
└── src/bird_fine/
    ├── data/
    │   ├── download.py         # xcapi DLスクリプト
    │   ├── preprocess.py       # 16kHz mono + 10秒チャンク
    │   ├── split.py            # 録音ID単位train/val/test
    │   └── dataset.py          # PyTorch Dataset + FeatureExtractor
    ├── models/
    │   └── ast_classifier.py   # ASTヘッド置換
    ├── training/
    │   ├── train.py            # HuggingFace Trainer fine-tune
    │   └── evaluate.py         # 混同行列・F1・Attention可視化
    └── inference/
        └── predict.py          # top-K推論CLI
```

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | パイプライン全体、ASTの仕組み、設計判断の根拠 |
| [docs/data_guide.md](docs/data_guide.md) | DLフォールバック戦略、チャンク化仕様、leakage対策split |
| [docs/training_guide.md](docs/training_guide.md) | ハイパラ意味、TensorBoardの見方、メトリクス解釈 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | OOM・データ不足・学習が進まない等のトラブル対処 |

## 環境

- **Python**: 3.12+
- **PyTorch**: 2.4+ (CUDA 12.4対応版)
- **HuggingFace Transformers**: 4.45+
- **想定GPU**: RTX 3060 Ti（VRAM 8GB）以上
- **OS**: Windows 11（DataLoader `num_workers=0` 固定）

## 流用元

| 流用元PJ | 流用内容 |
|---|---|
| `BirdProject` | xcapi DL基盤、録音ID単位splitの考え方、config.yaml構造 |

## ライセンス・注意

- Xeno-cantoの録音はCC BY-NC-SA等の条件付きCC。**研究・個人実験用途のみ**。
- 学習済みモデルを公開する場合は再配布条件に注意。

## 関連リンク

- [Audio Spectrogram Transformer (AST) 論文](https://arxiv.org/abs/2104.01778)
- [HuggingFace AST モデル](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [Xeno-canto](https://xeno-canto.org/)
- [xcapi（DLライブラリ）](https://github.com/bghani/xcapi)
