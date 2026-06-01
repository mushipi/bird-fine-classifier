# Data Guide

データ収集 → 前処理 → 分割 の各ステップの仕様と運用方法。

---

## 0. 種マスタの管理

### species_master.csv が起点

全データ収集はこのファイルで管理されている種を対象に行う。

```bash
# iNaturalist から北部九州の観察種を同期（candidate を追記）
uv run python -m bird_fine.data.sync_species_master --dry-run  # 件数確認
uv run python -m bird_fine.data.sync_species_master            # 本実行
```

取得した candidate に対して手動でステータスを付与する:

| status | 意味 |
|---|---|
| `target` | Stage2 が識別する種 → 学習データを収集 |
| `ood_tier1` | 近縁 OOD 種 → Outlier Exposure 用データを収集 |
| `ood_tier2/3` | OOD テスト用 |
| `ignore` | 収集不要 |
| `candidate` | 未決定（デフォルト）|

---

## 1. データダウンロード

### 学習データ（target 種）

`download.py` は `status="target"` の種を Xeno-canto からダウンロードする。

**フォールバック戦略:**

```
Japan で検索 → ヒットあり → 採用
    ↓ 0件
worldwide で検索 → ヒットあり → 採用
    ↓ 0件
YouTube で補完（下記参照）
```

**品質フィルタ** (`config.yaml` の `download.quality`):
- `"A"`: 最高品質のみ（推奨）
- `"A B"`: データ量重視ならこちら

```bash
uv run python -m bird_fine.data.download --metadata-only  # 件数確認
uv run python -m bird_fine.data.download                  # 全種 DL
uv run python -m bird_fine.data.download --species Mallard "Common Teal"  # 特定種
```

### OOD データ

`download_ood.py` は master の `ood_tier1〜3` 種をダウンロードし、3s チャンクに前処理する。

```bash
uv run python -m bird_fine.data.download_ood --metadata-only  # 件数確認
uv run python -m bird_fine.data.download_ood                  # 全 tier DL
uv run python -m bird_fine.data.download_ood --tiers 1 2      # tier 指定
```

### YouTube 補完（Xeno-canto 不足時）

日本産録音が著しく少ない種（例: カササギ）は YouTube で補完する。

```bash
# 取得
yt-dlp -x --audio-format wav -o "data/raw_youtube/%(title)s.%(ext)s" "URL"

# 16kHz mono に変換
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

**必須ルール:**
1. 鳴き声区間を人力で聴取確認してから使用
2. `species_master.csv` の `data_source` 列に `youtube` と記録
3. `notes` に URL と収録日を残す

---

## 2. 前処理

`preprocess.py` は raw 音声を 3s WAV チャンクに変換する。

**チャンク化ルール:**

| 元の長さ | 処理 |
|---|---|
| `< 1秒` | スキップ |
| `1〜3秒` | ゼロパディングで 3s に伸ばす（1 チャンク）|
| `> 3秒` | 3s ずつ非重複で切り出し、残り 1s 以上ならパディングして追加 |

設定（`config.yaml`）:

```yaml
preprocessing:
  sample_rate: 16000
  chunk_duration_sec: 3.0    # BirdNET の処理窓に合わせる
  min_chunk_duration_sec: 1.0
  overlap_ratio: 0.0
```

```bash
uv run python -m bird_fine.data.preprocess
```

**問題録音への対応（per-recording cap）:**

run09 の分析で「長尺録音（45〜49分）が決定境界を歪める」ことが判明。
問題録音が発見された場合:

```bash
# chunks_index.csv から特定録音を除外して re-split
uv run python -m bird_fine.data.exclude_train_recordings --xc-ids XC488112 XC488113
```

---

## 3. データ分割

`split.py` は `chunks_index.csv` を train/val/test に分ける。

### Leakage 防止（録音 ID 単位 split）

同じ録音から作られたチャンクは必ず同じ split に入る。

```
NG（チャンク単位）: XC123456_chunk000 → train / XC123456_chunk001 → test
OK（録音 ID 単位）: XC123456 の全チャンク → train
```

```bash
uv run python -m bird_fine.data.split
```

出力:

```
data/splits/
├── train.csv      # 学習チャンク（target 8種 + other が混在）
├── val.csv        # 検証チャンク
├── test.csv       # テストチャンク（評価用、学習中は触らない）
└── label_map.csv  # species → label_id
```

### "other" クラスの追加

Outlier Exposure を行う場合は split 後に実行:

```bash
uv run python -m bird_fine.data.prepare_other_class --dry-run  # 件数確認
uv run python -m bird_fine.data.prepare_other_class            # train/val.csv に追記
```

これにより:
- `train.csv`: target 8 種 + "other" 1220 chunks
- `val.csv`: target 8 種 + "other" 391 chunks（録音単位で分離済み）
- `label_map.csv`: "other" → label_id=8 を追記

---

## 4. 問題録音の発見と対処

過去の実験で判明した問題録音パターン:

| パターン | 症状 | 対処 |
|---|---|---|
| 長尺フィールド録音（45分超）| 多様な音響パターンが1種ラベルで大量投入され決定境界を歪める | `exclude_train_recordings` で除外 |
| juvenile タグ音声が 1 件しかない | distribution shift で幼鳥鳴き声が他種に誤分類 | Xeno-canto で `stage:juvenile` を指定して追加収集 |

```bash
# 問題録音の調査
uv run python -m bird_fine.analysis.confusion_audio  # 誤分類チャンクの mel 表示
```

---

## 5. データ量の目安

| 指標 | 最低限 | 推奨 |
|---|---|---|
| 種あたりの録音数 | 10件 | 30件以上 |
| 種あたりの train chunks | 100 | 300〜500 |
| "other" 全体 train chunks | 500 | 1000以上 |

per-recording cap（`prepare_other_class` のデフォルト: 30 chunks/録音）で
長尺録音による chunk 不均衡を防ぐ。

---

## 6. ライセンス

Xeno-canto の録音は CC BY-NC-SA 等の条件付き:
- ✅ 個人研究・教育・非商用利用
- ❌ 商用配布・未クレジットの再公開
- モデル公開時は録音者のクレジット要件を確認

YouTube 補完データはソースの利用規約を各自確認すること。
