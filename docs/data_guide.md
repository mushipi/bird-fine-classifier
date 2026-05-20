# Data Guide

データ収集（DL）→ 前処理 → 分割 の各ステップの仕様と運用方法。

---

## 1. データダウンロード

### 何をするスクリプトか

`src/bird_fine/data/download.py` は **Xeno-canto API v3** にxcapiライブラリ経由でアクセスし、
config.yamlで指定されたカモ類8種の音声をダウンロードする。

### フォールバック戦略

```
種ごとに以下の順で検索：
  1. Japan で検索 → ヒットあり？ → そのまま採用
  2. (1で0件なら) worldwide で検索
  3. (2でも0件なら) 録音なしとして記録
```

カモ類は渡り鳥なので、日本国内録音だけだと種によっては大幅にデータ不足になる。
そのため`fallback_worldwide: true` をデフォルトとしている。

### 品質フィルタ

config.yamlの `download.quality`:

- `"A"`: 最高品質のみ（推奨、データ少なくなる可能性あり）
- `"A B"`: A + B 品質（データ量重視ならこちら）

### 重複防止

xcapi Downloaderは `xcapi_runs.json` でDL済みIDを管理。再実行しても重複DLしない。
強制再DLしたい場合は出力フォルダごと削除する：

```powershell
Remove-Item -Recurse -Force data\raw
```

### コマンド例

```powershell
# 全種、メタデータのみ確認（DLしない、件数チェック用）
uv run python -m bird_fine.data.download --metadata-only

# 全種DL
uv run python -m bird_fine.data.download

# 特定種のみ
uv run python -m bird_fine.data.download --species Mallard "Common Teal"

# 種あたり50件に制限
uv run python -m bird_fine.data.download --max-per-species 50
```

### 出力構造

```
data/raw/
├── Mallard/
│   ├── XC123456.mp3
│   ├── XC123457.mp3
│   ├── metadata.csv          # 録音ごとの詳細（座標、品質、録音者...）
│   └── xcapi_runs.json       # DL履歴
├── Common_Teal/
│   └── ...
└── ...
```

### メタデータ列（metadata.csv）

主要なもの：

| 列名 | 意味 |
|---|---|
| id | Xeno-canto録音ID（XC{id}） |
| gen / sp | 学名（属/種） |
| en | 英名 |
| cnt | 国 |
| loc | 場所 |
| lat / lon | 緯度・経度 |
| length | 録音長（mm:ss） |
| q | 品質ランク（A〜E） |
| type | コール種別（call/song/alarm/...） |
| file | 音声URL |

---

## 2. 前処理

### 何をするスクリプトか

`src/bird_fine/data/preprocess.py` は raw音声を**ASTの入力フォーマット** に変換する：

1. mp3/wav/flac/ogg を librosa で読み込み
2. 16kHz mono にリサンプリング・ダウンミックス
3. 10秒チャンクに分割
4. PCM_16 wav として保存

### チャンク化ルール

| 元の長さ | 処理 |
|---|---|
| `< 3秒` | スキップ（情報量不足） |
| `3〜10秒` | ゼロパディングで10秒に伸ばす（1チャンク） |
| `> 10秒` | 先頭から10秒ずつ非重複で切り出し、残り3秒以上ならパディングして追加 |

`config.yaml`:
```yaml
preprocessing:
  sample_rate: 16000
  chunk_duration_sec: 10.0
  min_chunk_duration_sec: 3.0
  overlap_ratio: 0.0  # 0 = 重複なし、0.5なら50%重複
```

### コマンド例

```powershell
# 通常実行（既存チャンクはスキップ）
uv run python -m bird_fine.data.preprocess

# 既存を上書き
uv run python -m bird_fine.data.preprocess --overwrite
```

### 出力構造

```
data/processed/
├── Mallard/
│   ├── XC123456_chunk000.wav
│   ├── XC123456_chunk001.wav
│   ├── XC123457_chunk000.wav
│   └── ...
├── Common_Teal/
│   └── ...
└── chunks_index.csv          # 全チャンクのメタデータ
```

### chunks_index.csv の列

| 列名 | 意味 |
|---|---|
| species | 種名（フォルダ名と同じ） |
| xc_id | Xeno-canto録音ID |
| chunk_index | その録音内のチャンク番号（0,1,2...） |
| file_path | プロジェクトルートからの相対パス |
| duration_sec | チャンク長（常に10.0） |
| source_file | 元音声ファイル名 |

---

## 3. データ分割

### 何をするスクリプトか

`src/bird_fine/data/split.py` は chunks_index.csv を **train/val/test に分ける** スクリプト。

### Leakage防止の仕組み

**同じ録音から作られたチャンクは必ず同じsplitに入る** ようにする。これを「**録音ID単位split**」と呼ぶ。

```
NG例（チャンク単位split）:
  XC123456_chunk000 → train
  XC123456_chunk001 → test    ← 同じ録音！testで過大評価される

OK例（録音ID単位split）:
  XC123456_chunk000 → train
  XC123456_chunk001 → train   ← 同じ録音は同じsplit
  XC123457_chunk000 → test    ← 別録音
```

カモなどの鳴き声は **同じ録音内で似たフレーズが繰り返される** ため、
チャンク単位だとモデルが「録音の癖」を学習してしまい、テスト精度が見かけ上高くなる。
本番の未知音声での精度が低くなる原因。

### 層化分割

各種ごとに `train:val:test = 70:15:15` の比率を保つ。
1種で録音が少ない場合（例: 5件）でも、最低1件はtrainに、可能ならval/testにも分配。

### コマンド例

```powershell
uv run python -m bird_fine.data.split
```

### 出力構造

```
data/splits/
├── train.csv         # 全種のtrainチャンク
├── val.csv
├── test.csv
└── label_map.csv     # species名 → label_id (0〜7)
```

### label_map.csv の例

```csv
label_id,species
0,Common_Pochard
1,Common_Teal
2,Eurasian_Wigeon
3,Greater_Scaup
4,Mallard
5,Northern_Pintail
6,Northern_Shoveler
7,Tufted_Duck
```

アルファベット順でソート → label_idが振られる。学習・推論時は必ずこのmapを使う。

---

## 4. 運用上の注意

### データ件数の目安

| 状況 | 最低限 | 推奨 |
|---|---|---|
| 種あたりの録音数 | 5件以上 | 30件以上 |
| 種あたりのチャンク数 | 20件以上 | 100件以上 |

`--metadata-only` で件数を事前確認、明らかに足りない種はconfigから外す or `"A B"`品質に緩める検討を。

### データ不均衡

種ごとの録音数は大きくばらつく：

- **マガモ**: 数百件（人気種、世界中で録音）
- **スズガモ/ホシハジロ**: 数十件以下のことも

学習時の影響：
- macro F1で評価 → 少数派の精度も見える
- 必要なら `class_weights` を追加（[training_guide.md](training_guide.md) 参照）

### ライセンス

Xeno-cantoの録音は CC BY-NC-SA 等の条件付き。
- ✅ 個人研究、教育、非商用利用
- ❌ 商用配布、未クレジットの再公開
- 学習済みモデル公開時は録音者のクレジット要件を確認すること
