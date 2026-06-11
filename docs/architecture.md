# Architecture

bird-fine-classifier の全体設計。
モデル構造・データパイプライン・OOD 検知・デプロイ方針をまとめる。

---

## 1. システム全体像

```
フィールド録音（マイク）
        ↓
  Stage1: BirdNET CNN
  （83種 or デフォルト 6000種）
        ↓
  Dispatcher
  species_master.csv の en_birdnet × group を参照
        ├─ group="duck" の種 → AST duck Stage2
        ├─ group="crow" の種 → AST crow Stage2  ← 実装予定
        └─ 該当なし → Stage1 の結果をそのまま記録
               ↓
         Stage2: AST Transformer
         （group ごとの fine-tune モデル）
               ↓
         Energy Gate
         energy = T * logsumexp(logits / T)
               ├─ score < threshold → "unknown"（OOD 拒絶）
               └─ score >= threshold → 種名を出力
```

### なぜ 2 段階か

BirdNET は広く浅く分類するため、近縁種（カモ類・カラス類等）の細分類が苦手。
近縁種に特化した Transformer fine-tune モデルを Stage2 として置くことで精度を補う。
BirdNET のラベルバイアス（欧州データ偏重でカラスをほぼミヤマガラスと出力する等）も
Dispatcher + Stage2 で吸収できる。

---

## 2. 種マスタ設計

### species_master.csv（主マスタ）

`data/species_master.csv` がシステム全体の唯一の種情報源。
iNaturalist API から定期的に更新し、実際にフィールドに出現する種を管理する。

| 列 | 内容 |
|---|---|
| `taxon_id` | iNaturalist taxon ID（sync で自動設定）|
| `sci` | 学名 |
| `en_inat` | iNaturalist 英名 |
| `en_birdnet` | BirdNET が実際に出力するラベル（Dispatcher のキー）|
| `ja` | 和名 |
| `family` / `order` | 科 / 目 |
| `obs_count` | 対象 bbox での research-grade 観察数 |
| `last_observed` | 最終観察日（sync で自動更新）|
| `status` | candidate / target / ood_tier1〜3 / ignore |
| `group` | duck / crow / null |
| `birdproject` | BirdProject CNN に収録済みか |
| `data_source` | xeno-canto / youtube |
| `notes` | 備考 |

**種のライフサイクル:**

```
iNaturalist 観察記録
        ↓ sync_species_master.py
status="candidate"（観察あり・ML 未割当）
        ↓ 手動でステータスを付与
target      → Stage2 が識別する種
ood_tier1   → Stage2 の近縁 OOD（訓練データに混ぜる）
ood_tier2/3 → OOD テスト用
ignore      → 監視不要
```

**master の更新:**

```bash
# 新規観察種を candidate として追記（既存 status は変更しない）
uv run python -m bird_fine.data.sync_species_master --dry-run  # 件数確認
uv run python -m bird_fine.data.sync_species_master            # 本実行
```

### species_taxonomy.yaml（補完設定）

モデルパス・energy 閾値・温度パラメータのみを保持。種リストは master から生成する。

```yaml
duck:
  pipeline:
    stage2_model: "models/ast-duck-v11"
    energy_threshold: 10.35   # FPR<=5% での推奨値
    energy_temperature: 1.0
crow:
  pipeline:
    stage2_model: null        # TBD
    energy_threshold: null    # TBD
    energy_temperature: 1.0
```

**Dispatcher のルーティングテーブルは動的生成:**

```python
triggers = master[
    (master["group"] == group) &
    (master["status"].isin(["target", "ood_tier1"]))
]["en_birdnet"].tolist()
```

---

## 3. AST モデル構造

### 入力仕様

BirdNET が 3 秒窓で処理するため、Stage2 も 3 秒チャンクを入力単位とする。

```
音声 (16kHz mono)
        ↓ ASTFeatureExtractor (max_length=304)
log-mel spectrogram (304, 128)
        ↓ Patch Embedding (16×16 パッチ)
350 パッチ + 2 special tokens = 352
        ↓ Transformer Encoder × 12
CLS token output (768次元)
        ↓ Classification Head
Logits (num_labels=9: 8種 + "other")
```

**位置埋め込みのリサイズ:**

事前学習モデル（10s 用: 1214 次元）を 3s 用（352 次元）に線形補間でリサイズする。
`resize_position_embeddings()` が `train.py` に実装済み。

```python
# ONNX エクスポート済み・数値誤差 < 1e-5 で確認済み
# models/ast-duck-v11/model.onnx (326MB)
```

### "other" クラスと Outlier Exposure

run11 で OOD 種を第 9 クラス（"other"）として学習に混ぜる Outlier Exposure を実施。
目的は「"other" を正確に分類すること」ではなく、**energy 空間をキャリブレーションすること**。

詳細は `docs/ood_detection.md` 参照。

---

## 4. OOD 検知（Energy Gate）

```python
# 推論時
logits = model(input_values=x).logits          # (batch, 9)
energy = T * torch.logsumexp(logits / T, dim=-1)  # T=1.0
# energy が高い = in-distribution（自信あり）

if energy < energy_threshold:
    return "unknown"   # OOD 拒絶
else:
    pred = logits[:, :8].argmax()  # "other" ニューロンは無視して 8 種で分類
    return species_name[pred]
```

| モデル | AUROC (energy) | 閾値 (FPR≤5%) |
|---|---|---|
| run10（OE なし） | 0.894 | - |
| **run11（OE あり）** | **0.932** | **10.35** |

---

## 5. データパイプライン

### 学習データ収集

```bash
# 1. メタデータ確認
uv run python -m bird_fine.data.download --metadata-only

# 2. 本 DL
uv run python -m bird_fine.data.download

# 3. OOD 用データ収集（Outlier Exposure 用）
uv run python -m bird_fine.data.download_ood
```

**データソース方針:**

- **primary**: Xeno-canto (quality=A, Japan → worldwide フォールバック)
- **fallback**: YouTube（日本録音が Xeno-canto で不足する種）
  - `yt-dlp` で取得 → 16kHz mono 変換 → 鳴き声区間を人力確認
  - `source="youtube"` を master の notes に記録して Xeno-canto と区別

### 前処理パイプライン

```bash
uv run python -m bird_fine.data.preprocess          # raw → 3s WAV chunks
uv run python -m bird_fine.data.split               # train/val/test 分割
uv run python -m bird_fine.data.prepare_other_class # "other" クラスデータ準備
```

| 設定 | 値 | 根拠 |
|---|---|---|
| sample_rate | 16kHz | AST の仕様 |
| chunk_duration | 3s | BirdNET の処理窓に合わせる |
| split 単位 | 録音 ID | チャンク単位だと leakage が発生する |
| train/val/test | 70/15/15% | - |

---

## 6. 学習設計

### 通常学習（8 種、Outlier Exposure なし）

```bash
uv run python -m bird_fine.training.train --dry-run  # 動作確認
uv run python -m bird_fine.training.train            # 本学習
```

主要ハイパーパラメータ（config.yaml）:

| パラメータ | 値 | 備考 |
|---|---|---|
| learning_rate | 2.0e-5 | Transformer fine-tune の標準 |
| weight_decay | 0.03 | run05 で最適点と確認 |
| fp16 | true | RTX 3060 Ti 向け |
| gradient_checkpointing | true | VRAM 8GB 節約 |
| metric_for_best_model | f1_macro | クラス不均衡に強い |

### Outlier Exposure 学習（9 クラス）

"other" クラスを追加する場合の追加設定:

```yaml
model:
  num_labels: 9
  init_from: "models/ast-duck-v10"  # 既存モデルのヘッドを拡張して初期化

training:
  metric_for_best_model: "f1_macro_8class"  # "other" に引きずられない

other_class:
  loss_alpha: 1.5  # CE loss の "other" 重み（固定値のみ、Double Dipping 禁止）
```

**Double Dipping 注意**: WeightedRandomSampler か Loss 重みかの二者択一。
両方使うと "other" の勾配が爆発して 8 種の決定境界が破壊される。

---

## 7. 推論

```bash
# energy gate 込みの推論（デフォルト）
uv run python -m bird_fine.inference.predict --audio path/to/audio.wav

# gate なし（8 種分類のみ）
uv run python -m bird_fine.inference.predict --audio path/to/audio.wav --no-ood-gate

# ONNX エクスポート検証
uv run python -m bird_fine.inference.export_onnx
```

---

## 8. デプロイ（Jetson Nano）

ONNX エクスポートは `run11` モデルで検証済み:

```
batch=1: max_abs_diff=8.3e-06  ✓
batch=2: max_abs_diff=4.4e-06  ✓（動的バッチ）
energy score diff=8.3e-07      ✓
ファイルサイズ: 326MB（FP16 化で約 160MB 見込み）
TracerWarning: SDPA の is_causal が定数化
    → 304 フレーム固定運用では無害
```

**次ステップ**: FP16 量子化 → TensorRT 変換 → Jetson Nano での速度測定

詳細は `docs/roadmap.md` 参照。

---

## 9. ファイル構成

```
bird-fine-classifier/
├── config.yaml                    # 学習・前処理ハイパーパラメータ
├── species_taxonomy.yaml          # パイプライン設定（モデルパス・閾値）
├── data/
│   ├── species_master.csv         # 種主マスタ（iNaturalist ベース）
│   ├── raw/{Species}/             # Xeno-canto 生音源（git ignore）
│   ├── processed/{Species}/       # 3s WAV チャンク（git ignore）
│   ├── ood/{tier}/{Species}/      # OOD 生音源（git ignore）
│   ├── ood_processed/{tier}/{Species}/  # OOD チャンク（git ignore）
│   └── splits/                    # train/val/test CSV + label_map
├── models/{run-name}/             # 学習済みモデル（git ignore）
├── outputs/                       # 評価結果・可視化（git ignore）
├── src/bird_fine/
│   ├── data/
│   │   ├── download.py            # Xeno-canto DL（target_species）
│   │   ├── download_ood.py        # OOD データ収集・前処理
│   │   ├── preprocess.py          # raw → 3s チャンク
│   │   ├── split.py               # train/val/test 分割
│   │   ├── prepare_other_class.py # "other" クラス学習データ準備
│   │   ├── dataset.py             # PyTorch Dataset（条件付き SpecAugment）
│   │   └── sync_species_master.py # iNaturalist → species_master.csv
│   ├── models/
│   │   └── ast_classifier.py      # ASTForAudioClassification ラッパー
│   ├── training/
│   │   ├── train.py               # DuckTrainer（WeightedSampler + OE 対応）
│   │   └── evaluate.py            # test セット評価
│   ├── inference/
│   │   ├── predict.py             # energy gate 込みの推論
│   │   └── export_onnx.py         # ONNX エクスポート検証
│   └── analysis/
│       ├── ood_eval.py            # OOD AUROC・閾値算出
│       └── confusion_audio.py     # 誤分類音声の可視化
└── docs/
    ├── architecture.md            # 本ファイル
    ├── data_guide.md              # データ収集・前処理の詳細手順
    ├── training_guide.md          # 学習・評価の詳細手順
    ├── ood_detection.md           # OOD 検知手法の詳細
    ├── roadmap.md                 # Jetson Nano 本番・横展開の課題
    ├── experiments.md             # 実験ログ（run01〜run11）
    └── journal.md                 # 設計判断の経緯
```
