# Architecture

bird-fine-classifierの全体設計、ASTモデルの仕組み、主要な設計判断とその理由をまとめる。

---

## 1. 2段階分類パイプラインの位置づけ

```
┌──────────┐    ┌────────┐    ┌──────────┐    ┌────────┐
│ 環境音声 │ → │BirdNet │ → │ カモ類? │ → │  AST   │ → マガモ/コガモ/...
└──────────┘    └────────┘    └──────────┘    └────────┘
                  CNN          上位ゲート     Transformer
                                                 (本PJ)
```

### なぜ2段階構成か

BirdNet（汎用CNN、6000種以上の対応）は**広く浅く** 分類するため、近縁種（カモ類・カモメ類・カラス類）の細分類が苦手。
これらの近縁種に対しては：

1. **音響特徴が似ている**（共通の鳴き方）
2. **十分なクラス内バリエーション**（地域差、個体差、コール/ソング）
3. **クラス間の差が微細**（ピッチ、テンポ、フォルマント）

一方、対象カテゴリ（カモ類）に絞れば数十時間〜数百時間の事前学習を共有するAST（AudioSet 200万音声で学習済み）の方が圧倒的に有利。

### 学習目的の観点

このPJは**Transformer fine-tuneの体感** も狙いに含む。CNN（BirdNet）との比較で：

| 観点 | CNN (BirdNet) | AST (本モデル) |
|---|---|---|
| 受容野 | 局所→積層で広がる | 全パッチ間のattention |
| 周波数依存 | フィルタが局所的 | グローバルな帯域関係も学習 |
| 事前学習 | 鳥音特化 | 汎用AudioSet（200万音声） |
| パラメータ | ~数百万 | 86M |
| 推論コスト | 低 | やや高 |

---

## 2. AST (Audio Spectrogram Transformer) の仕組み

### 入力

- **音声波形**（16kHz mono, 任意長）
- 内部で**log-mel spectrogram**（128 mel bins）に変換
- 標準入力は10秒 → メルスペクトログラム shape `(1024, 128)` 程度

### モデル構造

```
[Audio 16kHz mono]
        ↓
[ASTFeatureExtractor]
   - Mel spectrogram (128 bins)
   - Mean/Std正規化（AudioSet統計）
        ↓
[Patch Embedding]
   - 16×16のパッチに分割
   - Linear射影 → 768次元
        ↓
[CLS token + Positional Embedding]
        ↓
[Transformer Encoder × 12]
   - Multi-Head Self-Attention (12 heads)
   - FFN, LayerNorm
        ↓
[CLS token output]
        ↓
[Classification Head (fine-tune対象)]
        ↓
[Logits (num_labels=8)]
```

### 事前学習の利用

`MIT/ast-finetuned-audioset-10-10-0.4593` は：

- ImageNetのViT重みで初期化
- AudioSet（200万音声、527クラス）でfine-tune
- **音響特徴の表現力が既に高い** → 少ないデータで転移学習が効きやすい

本PJでは**分類ヘッドのみ初期化**して、Transformer本体は事前学習重みから学習を始める：

```python
ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=8,
    ignore_mismatched_sizes=True,  # ヘッドのshapeが変わるため
)
```

---

## 3. データパイプライン設計

```
Xeno-canto API
    │
    │ xcapi.QueryBuilder / XenoCantoClient
    ▼
data/raw/{Species}/XC*.mp3
    │
    │ librosa.load(sr=16000, mono=True)
    │ chunk into 10s windows (zero-pad if <10s)
    ▼
data/processed/{Species}/XC*_chunk*.wav
    │
    │ split by xc_id (録音ID単位、leakage防止)
    ▼
data/splits/{train,val,test}.csv + label_map.csv
    │
    │ DuckChunkDataset → ASTFeatureExtractor
    ▼
PyTorch DataLoader → Trainer
```

### 設計判断の根拠

| 判断 | 採用 | 不採用案 | 理由 |
|---|---|---|---|
| 入力長 | **10秒固定** | 5秒/20秒/可変 | ASTの事前学習仕様（10-10）に合わせる |
| サンプリングレート | **16kHz** | 22.05/44.1kHz | AST仕様、鳥音の主要帯域も十分 |
| split単位 | **録音ID** | チャンク単位 | チャンク単位だと同録音のtrain/test混入でleakage |
| データ層化 | **種ごとに分割比率維持** | 全体ランダム | 不均衡データでvalがゼロになる種が出るのを防ぐ |
| パディング | **ゼロパディング** | リフレクション/反復 | シンプル、ASTのattentionが無音を自動で軽視 |

---

## 4. 学習設計

### 戦略：分類ヘッドのみ初期化 + 全層fine-tune

| 戦略 | 採用 | 不採用 | 理由 |
|---|---|---|---|
| 全層fine-tune | ✅ | head-onlyフリーズ | データ少なくてもAdamW + warmup + 低lrで安定。鳥音は事前学習(AudioSet)から離れた領域なので全層適応が必要 |
| 学習率 | 5e-5 | 1e-4以上 | Transformer fine-tuneの標準値、より大きいと事前学習を壊す |
| Warmup | 10% | なし | 序盤の勾配爆発を防ぐ |
| 混合精度 | fp16 | bf16/fp32 | RTX 30系でfp16が高速、VRAM節約 |
| Gradient Checkpointing | ✅ | OFF | VRAM 8GBではbatch=4でもメモリ際どい |

### 最適化指標：f1_macro

- accuracyだとクラス不均衡で多数派（マガモ）のスコアに引きずられる
- F1 macroは全種を等しく評価 → 少数派の性能も反映

---

## 5. 拡張性

将来的に以下の拡張を想定：

| 拡張 | 必要な変更 |
|---|---|
| **対象種追加**（10種〜） | config.yamlの`target_species`に追加、再DL、再学習 |
| **カモメ類モデル追加** | このPJを丸ごとコピー、対象種だけ差し替え |
| **データ増強**（SpecAugment等） | Datasetクラスに前処理ステップを追加 |
| **BirdNet統合** | `inference/`に「BirdNet出力 → 本モデル委譲」ロジックを追加 |

---

## 6. 既知の制約

| 制約 | 影響 | 緩和策 |
|---|---|---|
| データ少量（種あたり〜100件） | 過学習リスク | EarlyStopping, weight_decay, augmentation |
| クラス不均衡 | 少数派の精度低下 | クラス重み or オーバーサンプリング（未実装、必要時追加） |
| 録音条件のバイアス | 場所/機材で識別される可能性 | 多様な録音源を含める、推論時の品質を吟味 |
| VRAM 8GB | batch_size制約 | grad_accum + fp16 + checkpointing |
