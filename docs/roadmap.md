# システムロードマップ: 本番運用と横展開

## 現在地

モデル学習フェーズは run11 で一区切り。
「未知音を弾きつつ正確に細分類するコア」が完成し、
システム開発・運用フェーズへ移行する。

```
完成したコア:
  BirdNet Stage1 → Dispatcher → AST Stage2 + Energy Gate
  Outlier Exposure により AUROC energy=0.932 を達成
  energy_threshold=10.35 で FPR≤5% を担保
```

---

## システム全体構成（Jetson Nano 本番環境）

```
flowchart TD
    subgraph Edge Device [Jetson Nano]
        A[マイク録音] -->|非同期キュー| B(BirdNET: Stage1)
        B --> C{Dispatcher}
        C -->|一般種| D[ログ出力・終了]
        C -->|カモ類・カモメ類| E(AST Model: Stage2)
        E --> F{Energy Gate}
        F -->|score < 閾値| G[未知・ノイズとして棄却]
        F -->|score >= 閾値| H[対象種として記録]
    end
```

**既存コードとの対応:**
- Dispatcher のルーティングテーブル → `species_taxonomy.yaml` の `pipeline.stage2_triggers`
- Energy Gate の閾値 → `pipeline.energy_threshold: 10.35`
- Stage2 推論 → `src/bird_fine/inference/predict.py`

---

## 課題1: エッジ実装とシステム統合（最優先）

### 1-1. モデル軽量化・高速化（ONNX / TensorRT）

PyTorch 生モデル（86M params）を Jetson Nano で毎秒回すのは非現実的。
FP16 での ONNX 変換 → TensorRT 最適化の検討が必要。

**✅ ONNX エクスポート検証済み（2026-06-01）**

懸念していた位置埋め込みリサイズ（`resize_position_embeddings` で `nn.Parameter`
を直接書き換え）の問題は発生しなかった。`export_onnx.py` で検証完了:

```
batch=1: max_abs_diff=8.3e-06  ✓
batch=2: max_abs_diff=4.4e-06  ✓（動的バッチ正常）
energy score diff=8.3e-07      ✓
ファイルサイズ: 326MB（FP16 化で約 160MB 見込み）
opset 17 / onnx.checker パス
```

```bash
uv run python -m bird_fine.inference.export_onnx
# → models/ast-duck-v11/model.onnx を生成
```

**⚠ 残る注意点**: SDPA の `is_causal` が入力形状依存でトレース時に定数化される
（`TracerWarning`）。3s チャンク固定（304 フレーム）運用では無害だが、
入力長を変える場合は再検証が必要。

**次のアクション:**
1. FP16 量子化（onnxruntime or polygraphy）でサイズ・速度を測定
2. TensorRT 変換（Jetson Nano 上で `trtexec`）
3. Jetson Nano での推論レイテンシ実測

### 1-2. メモリ競合管理

BirdNET（TFLite）と AST（PyTorch / TensorRT）を同時にメモリに乗せた場合の
スワップ・クラッシュリスクを評価する。

設計案:
- BirdNET は常駐、AST はカモ類検出時にオンデマンドロード
- または両者を同一プロセスで管理して VRAM 競合を回避

### 1-3. 非同期処理パイプライン

録音プロセス（I/O）と推論プロセスを分離し、
処理落ちが発生しても音声データを取りこぼさない Queue 設計。

```
録音スレッド → Queue（バッファ） → 推論スレッド
                                  ↑
                          推論が遅れてもここでカバー
```

### 1-4. Dispatcher 実装

`species_taxonomy.yaml` の `pipeline.stage2_triggers` を読んで
BirdNET の検出結果を AST に渡すか判断するルーティングロジック。

```python
# 骨格
triggers = taxonomy["pipeline"]["stage2_triggers"]
if birdnet_result["species"] in triggers:
    run_stage2(audio_chunk)
else:
    log_and_skip(birdnet_result)
```

---

## 課題2: 実環境でのキャリブレーション（デプロイ後）

### 2-1. Energy 閾値のフィールド再調整

Xeno-canto（studio quality）で設定した閾値 10.35 は、
フィールド（風・波・環境音混入）で下方シフトする可能性がある。

```bash
# 設置初期に実データを収集して再評価
uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-v11
# → energy_threshold を species_taxonomy.yaml に反映
```

詳細は `docs/ood_detection.md` の **V1: 閾値の実フィールド再キャリブレーション** 参照。

### 2-2. End-to-End 精度評価

AST 単体の精度ではなく、BirdNET の検知漏れ（False Negative）を含む
システム全体のパフォーマンス監視と評価手法の確立。

評価対象:
- Stage1 の種ごとの検知率
- Stage2 への誤ルーティング率（OOD が trigger されてしまう割合）
- Energy Gate の実フィールドでの FPR / TPR

### 2-3. 8種 F1 の改善（run12 候補 / 保留中）

3s チャーク移行で低下した分類精度（run05: 0.838 → run11: 0.793）のリカバリ。

**方針**: ラボデータで無理にチューニングせず、
ベランダで実際に録音された誤検知データが溜まってから
ファインチューニングで対応する（実データ優先）。

---

## 課題3: 他種群への横展開

「Outlier Exposure + Energy Gate」の型をカモメ類等に展開する際の課題。

### 3-1. Stage・年齢の網羅的サンプリング

幼鳥の乞食鳴き（begging call）は成鳥と音響特徴が大きく異なり、
distribution shift の原因になる。

対策: Xeno-canto ダウンロード時に `stage:juvenile` を意図的に含め、
train/val/test に均等に分散させる設計。

```python
# download.py の拡張案
quality_filter = "A"
stage_filter = None  # or "juvenile" を明示的に追加
```

### 3-2. 広帯域ノイズへの耐性強化

海岸沿い（カモメ類の生息環境）は波・風のピンクノイズが常在する。

対策候補:
- SpecAugment の強度調整（低周波帯のマスクを強化）
- 背景ノイズを重畳した augmentation データの作成

### 3-3. クラス統合（Lumping）の判断基準

大型カモメ類など音響的に識別困難な種は、
無理に分けず「大型カモメ類」として統合する運用判断が必要。

基準案:
- ood_eval.py の `ood_misclassified.csv` で種間混同率を確認
- 混同率 > 30% が続く種ペアはLumpingを検討

---

## 優先度マトリクス

| 課題 | 緊急度 | 重要度 | タイミング |
|---|---|---|---|
| ~~ONNX エクスポート検証~~ | - | - | ✅ 完了（2026-06-01）|
| FP16/TensorRT 変換 | 高 | 高 | Jetson 実装前に必須 |
| 非同期 Queue 設計 | 高 | 高 | Jetson 実装前に必須 |
| Dispatcher 実装 | 高 | 高 | Jetson 実装前に必須 |
| メモリ競合評価 | 中 | 高 | Jetson 実装時に確認 |
| Energy 閾値フィールド調整 | 低 | 高 | デプロイ後 初期運用中 |
| End-to-End 評価 | 低 | 高 | デプロイ後 |
| 8種 F1 改善 (run12) | 低 | 中 | 実データ収集後 |
| 他種群横展開 | 低 | 中 | カモ類運用安定後 |

---

## 関連ドキュメント

| ファイル | 参照タイミング |
|---|---|
| `docs/ood_detection.md` | Energy Gate の詳細・再現手順 |
| `docs/experiments.md` | run 別実験記録（run10/11 の詳細）|
| `species_taxonomy.yaml` | 種リスト・Dispatcher ルーティング・閾値 |
| `src/bird_fine/inference/predict.py` | Stage2 推論の実装（ONNX 変換の起点）|
