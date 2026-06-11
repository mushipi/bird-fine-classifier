# OOD Detection: Energy-based Outlier Exposure による非対象種弾き

## 概要

カモ類8種を分類する AST Stage2 モデルは閉世界仮定で学習されており、カルガモ・
オカヨシガモ等の未知種や BirdNet の誤検出がそのまま8種のいずれかに分類される。
本ドキュメントは、この問題に対して採用した **Outlier Exposure + Energy スコア** に
よる OOD 検知手法をまとめ、将来の追加検証を可能にすることを目的とする。

---

## 問題の構造

```
BirdNet (Stage1)
  ↓ 「カモ類」と判定 ← ここで誤検出が起きうる
AST Stage2 (8種分類)
  ↓ softmax は必ず何かに分類する
誤った種名が出力される  ← これを防ぎたい
```

OOD 入力の発生源は主に2種類:
- **Tier1 (カモ科他種)**: カルガモ・オカヨシガモ等、BirdNet がカモ類として通す
- **Tier2/3 (水辺の他種)**: BirdNet の誤検出でたまに渡ってくる可能性

---

## 試みた3手法と結果

### 手法① Softmax Confidence 閾値（run10 モデル）

```
score = max(softmax(logits))
threshold: score < θ → unknown
```

| 指標 | 値 |
|---|---|
| AUROC | 0.838 |
| FPR≤5% での推奨 θ | 1.00（実質使用不可）|
| in-dist mean score | 0.955 |
| OOD mean score | 0.845〜0.863 |

**失敗理由**: softmax は logit を正規化するため、全 logit が小さくても
max 値が高くなる構造的問題（softmax overconfidence）。

---

### 手法② Energy スコア閾値（run10 モデル）

```python
energy = T * logsumexp(logits / T)   # T=1.0
# 高い = in-distribution / 低い = OOD
threshold: energy < θ → unknown
```

| 指標 | 値 |
|---|---|
| AUROC | 0.894 |
| FPR≤5% での推奨 θ | 10.29 |
| in-dist mean energy | 8.1 |
| OOD mean energy | 6.4〜6.9 |

**改善理由**: logit の絶対スケールを保持するため、OOD 入力で logit が均等に
小さい場合に自然にスコアが下がる。softmax に比べて +0.056 の AUROC 改善。

**残課題**: in-dist と OOD の分布が重なっており、FPR<5% を保てる実用閾値が存在しない。

---

### 手法③ Outlier Exposure + Energy スコア（run11 モデル）← 採用

OOD 種を "other" クラスとして学習に混ぜ、energy 空間を calibrate する。
分類精度（other_recall）は目的でなく、**energy 分布の分離**が目的。

```
学習データ: 8種 4921 chunks + "other" 1220 chunks（7種, recording分割, cap=30/rec）
  → WeightedRandomSampler で全9クラス均等サンプリング
  → CE loss weight = [1.0×8, 1.5×other]（固定値、Double Dipping なし）
  → SpecAugment は "other" クラスにのみ適用
  → metric_for_best_model = f1_macro_8class（8種のみ）
```

| 指標 | run10 | run11 | 差分 |
|---|---|---|---|
| AUROC (energy) | 0.894 | **0.932** | +0.038 |
| test f1_macro (8種) | 0.782 | **0.793** | +0.011 |
| val other_recall | - | 0.090 | ※目的外 |
| OOD mean energy | 6.7 | **6.2** | -0.5（分離改善）|

---

## 最終パイプライン

```
入力音声 (3s chunk)
    ↓
AST run11 → logits (9次元, 8種+other)
    ↓
energy = logsumexp(logits)
    ├─ energy < 10.35 → "unknown"（OOD 拒絶）
    └─ energy ≥ 10.35 → argmax(logits[:8]) → 8種の予測
```

### パラメータ管理

```yaml
# species_taxonomy.yaml
pipeline:
  energy_threshold: 10.35    # FPR≤5% での推奨値（ood_eval.pyで算出）
  energy_temperature: 1.0    # Temperature Scaling 用（現在はスケーリングなし）
```

### 再現コマンド

```bash
# OOD データ収集
uv run python -m bird_fine.data.download_ood

# OOD 評価（閾値算出 + AUROC）
uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-v11

# 推論（energy gate 込み）
uv run python -m bird_fine.inference.predict --audio path/to/audio.wav

# 推論（gate なし, 8種分類のみ）
uv run python -m bird_fine.inference.predict --audio path/to/audio.wav --no-ood-gate
```

---

## 設計上の重要な判断と根拠

### Double Dipping の回避

| NG | OK |
|---|---|
| Sampler で均等化 **かつ** Loss にデータ数ベース重みを掛ける | Sampler **または** Loss の片方だけ |

Sampler で均等化後にデータ数逆数ベースの Loss 重みを掛けると、
"other" の勾配が二重に補正されて8種の決定境界を破壊する。
Loss 重みは「Recall を上げたい意図」のための固定値 α=1.5 のみ。

### 条件付き SpecAugment

run06 の結果（8種全体に SpecAugment → test f1 -0.028）から、
8種の境界はマスクで壊れやすいことが判明済み。
"other" クラスは過学習防止のために波形多様性が必要なため、
`label == other_label_id` の条件分岐で "other" のみに適用。

### Outlier Exposure の正しい解釈

other_recall=0.090 は「分類器としての失敗」ではなく、Outlier Exposure 本来の
動作として正しい。目的は「OOD データを晒すことで energy 空間の logit 分布を
変化させること」であり、softmax 空間での分類精度は副産物に過ぎない。
AUROC energy 0.932 がこの目的の達成を示す。

### Energy スコアの符号規約

```python
# 本プロジェクトの convention（高い = in-distribution）
energy = T * logsumexp(logits / T)   # 正値

# 論文の convention（低い = in-distribution）
energy_paper = -T * logsumexp(logits / T)  # 負値

# 閾値 10.35 は本プロジェクト convention で calibrate されている
# 論文 convention での閾値は -10.35
```

---

## OOD テストセット仕様

`species_taxonomy.yaml` にて管理。`download_ood.py` で収集・前処理。

### Tier1（カモ科他種、最重要 OOD）

| 種 | 英名 | 音響的近縁 | BirdProject収録 |
|---|---|---|---|
| カルガモ | Eastern Spot-billed Duck | - | ✓ |
| オカヨシガモ | Gadwall | - | ✓ |
| トモエガモ | Baikal Teal | コガモに近縁 | ✓ |
| ヨシガモ | Falcated Duck | ヒドリガモと同属 | ✓ |
| ウミアイサ | Red-breasted Merganser | - | ✓ |
| カワアイサ | Common Merganser | - | ✗ |
| スズガモ | Greater Scaup | キンクロハジロと同属 | ✗ |

### Tier2（水辺の非カモ科）

| 種 | 英名 | BirdProject収録 |
|---|---|---|
| オオバン | Eurasian Coot | ✓ |
| カイツブリ | Little Grebe | ✓ |
| ハジロカイツブリ | Black-necked Grebe | ✗ |

### Tier3（コントロール）

チドリ類・シギ類・猛禽類等 10 種。詳細は `species_taxonomy.yaml` 参照。

---

## 未解決の検証ポイント（今後の追加検証候補）

### V1: 閾値の実フィールドデータでの再キャリブレーション

現在の閾値 10.35 は Xeno-canto のラボ録音（quality=A）で算出。
フィールド録音（環境音混入・低 SNR）では energy 分布がシフトする可能性。

```bash
# 実フィールド録音を集めて再評価
uv run python -m bird_fine.analysis.ood_eval --model-dir models/ast-duck-v11
# → species_taxonomy.yaml の energy_threshold を更新
```

### V2: Temperature T の最適化

現在 T=1.0（スケーリングなし）。T を上げると energy 分布が広がり
in-dist / OOD の分離が改善する可能性がある。

```bash
# T を変えて AUROC を比較
# ood_eval.py に --temperature オプションを追加して実施
```

### V3: Outlier Exposure データの多様性拡張

現在の "other" データは7種 1220 chunks。
多様性が不足している可能性があり、追加すると AUROC がさらに向上しうる。

```bash
uv run python -m bird_fine.data.download_ood --tiers 1 2
uv run python -m bird_fine.data.prepare_other_class
# → config.yaml の other_class.tier1/2_species を拡張してから実行
```

### V4: 実パイプラインでの end-to-end 評価

BirdNet Stage1 → AST Stage2 の連結評価。
Stage1 の誤検出率と Stage2 の OOD 弾き率の複合評価が必要。

### V5: run12 候補 — 8種の test f1 改善

run11 の test f1_macro_8class=0.793 は run05 の 0.838（10s チャンク）を下回る。
3s チャンク体制での 8 種精度改善（lr 調整・patience 拡大等）は別軸の課題。

---

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `species_taxonomy.yaml` | 種リスト・OOD tier 定義・閾値管理 |
| `src/bird_fine/data/download_ood.py` | OOD 音源収集・3s チャンク前処理 |
| `src/bird_fine/analysis/ood_eval.py` | AUROC 算出・閾値推奨・分布可視化 |
| `src/bird_fine/data/prepare_other_class.py` | "other" クラス学習データ準備 |
| `src/bird_fine/inference/predict.py` | energy gate 込みの推論 |
| `models/ast-duck-v11/` | Outlier Exposure 済みモデル |
| `docs/experiments.md` | run 別実験記録（run10/run11 参照）|
| `docs/journal.md` | 設計判断の経緯（Outlier Exposure 決定の項参照）|

---

## 参考

- Hendrycks et al., "Deep Anomaly Detection with Outlier Exposure", ICLR 2019
- Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
