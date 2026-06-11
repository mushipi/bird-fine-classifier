# モデル比較検討: 現行AST vs AudioMAE vs Audio Mamba (SSAM)

> 作成: 2026-06-02 / ステータス: **検討のみ（実装未着手）**
> 目的: バックボーンを現行 AST から AudioMAE / Audio Mamba に差し替える価値があるかを、
> 導入容易性・期待効果・リスクの観点から評価し、実装可否の判断材料を提供する。
> 前提: 実装する場合は現行と同じ **全体fine-tune** プロトコルに合わせる。

---

## 0. 結論サマリ（先に読む人向け）

| 観点 | AST（現行） | AudioMAE | Audio Mamba (SSAM) |
|------|------------|----------|--------------------|
| 事前学習 | 教師あり (AudioSet) | 自己教師あり (MAE再構成) | 自己教師あり (Mamba + マスク再構成) |
| アーキ | ViT / self-attention (O(n²)) | ViT / self-attention (O(n²)) | Mamba / 選択的SSM (O(n)) |
| パラメータ | ~86M (ViT-B) | ~86M (ViT-B) | Tiny 4.8M / Small 17.9M / Base 69.3M |
| HF/transformers統合 | ◎ ネイティブ | △ timm移植版あり（後述） | ✕ 専用実装＋`mamba-ssm`必須 |
| 導入容易性 | ◎ | ○〜△ | ✕（CUDAカーネルビルド） |
| 全体fine-tuneの実績/レシピ | あり（本プロジェクト） | あり（原論文がfine-tune型） | **なし**（論文はprobe評価のみ） |
| 期待効果（本件） | 基準 (test f1 0.838) | 同等〜+α、転移は理論上有利 | 小データで優位の可能性、ただし未知数 |

**推奨**: 第一候補は **AudioMAE 全体fine-tune**（ViTで現行ASTからの差分が小さく、
timm移植チェックポイントが入手可能、原論文がそもそもfine-tune型なのでレシピが流用できる）。
Audio Mamba (SSAM) は `mamba-ssm` のビルドリスクと fine-tune レシピ不在のため
**第二候補（PoC/研究的位置づけ）**。

**ただし最重要の前提**: 本プロジェクトの実験履歴（run01〜09）が示すボトルネックは
**アーキテクチャではなくデータ品質・キュレーション**（長尺問題録音 XC488112/113、
juvenile の distribution shift 等）である。バックボーン差し替えだけで大幅改善は
期待しにくい。アーキ比較は「データ施策と独立した別軸の実験」として位置づけるべき。

> **【2026-06-02 追記・上記前提の部分的更新】**
> run11 の種別誤分類をメタデータ駆動で精査した結果、**弱点が2種類に分離**することが判明し、
> 「ボトルネックは一律データ」という上記の前提に**明確な反例**が見つかった。
>
> | 弱点の型 | 代表種 | 真因 | 解 |
> |---|---|---|---|
> | データ不在型 | Tufted juvenile | 世界に2録音で枯渇（XC/GBIF/iNat 全滅） | **解決不能**（データ追加・別ソース・拡張すべて行き止まり）|
> | **表現力不足型** | **Goldeneye `song`** | Eurasian_Teal の song と音響類似・train データは275chunk と十分 | **バックボーン強化が本命**（AudioMAE/SSAM）|
>
> 埋め込み距離分析（AST最終層 mean pool, cos）で、誤分類された Goldeneye song の **68%が
> train Teal song の方に近く**（正解 song は Teal 寄り0%）、train GE song↔train Teal song の
> 重心間 cos も **0.880** と高い。**データが十分でも現行 AST の表現力では song を分離できない**
> 種（Goldeneye）が実在する。これは「アーキ強化が本命の解」となるケースであり、本ドキュメントの
> 第一候補 **AudioMAE 全体fine-tune** を、データ施策の代替ではなく **Goldeneye song 分離という
> 定量的根拠を持つ本命施策**として位置づけ直す。詳細は `docs/journal.md` 2026-06-02 を参照。

---

## 1. 参照論文の正確な位置づけ（取り違え注意）

共有された論文は次のものである。

- **Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations**
  Sarthak Yadav, Zheng-Hua Tan, *Interspeech 2024*, pp.552–556.
  DOI: 10.21437/Interspeech.2024-1274 / コード: github.com/SarthakYadav/audio-mamba-official

重要な注意点が2つある。

1. **論文の主役は「Audio Mamba (SSAM)」であり「AudioMAE」ではない。** 両者は別モデル。
   - **AudioMAE** = "Masked Autoencoders that Listen" (Huang et al., NeurIPS 2022, Meta)。
     ViT に MAE（マスク自己符号化）を適用した音声モデル。論文では Table 1 の比較
     ベースライン（86Mパラメータ）の一つとして登場するのみ。
   - **SSAM** = 本論文の提案手法。Mamba（選択的状態空間モデル, SSM）ベース。

2. **論文の評価プロトコルは「バックボーン凍結＋単層MLPプローブ」（HEARベンチマーク方式）**
   であり、本プロジェクトの **全体fine-tune** とは異なる。
   - 論文は事前学習済みモデルから固定長の特徴ベクトルを抽出し、その上に1隠れ層の
     MLP（1024ニューロン）だけを学習して10タスクで評価している。
   - したがって論文の集約スコア s(m) や「SSAM が SSAST 比 +20pt超」といった主張は、
     **全体fine-tune を行う本プロジェクトへ直接転用できない**。傾向の参考に留める。

### 論文 Table 1 の関連数値（参考）

| モデル | 事前学習データ | パラメータ | 位置づけ |
|--------|--------------|-----------|---------|
| AudioMAE | AudioSet | 86M | 強力なSSLベースライン |
| SSAST (公式) | AudioSet+LibriSpeech | 89M | Mambaの直接比較対象 |
| SSAM-Tiny / Small / Base | AudioSet | 4.8M / 17.9M / 69.3M | 提案手法 |

論文の主張（probe前提）:
- SSAM は同等規模の SSAST を集約スコアで大きく上回る（+20pt超）。
- **少ない事前学習データでも優位**（10%データ時の差が最大）、**小型モデルでも優位**。
- 入力長・パッチサイズの変化に SSAST より適応的（Table 2/3）。
- ただし MW-MAE（非causal）には及ばず、Mamba を MAE 枠組みに入れるのは future work。

**注意**: 論文の下流10タスクに鳥の細分類は含まれない（ESC-50, Speech Commands,
NSynth Pitch, FSD50K 等）。最も近いのは環境音 ESC-50。鳥 fine-grained への転移は未検証。

---

## 2. 3モデルの技術比較（軸別）

### 2.1 入力仕様の差（前処理への影響大）

| 項目 | AST（現行） | AudioMAE | SSAM |
|------|------------|----------|------|
| サンプリングレート | 16 kHz | 16 kHz | 16 kHz |
| メルビン数 | **128** | **128** | **80** |
| 窓 / ホップ | 25ms / 10ms | 25ms / 10ms | 25ms / 10ms |
| 事前学習入力長 | 10s (1024 frame) | 10s (1024 frame) | 2s |
| パッチ | 16×16, stride(10,10) | 16×16 (非重複) | 可変 (4,16)/(4,8) 等 |
| 本プロジェクト実入力 | 3s / 304 frame ※ | （要設定） | （要設定） |

※ 現行は `config.yaml: feature_extractor_max_length=304`、`models/ast-duck-v10/config.json:
num_mel_bins=128, patch_size=16, max_length=304`。チャンクは `chunk_duration_sec=3.0`。
（補足: ルート `CLAUDE.md` の「10秒チャンク」は古い記述。現行configは3秒が正。）

含意:
- **AST→AudioMAE**: メルビン数 128 で一致。前処理（メル特徴）の互換性が高い。
  ただし AudioMAE は10s/1024frame前提なので、3s入力に合わせるなら位置埋め込みの
  補間が必要（現行 `resize_position_embeddings()` と同種の処理）。
- **AST→SSAM**: メルビン数が **128→80** に変わるため、メル特徴抽出（feature extractor）
  の作り直しが必要。`ASTFeatureExtractor` がそのまま使えない。

### 2.2 計算量・VRAM適合性（RTX 3060 Ti, VRAM 8GB）

- AST / AudioMAE ともに ViT-B (~86M)。現行 AST が fp16 + gradient_checkpointing +
  effective batch 16 で学習できている実績があるため、AudioMAE もほぼ同条件で載る見込み。
- SSAM は **Tiny 4.8M / Small 17.9M / Base 69.3M** と選択肢が広く、VRAM 余裕は大きい。
  Mamba は系列長に対し線形計算量で、長い入力ではViTより有利（ただし本件は短尺3s）。

### 2.3 事前学習パラダイムと転移の理論的相性

- **AST（教師あり/AudioSet）**: AudioSet のラベル分布（人間生活音中心）に最適化。
  鳥 fine-grained とはドメイン粒度が乖離。
- **AudioMAE（自己教師あり/再構成）**: ラベルに縛られず低レベルのスペクトル構造を学ぶ。
  小ラベルデータでの転移に理論上有利。原論文はAudioSet/ESC-50/SpeechCommandsで
  fine-tune時に強い性能。
- **SSAM（自己教師あり/Mamba）**: 同上＋論文上「小データで優位」。ただし probe 評価での話。

---

## 3. 導入容易性と移行コスト（実装観点）

### 3.1 AST（現行・基準）
- transformers ネイティブ。`ASTForAudioClassification` + `ASTFeatureExtractor`。
- 追加依存なし。最も容易。

### 3.2 AudioMAE
- **HF transformers には未収録**だが、**timm互換の移植チェックポイントが存在**:
  - `gaunernst/vit_base_patch16_1024_128.audiomae_as2m`（AudioSet-2M自己教師あり事前学習）
  - `gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k`（AS-20kでfine-tune済み）
  - `hance-ai/audiomae` 等のコミュニティ実装も存在
  - Meta公式: `github.com/facebookresearch/AudioMAE`（ViT-B、論文の本家）
- 必要作業（半カスタム）:
  1. timm でバックボーン読み込み → 分類ヘッド付与（`num_labels`）
  2. メル特徴抽出を自前実装（128 mel・正規化統計をチェックポイントに合わせる）
  3. 10s/1024frame前提 → 3s入力に合わせた位置埋め込み補間（現行ロジックを移植）
  4. HuggingFace Trainer ループに載せる（Trainer自体はモデル非依存なので流用可）
- 追加依存: `timm`（純Python、ビルド不要）。導入リスクは低い。

### 3.3 Audio Mamba (SSAM)
- **`mamba-ssm` が必須**（選択的スキャンのCUDAカーネル）。`causal-conv1d` も併用。
  - 要件: Linux / NVIDIA GPU / PyTorch 1.12+ / CUDA 11.6+。本環境（CUDA 12.4ビルドの
    torch 2.4, Python 3.12, RTX 3060 Ti）は要件を満たすが、**ビルドは難所**。
  - 注意: `pip install mamba-ssm causal-conv1d --no-build-isolation` が定石。
    **uv 運用との相性**（`--no-build-isolation` 相当の扱い）を事前検証する必要がある。
- チェックポイントは著者公式リポジトリ（github.com/SarthakYadav/audio-mamba-official）。
- **全体fine-tune用の分類ヘッド・レシピは論文・公式に提供されていない**（probeのみ）。
  ハイパラ（lr, layer-wise decay, warmup等）を自前で探索する必要がある。
- メルビン数80・2s入力前提なので、前処理を作り直す必要。

**導入容易性まとめ**: AST ≫ AudioMAE > SSAM。

---

## 4. 現行コードの再利用可否マッピング

調査対象: `src/bird_fine/` 配下。

### 流用可（モデル非依存）
- `data/download.py` — Xeno-canto ダウンロード
- `data/preprocess.py` — 16kHz / 3sチャンク化（メルビン数に依存しない波形処理）
- `data/split.py` — 録音ID単位の層化分割（train/val/test）
- `training/train.py` の Trainer ループ本体・`WeightedRandomSampler`・
  `DuckTrainer.compute_loss`（"other"重み付きCE）・`make_compute_metrics`（f1_macro_8class）
- `training/train.py` の `expand_classifier()`（分類ヘッド拡張）— ヘッド形状次第で流用可
- `inference/predict.py` の Energy Gate による OOD 検知（logits依存、バックボーン非依存）
- `inference/export_onnx.py`（ただしSSAMは独自演算でONNX化に難あり）

### 要差し替え（AST固有）
- `models/ast_classifier.py` — モデルロード（`ASTForAudioClassification` → 各モデルのローダ）
- `data/dataset.py` の `ASTFeatureExtractor` — メル特徴抽出
  - AudioMAE: 128mel維持だが正規化統計・入力長を合わせる
  - SSAM: **80mel** に変更が必要 → feature extractor 全面作り直し
- `training/train.py` の `resize_position_embeddings()` — 新モデルのパッチ/埋め込み形状に
  合わせて再実装（AudioMAEはViTなので近い実装で済む。SSAMはSSMで位置埋め込みの扱いが異なる）

### 注意点
- メルビン数や入力長が変わると、`expand_classifier()` 以外の **前処理〜入力テンソル形状**が
  連鎖的に影響を受ける。特に SSAM（80mel/2s）は影響範囲が広い。
- AudioMAE は 128mel を共有するため、AST資産の再利用度が最も高い。

---

## 5. 期待効果のリアルな評価（鳥 fine-grained × 小データ）

本プロジェクトの実験履歴（ルート `CLAUDE.md` / `docs/experiments.md`）が示す事実:

- run01 baseline は f1_macro 0.848（epoch4）で**明確な過学習**。
- 正則化・SpecAugment・録音追加・チャンク上限など各種介入（run02〜08）は
  いずれも現行ベスト run05（test f1 **0.838**）を超えられていない。
- run08/09 の分析で、誤分類は**特定録音に異常集中**（XC197026 juvenile、XC349677、
  長尺 XC488112/113）。真因は「chunks数」ではなく**特定録音が学習させる音響パターンの幅**
  と distribution shift（juvenile）と判明。

ここから導かれる見立て:

1. **ボトルネックはアーキテクチャではなくデータキュレーション**。バックボーンを
   強くしても、決定境界を歪める録音や distribution shift は残るため、**劇的改善は
   見込みにくい**。
2. 一方で自己教師あり事前学習（AudioMAE/SSAM）は、AudioSet教師ラベルに縛られず
   低レベル音響特徴を学ぶため、**過学習しやすい小データでの汎化に理論上は有利**。
   現行の過学習傾向に対し改善余地がある可能性はある。
3. SSAM の論文上の強み（小データ・小型モデルで優位）は本件の小規模データと相性が
   良い**可能性**があるが、**probe結果でありfine-tuneでの再現は保証されない**。

**現実的な期待値**: 同等〜+1〜2pt程度、要ハイパラ再調整。データ施策（録音キュレーション、
juvenile対処）と独立した別軸の実験として扱うのが妥当。アーキ変更を「本命の改善策」と
見なすべきではない。

---

## 6. 推奨と判断基準

### 推奨ランキング（導入容易性 × 期待効果 × リスク）

1. **AudioMAE 全体fine-tune（第一候補）**
   - ViTで現行ASTからの構造差が小さい / 128mel共有で前処理再利用度が高い /
     timm移植チェックポイント入手可 / 原論文がfine-tune型でレシピ参照可。
   - 自己教師あり事前学習による過学習耐性に期待。
2. **Audio Mamba (SSAM)（第二候補・PoC）**
   - 小データ優位の理論的魅力はあるが、`mamba-ssm`ビルド・fine-tuneレシピ不在・
     80mel前処理作り直し・ONNX化困難とリスクが多い。研究的探索として位置づける。

### 実装フェーズへ進む際の Go/No-Go チェックリスト

実装に着手する前に、以下を**読み取り/小規模検証で確認**してから本実装に進む:

- [ ] **チェックポイント入手**: AudioMAE は `gaunernst/...audiomae_as2m` を timm でロードできるか
      （SSAMは公式リポジトリの重み形式とライセンスを確認）
- [ ] **依存導入**: AudioMAEは `uv add timm` で済むか / SSAMは `mamba-ssm` + `causal-conv1d`
      が uv 環境でビルド成功するか（`--no-build-isolation` 相当の扱いを要検証）
- [ ] **VRAM実測**: 新モデルが fp16 + gradient_checkpointing + effective batch 16 で
      RTX 3060 Ti (8GB) に収まるか（OOMしないか）小バッチで確認
- [ ] **入力整合**: メルビン数（AudioMAE=128 / SSAM=80）・入力長・正規化統計を
      チェックポイント仕様に合わせられるか
- [ ] **A/B再現条件**: 後述の比較設計どおり、同一split・同一指標・同一seedで
      現行ASTを再走でき、公平比較できるか

いずれかが No なら、そのモデルは見送り or 後回し。

---

## 7. 実装フェーズの概略（参考・未着手）

全体fine-tune での公平な A/B 比較設計:

1. **データ固定**: `data/splits/{train,val,test}.csv` をそのまま使用（録音ID単位分割を維持）。
2. **指標固定**: `f1_macro_8class`（"other"に引きずられない既存8種F1）を主指標に、
   現行 `make_compute_metrics` を流用。
3. **seed固定**: `random_seed=42` を含め乱数条件を揃える。
4. **比較対象**: ①現行AST（再走 baseline, 期待 test f1≈0.838）vs ②AudioMAE全体fine-tune。
   余力があれば ③SSAM。
5. **run管理**: 既存ルールに従い `models/<run-name>/` を分け、`docs/experiments.md` に
   学習前事前登録（仮説・予測値の固定）→ 学習後に結果追記。`docs/journal.md` に経緯。
6. **1 run 1 変更の原則**: バックボーン差し替えは大きな変更なので、ハイパラは現行値を
   起点に最小限から始め、必要なら layer-wise LR decay 等を段階導入。

---

## 8. 参考文献・リンク

- Yadav & Tan, "Audio Mamba: Selective State Spaces for Self-Supervised Audio
  Representations", Interspeech 2024. DOI 10.21437/Interspeech.2024-1274
- 公式コード: https://github.com/SarthakYadav/audio-mamba-official
- Huang et al., "Masked Autoencoders that Listen", NeurIPS 2022. arXiv:2207.06405
- AudioMAE 公式: https://github.com/facebookresearch/AudioMAE
- AudioMAE timm移植: https://huggingface.co/gaunernst/vit_base_patch16_1024_128.audiomae_as2m
- mamba-ssm: https://github.com/state-spaces/mamba / https://pypi.org/project/mamba-ssm/
- 現行AST: `MIT/ast-finetuned-audioset-10-10-0.4593`
