# 群分類器 作成プレイブック（標準フロー）

新しい分類群（duck / crow / 将来 gull …）の Stage2 細分類器を作るための標準手順と判断則。
**生きた文書**: 各群の実測で確定した知見を「追記ログ」に足していく。状態タグ: ✅確定 / 🔬検証中 / ⬜未着手。

最終更新: 2026-06-13 / 由来: duck（完成・運用中 `ast-duck-D-base-soup`）＋ crow（Phase1.5 進行中）。

> **コマンド表記の約束**: 本文中の `download.py` 等は**略記**。実体は段階型ドライバ
> `scripts/build_group.sh <stage>` か `uv run python -m bird_fine.<module>` / `tools/*.py`。
> 通しで回すときはドライバを使う（下記）。

---

## 実行: 段階型ドライバ `scripts/build_group.sh`（✅確定）

各段は独立に再実行可能・冪等寄り。**判断点ではドライバは止まって情報を出すだけで決めない**
（薄種補填 / grade緩和 / 複合クラス / OOD動作点 / 昇格可否 は人間が判断＝本 playbook の規律を守る）。

```bash
# 群固有値は config から導出: GROUP=config名, SPLITS=config の splits_dir, モデル名=ast-<group>[-<tag>]-<arm>-s<seed>
scripts/build_group.sh data  --config config-<group>.yaml                       # §2 収集→判断
scripts/build_group.sh prep  --config config-<group>.yaml [--grade-ablate "<種>"] # §3 前処理+split
scripts/build_group.sh train --config config-<group>.yaml --arm lean --seeds "42 1 2" [--tag AB --splits-dir <d>]  # §4 Lean
scripts/build_group.sh embed --config config-<group>.yaml [--tag AB]             # §4 Perch埋め込み+教師proba(lean s42 前提)
scripts/build_group.sh train --config config-<group>.yaml --arm kd --seeds "42 1 2" [--tag AB]                     # §4 Full(KD)
scripts/build_group.sh eval  --config config-<group>.yaml [--tag AB --eval-splits-dir <共通test>]                  # §4 録音単位CI
scripts/build_group.sh ood   --config config-<group>.yaml [--tag AB]            # §6 OOD閾値→人間が動作点選択
scripts/build_group.sh confusion --config config-<group>.yaml                   # §5 混同→複合は実測してから
scripts/build_group.sh register  --config config-<group>.yaml                   # §7 taxonomy追記スニペット出力
```
- `--dry-run` で発行コマンドのみ表示（実行前確認）。`PY` / `PERCHPY` 環境変数で venv 上書き可。
- `--tag`＝モデル名識別子（grade変種 AB 等）、`--splits-dir`＝学習データ（tag と独立）。grade ablation 時は `--tag AB --splits-dir data/splits-<group>-AB` のように両方指定。
- 学習は mainPC(GPU)、編集/commit は GT105（§8 配布規律）。

---

## 0. 設計思想
- **2段パイプライン**: BirdNET(Stage1) が群を検出 → 群専用 Stage2 が種/複合に精緻化。
- **group 汎用**: 統合(dispatch)は `species_taxonomy.yaml` の `stage2_model` 設定済み群を自動でトリガ対象にする
  （`BirdProject/stage2_refine.py`, 完成済）。**新群はモデルを作って taxonomy に1行足すだけで配線される**。
- **訓練側は `--config config-<group>.yaml` / `--splits-dir` で多群・多アーム化**（推論側 predict.py は無改修）。

## 1. 前提・足場（✅確定）
- `config-<group>.yaml`（duck の `config.yaml` / `config-crow.yaml` を雛形に、群固有部だけ差替）:
  `target_species` / `preprocessing.splits_dir`(群別) / `training.output_dir` / `model.num_labels`(=種数+other) /
  `other_class.tier*`(=対象外同科＋近接非同科)。前処理(3s/16k)・AST基盤・学習ハイパラは群間共有でよい。
- `data/species_master.csv` に群の種を登録（`group` / `status`=target|ood_tier1..3 / en_birdnet / ja / sci）。
  dispatcher のトリガ = `group一致 & status∈{target,ood_tier1}` の `en_birdnet`。
- **3秒固定チャンク**は運用制約（BirdNET 3s窓）。全群で遵守。

## 2. データ収集（✅手順確定 / 群ごとに実測）
- `download.py --config config-<group>.yaml`（Japan→worldwide フォールバック, grade `A B`）。
- **判断則（薄種対応, 優先順）**:
  1. **worldwide 補填**: フォールバックは Japan が**0件の時しか**発動しない。Japan が少数(>0)で止まった種は
     `--species "<種>" --worldwide-only --exclude-existing` で追加収集。← 最も効く（crow ハシボソ 2→102）。
  2. **grade 緩和**: それでも薄ければ `--quality "A B C"`（C以下）。**ただし基本不要**（✅crow 実測: A/A+B/A+B+C で
     clean-test f1 差なし, C は微減傾向）。**分離容易な群＋データ非枯渇では grade緩和は効かない**。field ノイズ頑健性は別途要評価。
  3. **複合化/縮小**: どうしても集まらなければ複合クラス（§5）か対象種から外す。
- **地域不在種**は worldwide のみ（crow ミヤマ/カササギは Japan ゼロ）。タクソンズレに注意
  （例: XC "Eurasian Magpie"=Pica pica 欧州種 ≠ 日本の Pica serica）。記録に残す。
- **混入チェック**: en 検索が別種を拾うことがある（crow Rook 配下に Phylloscopus 2件）。学名サブディレクトリを確認し除去。
- **grade 追跡**（重要・ハマりどころ）: chunks の `xc_id` は XC番号を持たないファイル名ステム。
  grade は `data/raw/<種>/metadata.csv` の `file-name` を **xcapi と同じ `_sanitize_filename`（`<>:"/\|?*`→`_`, 末尾`. `除去）**
  で `source_file` に突合して取る（これで100%結合, `tools/make_grade_ablation_splits.py: grade_map`）。

## 3. 前処理・分割（✅確定）
- `preprocess.py --config …` → 16k mono / 3秒チャンク。**data/raw 全ディレクトリを処理**（種=トップ階層名, ラベル）。
- `split.py --config …` → 録音単位 0.70/0.15/0.15 層化（`data/splits-<group>/`, `label_map.csv`）。
- **grade アブレーション**（任意）: `make_grade_ablation_splits.py --ablate <種> --out-prefix data/splits-<group>`
  → A / A+B / A+B+C の3 split を **val/test 完全共通**で生成（grade-B/C は train のみ）。

## 4. 学習・評価（✅レシピ / 🔬群ごとに比較）
- **既定 = Lean**: AST fine-tune（`train.py --config … --splits-dir … --output-dir … --seed …`）×複数seed → `soup_ast.py` で soup。
  分類ヘッドは `label_map` の種数に自動再初期化（config.num_labels は上限指定）。選択指標 `f1_macro`。
- **KD（Full）は条件付き**: Perch教師(OOF proba)→`--distill --kd-lambda1 --kd-temp2`→soup。
  - ✅duck知見: **KD-from-Perch は「弱い生徒(素CNN +0.148 有意)・seed分散低減」に効く**が、
    **天井近い AST 最終 soup では乗らない**（clean で base>KD 有意）。→ **既定 Lean、KD は弱種/低データ/Full比較で有意な時のみ**。
- **評価規律（最重要・厳守）**:
  - **録音単位 macro-f1 ＋ 録音クラスタ bootstrap CI**（`tools/probe_sweep/soup_ci.py`）。chunk単位点推定で優劣を断定しない。
  - **CI が0を跨ぐ差(≈±0.05)は「差なし」**。単一seedの +0.0XX を効果と解釈しない（必要なら soup）。
  - **リーク厳禁**: 候補モデルが学習で見た録音を test に入れない。⚠duck 事例: 旧運用 Cprod は test に45%リークで
    0.910 に膨張、リークフリー honest では 0.787。**昇格判定は必ず候補未学習の録音で**。

## 5. 混同分析・複合クラス（✅方針 / 🔬群ごと）
- 録音単位 混同行列（`analysis/confusion_audio.py`）で**分離不能ペア**を特定。
- 音響的に割れないペアは `<group>.display_groups` で**複合(slash)表示**（再学習ゼロ・最頻種を捨てない）。
  - ✅duck: カルガモ↔マガモは **Perch本体でも0/30＝本質的** → 「マガモ/カルガモ」複合（`tools/perch_native_confusion.py`）。
  - 群ごとに混同構造は違う（crow ハシブト↔ハシボソは要実測）。複合は**実測してから**。

## 6. OOD 閾値キャリブレ（✅手順）
- `ood_fp_audit.py --config config-<group>.yaml --ast-model <soup>`（OOD=対象外同科＋非同科, `data/ood_processed`）。
- 動作点 **「真<群>最優先・保持≥0.90」**で energy 閾値を導出（録音平均 energy）。
- ⚠**モデルを替えたら必ず再導出**（energy 分布がシフトする。duck: D-base-soup で 2.717→3.081, 旧値だと非カモFP0.88でゲート無効化）。
- 閾値は **Xeno-canto 域の暫定**。**デプロイ後フィールド再キャリブレ必須**（季節・現地ノイズで分布が変わる）。

## 7. 登録（自動配線・✅）
- `species_taxonomy.yaml` の `<group>.pipeline`: `stage2_model` / `energy_threshold` / `energy_temperature:1.0`。複合あれば `display_groups`。
- → 統合は group 汎用なので **process.py 無改修で dispatch 対象入り**。`predict.py --group <group> --json` で和名/複合/OOD棄却を検証。

## 8. 横断的な落とし穴（✅実体験）
- **`pgrep -f "パターン"` の自己マッチ**: SSHコマンド文字列自身がパターンを含むと、pgrep が自分のシェルにマッチして
  常時「実行中」の偽陽性。→ **`ps aux | grep X | grep -v grep`** か、スクリプト内 pgrep（自分は除外される）を使う。
- **venv の `bin/python` を `.resolve()` しない**: symlink を素の interpreter に解決し venv が無効化（site-packages 不可視→import失敗）。symlinkパスのまま叩く。
- **heredoc over ssh のクォート崩れ**: Python ワンライナを ssh 越しに書くと壊れやすい。**ファイルに書いて転送**してから実行。
- **git 配布**: 学習は mainPC(GPU)、編集/commit/push は **GT105(push ノード)**。mainPC はトークン無効→**bundle リレー**で追従
  （`git bundle create … A..B` → 転送 → `git fetch … && git reset --hard FETCH_HEAD`）。

## 9. 各群の状態
- **duck** ✅運用中: `ast-duck-D-base-soup`(honest 0.897), OOD 3.081, 複合=マガモ/カルガモ。BirdProject 統合済(settings既定disabled)。
- **crow** 🔬Phase3完了→Phase4へ: 4種(ハシブト164/ハシボソ102/ミヤマ97/カササギ100), **A+B lean soup 採用**(KD効果ゼロ・grade緩和不要・複合不要)。録音f1≈0.88(Carrion最弱0.80)。残=Phase4 OOD→Phase5 登録(crow)。
  → grade緩和/KD/複合 すべて不要が実測で確定。
- **gull** ⬜未着手（本 playbook ＋ config-gull.yaml で展開予定）。

---

## 追記ログ（確定知見をここに足す）
- 2026-06-13 初版。duck 完成＋crow Phase0-1.5 までの手順・判断則を集約。
- 2026-06-13 **crow grade ラダー結果（A / A+B / A+B+C, 単一seed, 共通test n=51 grade-A）**:
  録音単位f1 = A 0.903 / A+B 0.903 / A+B+C 0.881。ペア差は全て★差なし（A+B−A=−0.0001, A+B+C−A+B=−0.022 微減傾向）。
  → **§2 判断則に追記**: **分離容易な群＋データ非枯渇では grade緩和(B/C)は clean-test を改善しない**
  （crow 4種は声が明確に違い少データで天井 ~0.90。カモの「薄種は録音増やせ」と逆）。grade-C は微マイナス傾向＝ノイズ混入。
  **但し書き**: clean(grade-A)test 上の結論。**フィールドのノイズ頑健性は別問題**（B/C 学習が field で効く可能性→現地評価要, 設計書 domain gap）。
  単一seed・小nゆえ「差なし」は検出力にも依る。→ grade緩和の既定方針: **まず worldwide A+B、Cまでは基本不要**。
- 2026-06-13 **段階型ドライバ `scripts/build_group.sh` 新設**（再現性: 執行可能性の回収）:
  crow 専用ベタ書き `crow_full_pipeline.sh`（mainPCの`~/`・群ハードコード・git管理外）を、群名を `--config` から
  導出する汎用ドライバへ昇格。8段（data/prep/embed/train/eval/ood/confusion/register）をサブコマンド化し、
  判断点では止まって情報を出すだけ（自動化しない）。`--dry-run` 全段検証済、発行コマンドは今夜の crow バッチと一致。
  ＋ `config-template.yaml`（新群の差替箇所を TODO 化）を追加。**新群は config 作成→各段を順に回すだけ**で配線まで。
- （以後: crow Lean vs Full(KD) / 混同・複合 / OOD / 登録 / gull 展開 … を追記）
- 2026-06-14 **crow Lean vs Full(KD) 決着＋夜間ジョブ不具合の教訓**:
  夜間自動(schedule-run crow-full)は前半 grade ラダー無傷も、**後半KD枝が `teacher_proba-crow/val.npz` 欠落で全滅**。
  原因＝`crow_full_pipeline.sh` の教師proba export が `--splits train` 固定で、KD学習が監視に使う val 教師proba を出さず
  → KD学習 FileNotFoundError → 以降 HF Hub 401 / soup_ci npz欠落の連鎖。**根治: export を `--splits train val` に修正**。
  欠落 val.npz 再生成 → KD枝のみ再走で復旧。**結果(録音f1, n=51): lean-soup 0.8815 vs kd-soup 0.8833, 差+0.0018[−0.053,+0.056]=★差なし**。
  → **§4/§5 判断則: 天井近い群(crow)では KD は乗らない。KD価値は弱い生徒/seed分散低減に限定（duck で確認済の再現）。crow=lean soup 採用**。
  **教訓(§8 落とし穴)**: 教師proba export は **train(OOF)＋val(direct) の両方**が必須（KD学習の val監視用）。`--splits train` 単独は KD を静かに全滅させる。template/ドライバは val を必須にすること。
- 2026-06-14 **crow Phase3 混同分析（lean-soup, 録音単位 n=51）**:
  種別 f1: Magpie 0.960 / Rook 0.897 / Large-billed 0.870 / **Carrion 0.800(最弱)**。
  Carrion の誤りは Rook2/Large-billed2/Magpie1 と**3クラスに散逸**（precision 1.00＝他種は Carrion に化けない）。
  → **カモのカルガモ壁（非対称全崩壊）とは別物。crow に音響的に割れる種ペア無し → display_groups（複合）不要、4種そのまま出力**。
  Carrion の recall 0.67 は support15 の小標本振れ＋データ律速で、複合でなくデータ増でしか動かない（据え置き）。
