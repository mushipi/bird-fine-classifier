# 開発ジャーナル

思考・失敗・気づきの時系列ログ。日付ごとにセクションを切り、`experiments.md` には書ききれない判断の経緯や学びを残す。

**書く対象:**
- 設計判断とその理由
- 失敗、ハマりポイント、再発防止策
- 仮説と検証の流れ
- ツール選定や運用方針の変更

**書かない対象:**
- run の数値結果そのもの（→ `experiments.md`）
- コードの説明（→ コード自体とコミットメッセージで読める）

---

## 2026-05-19 run02 で overfit 対策が裏目に出た（失敗ログ）

### やったこと

run01 baseline の overfit が明確だったので、改善のため以下を**同時に**適用して run02 を実行：

- `learning_rate`: 5.0e-5 → 2.0e-5
- `weight_decay`: 0.01 → 0.1
- `early_stopping_patience`: 3 → 2
- SpecAugment: なし → enabled (freq_mask 24×2, time_mask 80×2)

### 結果

`f1_macro` が **0.848 → 0.826** に悪化。ベスト epoch は 4 → 2 と前倒し（期待と逆方向）。
詳細: `docs/experiments.md` の run02 セクション参照。

### なぜ失敗したか

1. **4 つ同時に変えたせいで切り分け不能になった**。どのパラメータが寄与/阻害したか分離できず、次の手が打ちにくい
2. **`patience=2` が方針として矛盾**。lr を半減して学習を遅くしたなら、評価機会は減らすのではなく増やす（`patience=3〜4`）べきだった
3. SpecAugment のマスク強度（freq 19%, time 計16%）が AST にとって強すぎる可能性
4. **良かった点**: eval_loss は 0.622 → 0.365 に改善。確信度の校正自体は向上している（=正則化は効いている）

### 学び

- **正則化の同時多変更は禁じ手**。今後は「1 run 1 変更」を原則にする → CLAUDE.md に明文化
- ハイパラ変更時は**パラメータ間の整合性チェック**を変更前にやる（例: lr↓ なら patience↑）
- ハイパラを「主軸（lr, wd, patience）」と「補助（augmentation）」に分け、主軸先・補助後の順序で評価
- 失敗 run も baseline と並べて記録することで「やらない理由」の知見が蓄積される

### 次にやる

- [ ] `src/dataset.py` で SpecAugment が train_ds のみ適用されているか実装確認（eval に漏れていれば eval_loss 悪化の説明がつく）
- [ ] run03 案を 1 つに絞る:
  - **案A**: `lr=2e-5` + `wd=0.05`（控えめ）+ `patience=4`、SpecAugment オフ。主軸単独評価
  - **案B**: SpecAugment 単独（lr/wd/patience は run01 と同じに戻す）。補助単独評価
- [ ] どちらを先に走らせるかは A 推奨（主軸が大きく効く可能性が高いため）

---

## 2026-05-20 .gitignore が src/bird_fine/models を巻き込んで除外していた

### やったこと

初回コミット作成後、`train.py` を読んでいて `from bird_fine.models.ast_classifier import build_ast_classifier` の import 先が初回コミットに含まれていないことに気づいた。

### なぜそうなったか

`.gitignore` の `models/`（学習出力 `models/ast-duck/` を除外する意図）が、先頭スラッシュ無しのため**全階層の `models/` ディレクトリ**にマッチ。`src/bird_fine/models/` も巻き込み、モデル定義 `ast_classifier.py` / `__init__.py` が追跡対象外になっていた。`git check-ignore -v` で `.gitignore:30:models/` がヒットすることを確認。

### 結果

clone しただけでは `ImportError` で学習も評価も動かない壊れた状態だった。`models/` → `/models/` に修正（`data/raw` 等も同様にルート限定へ統一）し、`src/bird_fine/models/` を追跡対象に戻した。

### 学び

- `.gitignore` でディレクトリ名だけ書くと**深さに関係なく全マッチ**する。特定の場所だけ除外したいなら**先頭スラッシュ必須**
- 大容量ディレクトリ名（`models`, `data`, `outputs`）はソースのパッケージ名と衝突しやすい。除外パターンは最初からルート限定で書くべきだった
- 初回コミット後は「import 先がコミットに入っているか」を一度確認すると安全

### 次にやる

- run03 の準備に戻る（案A: lr=2e-5 + wd=0.05 + patience=4、SpecAugment オフ）

---

## 2026-05-20 run03 の設計を「lr 単独評価」に変更

### やったこと

run03 のハイパラ構成を決定し、条件・仮説を `experiments.md` に学習前記録（事前登録）した。当初案（2026-05-19 セクションの「次にやる」）は案A = `lr=2e-5 + wd=0.05 + patience=4` だったが、案B = lr 単独（wd は baseline 0.01 据え置き）に変更。

### なぜ変更したか

- 案Aは run02 直後、「SpecAugment が eval に漏れていないか（仮説d）」が未検証の段階で立てた案だった
- 今回 `dataset.py` / `train.py` を確認し、SpecAugment は train_ds のみ適用・漏れ無しと確定（仮説d 否定）
- これにより run02 の eval_loss 改善（0.622→0.365）は「正則化が素直に効いた結果」と確定。lr↓ も wd↑ も loss 方向としては正しかった
- 残る f1 悪化の容疑者は patience=2 の早期打ち切りと wd=0.1 の過剰。案A は wd=0.05 をそこにぶつけるが、lr と wd が同時に動き切り分けが甘くなる
- run02 最大の反省「複数同時変更で切り分け不能」を繰り返さないため、wd は baseline に固定し lr を単独評価する案Bを採用

### 学び

- 仮説（SpecAugment 漏れ）の検証結果が次の実験計画を変える。検証を後回しにせず先に潰すと設計がクリアになった
- 結果を見てからの後付け解釈（HARKing）を防ぐため、run03 から「仮説と予測を学習前に固定」する運用にした。experiments.md に学習前記録セクションを設ける

### 次にやる

- dry-run（`uv run python -m bird_fine.training.train --dry-run`）で models パッケージの import を確認
- 問題なければ本番学習 → experiments.md の結果欄を埋めて 1 commit

---

## 2026-05-21 dry-run が全クラスを検証できていなかった（気づきログ）

### やったこと

run03 本番学習の前に dry-run を実行しようとして、まず既存の `models/ast-duck-v3/` を確認したところ、中身が dry-run の残骸（1 epoch / 4 step、`eval_f1_macro=1.0`）だった。本番 run03 はまだ走っていない状態。

### なぜ気づいたか

dry-run の `eval_f1_macro=1.0` が不自然だった。再初期化直後の分類ヘッドを数ステップ学習しただけで満点が出るのは、8クラス分類として明らかにおかしい。

原因はサブセット抽出。`data/splits/train.csv` は species 順にソートされているため、先頭から flat に `.head(50)` で切ると**先頭1〜2種しか入らない**。dry-run が実質1〜2クラス分類になっていて、自明に満点を取っていただけだった。dry-run の「8クラスで end-to-end に動くか」という検証目的を満たせていなかった。

### 結果

- `train.py` の dry-run 抽出を `groupby("species").head(n)` の層化サンプリングに修正（commit `a0fdd60`）
- 旧 dry-run 残骸を削除し、修正版で再実行 → `train 48 / val 24`（8種 × 6 / 3）で全種カバー、`eval_f1_macro=0.1381`（8クラスのランダム水準 1/8≒0.125 付近）。これが正しい dry-run の姿
- 数値詳細は run の結果ではないので experiments.md には残さない（dry-run のため）

### 学び

- **dry-run の「通った」は中身を見て判断する**。終了コード 0 やメトリクスが出ること自体は検証になっていない。今回は f1=1.0 という「良すぎる値」が逆に異常のサインだった
- **層化前提のデータで flat な head/sample を使わない**。CSV がキー順ソートされていると、先頭スライスは特定クラスに偏る。サブセットを作るときは常に層化を意識する
- dry-run の出力先が config の `output_dir`（= `models/ast-duck-v3`）と同じため、dry-run のたびに本番ディレクトリへ残骸が残る。本番学習の前には毎回掃除が要る運用上の落とし穴

### 次にやる

- `models/ast-duck-v3/`（今回の dry-run 残骸）を削除 → run03 本番学習（15 epoch）
- 完了後 experiments.md の結果欄を埋めて `run03:` でコミット

---
