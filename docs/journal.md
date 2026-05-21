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

## 2026-05-21 run03 学習直後にシステム異常終了、成果物が破損（失敗ログ）

### やったこと

run03 本番学習（15 epoch 上限 / EarlyStopping patience=4）を実行。学習は epoch 6 で EarlyStopping により正常終了（best = epoch 2）。後日「実行状況をチェック」した際に、成果物の一部が破損していることに気づいた。

### 結果

- 学習プロセス自体は完走（7:36 開始 → 7:47:53 に最終 eval まで出力）。best モデル checkpoint-252（epoch 2, f1_macro 0.875）を取得。数値は experiments.md run03 参照
- Windows イベントログに **Kernel-Power ID 41**（7:48:29）と ID 6008（予期しないシャットダウン）。学習完走の約30秒後にシステムが異常終了し再起動（7:48:25 起動完了）
- 異常終了で**直前に書かれたファイルがゼロ埋め破損**していた:
  - root 成果物（`models/ast-duck-v3/` 直下）の `model.safetensors` / `config.json` / `label_map.csv` / `training_args.bin` が全滅
  - `checkpoint-756/` の `trainer_state.json` / `rng_state.pth` / `scaler.pt` / `scheduler.pt` がゼロ埋め
  - TensorBoard ログ（`runs/`）と `checkpoint-252` / `checkpoint-630` は無傷
- 復旧: best モデル checkpoint-252 は完全無傷（transformers で実ロードし推論まで確認）。root 成果物は checkpoint-252 と `data/splits/label_map.csv` からのコピーで再生成。**再学習は不要**

### なぜそうなったか

- 異常終了の原因は Kernel-Power 41 — クラッシュ / ハング / 電源遮断のいずれか。Windows Update の計画再起動（ID 1074）ではない
- ファイル破損のメカニズムは NTFS のライトバックキャッシュ。ファイルサイズ（メタデータ）は MFT にコミット済みだが、データブロックがディスクにフラッシュされる前に電源が落ちた → サイズは正常なのに中身が全ゼロのファイルになる。学習完走間際の数十秒（7:47:48〜53）に書かれたファイルだけがこの窓に該当した
- GPU 学習中〜直後は消費電力スパイクが大きい。**電源（PSU）容量が不足ぎみで、負荷ピークでクラッシュした可能性**がある

### 学び

- **学習完了 ≠ 成果物が安全**。プロセスが正常終了しても、OS のライトバックキャッシュ未フラッシュ分は不正シャットダウンで飛ぶ。学習直後はすぐ再起動・電源断を起こさない
- **checkpoint を複数残す運用が効いた**。`save_total_limit` で checkpoint-252 / 630 が残っていたため、root が全滅しても best から復旧できた。1 checkpoint しか残らない設定だったら詰んでいた
- **TensorBoard ログは別系統で残る**。`trainer_state.json` が飛んでも `runs/` から数値を復元できた。ただし最後の epoch 6 eval は tfevents 未フラッシュで欠損 → ログも完全ではない
- ゼロ埋め破損はファイルサイズが正常に見えるので、サイズだけでは気づけない。中身の非ゼロバイト数 / safetensors ヘッダの妥当性で判定する
- 既知の落とし穴「Win+Ctrl+Shift+B」とは別の異常終了。CUDA 文脈だけでなく**ディスク書き込み中の電源喪失**もリスクとして認識する

### 次にやる

- [ ] PSU 疑いの切り分け: 次の学習（run04）は GPU 負荷を下げて回す（batch size 縮小 / `nvidia-smi -pl` で電力上限を絞る 等）。落ちなければ電源容量が原因の傍証になる
- [ ] run04: weight_decay 単独評価（run03 の H1 検証分岐どおり）。lr=2e-5 / patience=4 据え置き、wd のみ変更

---

## 2026-05-21 run04 の設計を「weight_decay 単独評価」に確定

### やったこと

run03 の検証分岐（H2 不成立 → 別の正則化手段が必要）に従い、run04 を weight_decay 単独評価に確定。wd を 0.01 → 0.05 に強化し、experiments.md に学習前記録（事前登録）した。lr=2e-5 / patience=4 / SpecAugment オフは run03 から据え置き。

### なぜ wd=0.05 か

- 候補は wd=0.05 / 0.1 / 0.03。run02 が使った 0.1 は patience=2・SpecAugment と同時変更で切り分け不能だった値
- 0.01→0.1 は 10× ジャンプで、効いた/効かないの解釈が粗くなる。dose-response（用量反応）を追うなら一段ずつ上げたい
- 0.03 は弱すぎて overfit への効きが見えにくく、結論が出ない懸念
- → まず 0.05 で wd の効き方を見る。overfit タイミングが後ろにずれれば run05 でさらに調整、f1 が落ちれば 0.03 に弱める。分岐は experiments.md run04 に事前登録済み

### 学び

- run03 で「lr は f1 に効くが overfit タイミングには効かない」と切り分けられたので、run04 の狙いは f1 の絶対値ではなく **overfit タイミング（best_epoch）を後ろにずらせるか** に明確化できた。主軸指標が H1 に出ている
- run02 の失敗（同時多変更）以降、1 run 1 変更を徹底できている。run03→run04 も実質変更は wd のみ

### 次にやる

- [ ] run04 学習前に GPU 電力上限を下げる（管理者 PowerShell で `nvidia-smi -pl 150`）。PSU 疑いの切り分けも兼ねる
- [ ] dry-run（層化サンプリング修正済み）で v4 の import / end-to-end を確認 → run04 本番学習（15 epoch）
- [ ] 完了後 experiments.md の結果欄を埋めて `run04:` でコミット

---

## 2026-05-21 run04 結果 — wd は overfit タイミングに効く（切り分け成功）

### やったこと

run04（weight_decay 単独評価、0.01→0.05）を GPU 150W 制限下で実行。dry-run で v4 設定の end-to-end を確認してから本番学習。約17分で正常完走。

### 結果

- wd↑ で best_epoch が 2（run03）→ 5 に後退。overfit ピークが3エポック遅れた。f1_macro は 0.875 → 0.850 に低下。数値詳細は experiments.md run04
- 仮説判定: H1 成立 / H2・H3 不成立
- GPU 150W 制限下でシステム異常終了は再発せず

### なぜそうなったか / 切り分け

- run03 で「lr は overfit タイミングに効かない、別の正則化手段が要る」と分かっていた。run04 はその手段が wd かを検証 → **wd は効く**と確定
- ただし train_loss の急降下は run04 でも止まらない（epoch 4 で 0.002、run03 と同等以上に速い）。**訓練データの暗記速度は wd では変わらない**
- 変わったのは eval 側。run03 は eval f1 が epoch 2 ピーク後すぐ低下、run04 は epoch 5 ピークで以降 0.84〜0.85 に安定。**wd は「eval ピークの位置と安定性」に効く**
- → 「overfit」を train_loss の暗記速度と eval の劣化タイミングに分けて考えると、wd は後者だけに効く構造が見えた

### 学び

- 「overfit 対策」と一括りにせず、**train 暗記速度 / eval ピーク位置** を別指標として追うべき。lr も wd もこの2つに別々に効く（lr→f1 絶対値、wd→eval ピーク位置）
- wd↑ は overfit 後退と引き換えに f1 を削る。「効く/効かない」ではなく最適点を探す問題。run03(0.01)→run04(0.05) の dose-response を踏まえ run05 は中間を狙う
- PSU 疑いの傍証が1つ得られた（150W で落ちなかった）。ただし n=1、確証には継続観察が要る

### 次にやる

- [ ] run05: wd=0.03（run03 0.01 と run04 0.05 の中間）で単独評価。overfit 後退の利得を f1 コスト最小で取れる点を探す
- [ ] PSU 観察継続。次も 150W 制限で回し、落ちなければ傍証を積む

---

## 2026-05-22 run05 結果 — wd チューニングは天井を破らない（負の知見）

### やったこと

run05（weight_decay 中間点評価、0.03）を GPU 150W 制限下で実行。run03（wd=0.01）/ run04（wd=0.05）の中間として dose-response の単調性を検証した。

### 結果

- wd 0.01 / 0.03 / 0.05 → best f1 0.875 / 0.840 / 0.850。中間の 0.03 が最低で、dose-response は単調でなかった
- 仮説判定: H1 成立 / H2・H3 不成立
- GPU 150W 制限下で異常終了なし（PSU 観察2回目クリア）

### なぜそうなったか — メトリクスのノイズに気づいた

- run05 の eval f1 が epoch 4〜10 で 0.840 にほぼ完全固定だったのが手がかり。収束後の素の汎化性能がそのまま見えている状態
- 3 run を並べ直すと run04 も ep5〜9 で 0.843〜0.850 とほぼ平坦。run03 だけ ep2 に 0.875 の単発スパイクがあり、ep3〜5 は 0.82〜0.85
- → run03 の 0.875 は wd=0.01 の実力ではなく、early-stopping がノイズの山を1つ拾っただけ。3 run の真の水準はどれも ≈0.84〜0.85

### 学び（重要な負の知見）

- **early-stopping で選んだ best_f1 は単一エポックのノイズを拾う**。run を比較するならスパイクではなく eval 曲線のプラトー水準を見るべき。val 313 件では 1エポックの f1 は数サンプル分くらい簡単に振れる
- **wd（0.01〜0.05）は val f1 を実質動かさない**。3 run 費やして得た負の知見。lr も wd もこのデータでは ≈0.85 の天井を破れない
- train_loss は毎 run epoch 4 で ≈0.003、epoch 8 で ≈0.0001。訓練 2011 件は完全暗記される。ハイパラをいじっても暗記は止まらず、汎化天井も動かない
- **実験の打ち切り判断**: 「1 run 1 変更」を守って3 run 回したからこそ、wd 軸は無効と確信を持って言える。次は軸を変える

### 次にやる

- [ ] run03 / 04 / 05 の best モデルを test セット（338件、未使用）で評価。early-stopping バイアス抜きの比較で run03 0.875 が本物か確認（`evaluate.py`）
- [ ] wd チューニング打ち切り。0.85 天井を破る別レバーへ — SpecAugment のクリーン単独評価（run02 は同時変更で評価不能だった）、またはデータ量・質の改善

---
