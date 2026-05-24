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

## 2026-05-22 test 評価で val 順位が逆転 — モデル選択バイアスの発見

### やったこと

run05 所見の宿題どおり、run03 / 04 / 05 の best モデルを未使用の test セット338件で評価（`evaluate.py --model-dir`）。

### 結果

- test の f1 順位は val の真逆。val 最良の run03（0.875）が test 最低（0.775）、val 最低の run05（0.840）が test 最高（0.838）
- run03 の val−test gap は −0.100、run05 は −0.002。数値詳細は experiments.md「test セット評価」セクション

### なぜそうなったか

- `load_best_model_at_end` は全エポックの val f1 の最大値を取る。val 313件ではチャンク単位 f1 が ±0.02〜0.03 揺れ、複数エポックの最大を選べばその上振れを必ず拾う = 上方バイアス
- run03 は epoch 2 の f1 スパイク（0.875）を掴んだ。だが epoch 2 のモデルは学習途中で、test では 0.775 しか出ない。スパイクの分だけ val が過大評価されていた
- run05 は val 曲線が平坦（7エポック 0.840）。掴むべき「山」が無く選択バイアスが乗らないので、val（0.840）≈ test（0.838）
- gap が best_epoch と連動（2→5→6 で gap −0.100→−0.044→−0.002）するのも符合。早期に止まったモデルほど未学習で test が悪く、早期のスパイクほど偶然性が高い

### 学び（重い方法論の知見）

- **run01〜05 を val f1 で比較していたこと自体が誤りだった**。比べていたのは「ノイズの最大値を選んだ、バイアスの乗った数値」。run03 を baseline 超えの最良とした判断は test で覆った
- **early-stopping / best-model 選択は、選択指標が noisy だと「最良」でなく「最も運の良い」モデルを選ぶ**。val を小さくするほど、評価頻度を上げるほどこのバイアスは強まる
- 今後の鉄則: run 比較とモデル採用は **test（selection に使っていない hold-out）で行う**。val は学習中の早期終了判定に使うだけと割り切る
- 「1 run 1 変更」で3 run 回して dose-response の非単調に気づき（run05）、その違和感を test で潰せた。違和感を放置せず検証に回す流れは機能している

### 次にやる

- run05 を現行ベストとして扱う（test f1 0.838）
- 改善の主軸はハイパラでなくデータ: 少数種（Tufted_Duck n=10 等）の追加収集 / class-weighted loss、SpecAugment のクリーン評価
- モデル選択の仕組み自体の見直し（val ピーク依存からの脱却）も課題として残す

---

## 2026-05-22 ハシビロガモはなぜ強い? —「少数種=データ不足」が思い込みだった

### やったこと

test 評価のあと「ハシビロガモは test 14件と少数なのに F1 0.923。なぜ?」という疑問が出た。種ごとの train データ量（録音数・チャンク数）を実際に数えた。

### 分かったこと

- Tufted_Duck（キンクロハジロ）は **trainチャンク数 694 で全種最多**。test 評価セクションに「少数種のデータ不足」と書いていたが、これは誤りだった
- カラクリは「1録音あたりのチャンク数」。Tufted_Duck は録音20本に対し 694 チャンク（1録音35チャンク）。長い録音を細切れにしただけで、実質の多様性は20シーン分しかない
- Northern_Shoveler（ハシビロガモ）は trainチャンク最少72だが録音31本。多様な録音から学べて F1 0.923 / precision 1.000
- test F1 は train チャンク数でなく **train 録音数** と連動。録音20本の2種（Tufted_Duck / Eurasian_Wigeon）が test 最下位2つ、録音30本以上の6種は 0.84〜0.95
- さらに Tufted_Duck は 694 チャンクの突出でチャンク不均衡を起こし、モデルが過剰予測（precision 0.269、コガモ14・ヒドリガモ5を誤って吸着）

### なぜ思い込んだか

test n=10 が最少 → 反射的に「学習データも少ない」と結論した。test n（15% split のチャンク数）と train データ量は別物。録音単位 split では、長い録音がどの split に落ちるかで test n と train チャンク数は独立に動く。

### 学び

- **「support が小さい＝弱い種」ではない**。効くのは録音の多様性。test n の小ささと train データ量を混同しない
- test 評価で「val を疑え」と書いた直後に、自分は train データ量を数えずに思い込んだ。**疑う対象は val だけではない。自分が触っていない数字は一度すべて数える**
- 修正: experiments.md の test セクションと記事ドラフト（qiita_draft.md）の該当箇所を「録音数の多様性 + チャンク不均衡」に書き換えた

### 次にやる

- 録音数の少ない種（Tufted_Duck / Eurasian_Wigeon、各20録音）の音源を追加収集
- チャンク不均衡の是正（1録音あたりチャンク数の上限、または class-weighted loss）

---

## 2026-05-24 run06: SpecAugment 単独投入は失敗 —「train 暗記の抑制」は汎化を保証しない

### やったこと

run02 で評価不能だった SpecAugment を、現行ベース（run05）から 1 変数だけ flip して切り分け。config.yaml で `spec_augment.enabled: false → true`（freq 24×2 / time 80×2、wd/lr/patience は run05 据え置き）。事前登録（`docs/experiments.md` run06 セクション）で H1〜H4 と分岐を学習前に固定。学習完了後 `evaluate.py --model-dir models/ast-duck-v6 --no-attention` で test 評価。

### 結果

- val f1 0.846（run05 比 +0.006）/ **test f1 0.810（run05 比 −0.028）** / best_epoch **2**（run05 は 6）
- 仮説検証: H1a 成立（train_loss @ ep4 が run05 の 0.003 → 0.025、10倍に上振れ）/ H1b **不成立**（best_epoch は ≥7 予測に対し 2 に早期化）/ H2 **不成立**（test ≥0.85 予測に対し 0.810）/ H3a 僅か不成立（gap 0.036）/ H4 部分成立（Tufted_Duck +0.05、Eurasian_Wigeon ≈0）
- 種別: Tufted_Duck 0.389→0.438（+0.05）と狙いどおり改善も、**Northern_Shoveler が 0.923→0.733（−0.19）で大きく悪化**。全体下落の主因はここ

### なぜそうなったか

- SpecAugment は train の暗記を確かに抑えた（H1a 成立）。だが val 曲線は平坦化せず「epoch 2 にスパイク → 以降緩やかに悪化」型に変わった。`load_best_model_at_end` は最大値を掴むので **未学習に近い checkpoint-252（epoch 2）が選ばれた**
- 結果として val−test gap が 0.002（run05）→ 0.036（run06）に拡大。**val ピーク選択バイアスを引き戻している**。run03（best_epoch=2, gap=0.100）と同じ症状で、対策のはずの SpecAugment が選択バイアスを増幅した格好
- 少数種を救うはずが、効いたのは Tufted_Duck だけ（+0.05）。一方で Northern_Shoveler（録音31本、チャンク72の中堅）が −0.19。SpecAugment による周波数/時間マスクが、訓練データの少ない中堅種のクラス境界を曖昧化した可能性

### 学び（重い方法論の知見）

- **train_loss の抑制 ≠ 汎化向上**。「完全暗記の阻止」を成功の代理指標にできない。H1a 成立 + H2 不成立 の組み合わせがこれを直接示している。「過学習対策」と書かれた手法でも、test で改善するとは限らない（むしろ best_epoch が早期化して悪化することがある）
- **正則化系の手法は val 曲線の形状を変える** → `load_best_model_at_end` との相性が悪い。曲線が「平坦なプラトー」型から「初期スパイク + 緩やかな悪化」型に変わると、選択バイアスが復活する。run03 → run05 で gap が縮んだのは val 曲線が平坦化したからで、SpecAugment はその逆方向に作用した
- **事前登録 → 結果のサイクルは機能している**: H2 を「test f1 ≥0.85」で固定していたから、val が微増しても「失敗」と即断できた。事前登録なしなら「val ちょっと上がったし採用?」と判断を曇らせていた可能性
- **「補助レバー単独で 0.85 天井を破れる」期待は捨てる**: ハイパラ3 run（lr/wd/wd 中間点）+ SpecAugment 1 run、4 run 回して全て天井を破れず。残るレバーはデータ側（録音追加、不均衡是正）とモデル選択方式の改善

### 次にやる

- **データ軸に完全に切り替える**: Xeno-canto から Tufted_Duck / Eurasian_Wigeon の録音追加。チャンク数でなく「録音数」を増やすことが目的
- **モデル選択方式の見直し**: `load_best_model_at_end` の val 単点ピーク依存をやめ、val f1 の移動平均 / 上位 k checkpoint の平均 / val loss 併用 などを検討。run03 と run06 で踏んだ同じ罠（best_epoch=2 → test 大幅劣化）を構造的に潰す
- SpecAugment 再評価は録音追加後に回す。ベース性能が上がった状態で別物として測る

---

## 2026-05-24 run07 準備: 弱2種に quality=B から +10 録音追加 — preserve-existing で test 固定

### やったこと

run06 の結論「データ軸へ切替」を実行。Tufted_Duck / Eurasian_Wigeon（各 train 録音20本）に Xeno-canto から quality=B の録音を +10 ずつ追加。test 338 件を run01〜06 と同一に保つため、split.py に `--preserve-existing` モードを実装。

### 設計判断（事前検討）

**Quality 緩和の対象**: 既存 raw は quality=A のみで Tufted 29本 / Wigeon 30本が上限。これ以上 A で増やせない。「弱2種だけ quality=B を許可」を採用。全種で A+B にすると比較公平性は出るが既存 splits が崩れる代償が大きい。「弱い種だけ品質基準が低い」というバイアスは付くが、test 数値を直接比較するメリットを優先。

**Split 戦略**: 既存 val/test を完全固定し、追加録音は全て train に振る。preserve-existing モードを `split.py` に新設。理由: test 数値を run01〜06 と直接比較したいから。再 split すると val/test 構成が変わり比較が壊れる。

**録音数**: 「他種中央値まで（+10）」を採用。Tufted 29 + 10 = 39 / Wigeon 30 + 10 = 40 で他種の下位レンジに揃う。+200本など大量追加は弱2種が逆方向の多数派になり、また quality=B noise が大量に混じる懸念。

**ベース**: run05（現行ベスト、SpecAugment off）に揃える。run06 で SpecAugment は不採用確定（test 0.810 < 0.838）したものを残す理由はない。1 変数変更原則。

### 実装

`src/bird_fine/data/download.py`:
- `--quality` で config.yaml の quality を上書き
- `--exclude-existing` で `metadata.csv`（実 DL 済み）の XCID をスキップ。`metadata_only.csv` は対象外（候補メタは「DL 済み」ではない）
- `--worldwide-only` で countries フィルタを外して worldwide 単独検索（追加収集は地域多様性を取りたいので Japan→worldwide の fallback ロジックを切る）
- 取得録音は XCID 昇順で安定ソート → 再現性確保

`src/bird_fine/data/split.py`:
- `--preserve-existing` で既存 train/val/test の (species, xc_id) を読み込み、chunks_index.csv の新規録音だけ train に追加

### 実行

```
uv run python -m bird_fine.data.download --species "Tufted Duck" "Eurasian Wigeon" \
    --quality B --exclude-existing --worldwide-only --max-per-species 10
uv run python -m bird_fine.data.preprocess
uv run python -m bird_fine.data.split --preserve-existing
```

### 追加された XCID（再現性のため記録）

- **Tufted_Duck (+10)**: XC32831, XC34072, XC96339, XC97432, XC111191, XC138103, XC138253, XC243761, XC244007, XC251919
- **Eurasian_Wigeon (+10)**: XC28030, XC37499, XC83875, XC83876, XC88827, XC92776, XC96338, XC110737, XC111211, XC143305

### 結果（split 状態）

```
train: 2082 chunks / 311 recordings  ← 元 2011 / 292 から +71 chunks / +19 unique_xcid
val:    313 chunks /  61 recordings  ← 完全同一（run01〜06）
test:   338 chunks /  69 recordings  ← 完全同一（run01〜06）

Tufted_Duck train: 694 chunks (20 xc_id) → 728 chunks (29 xc_id) ※
Eurasian_Wigeon train: 154 chunks (20 xc_id) → 191 chunks (30 xc_id)
```

※ Tufted_Duck の追加分のうち XC32831 と XC34072 は preprocess.py の `_extract_xc_id` の既知バグでどちらも xc_id=`Aythya` に集約（XCプレフィクスのないファイル名 `Aythya fuligula ...mp3` に対して stem.split()[0] フォールバックが効くため）。チャンクは10録音分すべて train.csv に入っており**学習データとしては +10録音**。xc_id ベースのカウントだけ +9 になる。

### 既知の課題（TODO）

- **preprocess.py の `_extract_xc_id` バグ**: XC プレフィクスのないファイル名で先頭単語が同じ録音同士が同一 xc_id に集約される。修正案: metadata.csv の `file-name` → `id` マップを読み込んで正しい XCID をルックアップ。既存 splits の xc_id 体系と整合性が崩れるため、修正は run07 完了後に別案件として実施
- 全種で衝突を調査済み: Tufted_Duck 1件のみ。他種は no_XC_prefix ファイルがあっても偶然先頭単語が分散していて衝突なし

### 学び

- **「既存 splits を維持して追加録音を train だけに足す」運用は test 比較性を守るために重要**。再 split は seed=42 でも追加データで shuffle 順序が変わり、結果として val/test 構成が全部変わる
- データ追加は **コードのコミットと、データ追加実行ログ（XCID 一覧）の docs コミット**の 2 段階で残す。data/raw は git 管理外なので、再現性は XCID リストとコマンドで担保する

### 次にやる

- run07 学習 → test 評価 → 事前登録した H1〜H5 で検証 → 結果コミット
- preprocess.py の `_extract_xc_id` バグ修正は run07 完了後に別案件で

---

## 2026-05-24 run07: 録音 +10 では 0.85 天井を破れず — チャンク不均衡が真のボトルネック

### やったこと

弱2種に quality=B 録音を +10 ずつ追加し（test を完全固定したまま）、run05 と同条件で学習・test 評価。事前登録で H1〜H5 を固定。

### 結果

- val f1 0.8446 (epoch 4) / **test f1 0.8268** / acc 0.870
- run05 比: val +0.005 / **test −0.011（悪化）**
- 仮説検証: H1〜H4 不成立、H5a だけ成立（val−test gap 0.018）
- 種別: Tufted_Duck +0.043（小幅改善）、Wigeon ±0、Mallard −0.029、Northern_Shoveler −0.108

### なぜそうなったか

**Tufted_Duck の過剰予測は未解決**:
- run05 では Tufted と予測した26件中正解7件（precision 0.269）。run07 では27件中正解8件（precision 0.296）。**recall は 0.700→0.800 で +1件 正解増えただけ**で、誤吸引のパターン（Eurasian_Teal を 14件、Wigeon を 4件 Tufted と誤判定）は run05 と同一
- 原因は **train チャンク数の不均衡**。Tufted_Duck は train 694→728 で全種最多のまま、他種 (72〜534) との差がさらに開いた。softmax の事前分布が Tufted に偏る構造は録音追加では緩和されない
- つまり「録音多様性 ↑」と「チャンク不均衡悪化」が同時に起き、後者が前者の効果を打ち消した

**Northern_Shoveler −0.108 はサンプル数バイアス**:
- test n=14 で正解 −1（12→11） / 他種からの吸引 +2 件 の小さな絶対数変化
- 「F1 大幅悪化」の見た目だが、小サンプル種の F1 を絶対値で語るのは罠
- ただし precision 1.000 → 0.846 への低下は「他種からの吸引増」を意味し、Tufted の過剰予測の余波の可能性

**Mallard −0.029 は本物**:
- test n=46 で run05 0.918 → 0.889。precision が 0.830 に低下で他種からの吸引が増えた可能性
- 弱2種に集中した録音追加が、強い種の境界を曖昧化する副作用を示唆

### 学び

- **「録音追加」と「チャンク不均衡是正」は同時にやるべきだった**: 録音を増やすと自動でチャンクも増える。録音多様性向上の正の効果が、チャンク不均衡悪化の負の効果に相殺される。次は **1録音あたりチャンク上限**で先に均すか、両方同時に介入する
- **F1 単独で「成功・失敗」を判断する罠**: Northern_Shoveler −0.108 は数字としては大きいが、混同行列の絶対数は「正解 −1 / 吸引 +2」。 **小サンプル種の F1 は混同行列とセットで読む**。今後の所見では F1 だけでなく precision/recall の内訳と絶対数を必ずチェック
- **事前登録の予測値は厳しめが正解**: H2a で Tufted ≥ 0.50 と置いたが、実測 0.432 で「微増だが予測未達」と即断できた。予測値を緩めにすると「ちょっと改善したから採用?」と判断を曇らせる。今回の H2a / H2b / H3 はそれぞれ run05 から +0.11 / +0.04 / +0.13 の改善を要求していた → 全て未達で「録音 +10 では足りない」が明確に出た
- **データ軸でも 0.85 天井は破れない**: ハイパラ系（run02〜06）+ データ系（run07）の5 run で計5本の弾を撃ち、全てで run05 を超えられない。これは「8種カモ分類で AST が学べる上限がこの辺り」を示唆する可能性も。チャンク不均衡是正で本当に動くかが次の正念場

### 次にやる

- **run08 はチャンク不均衡是正単独評価**: 1録音あたりチャンク上限（例: 30）を設けて Tufted の 728 chunks を ≈600 まで均す。録音は追加せず data の使い方だけ変える。Tufted の過剰予測が precision でどこまで改善するかを見る
  - 注意: val/test もチャンク数が変わると preserve-existing と矛盾するので、**train のチャンクのみサブサンプリング** する方針が筋
- **class-weighted loss は run08 と並行検討**: chunk 不均衡是正と効果が重なるので、まず chunk 上限で素の効果を測る方が筋
- preprocess.py の `_extract_xc_id` バグ修正は引き続き TODO

---

## 2026-05-24 run08 準備: Tufted の長尺2録音だけ clip して chunk 不均衡を是正

### やったこと

`src/bird_fine/data/cap_train_chunks.py` を新規実装。`train.csv` のみ xc_id 単位で chunks 数の上限を設けて seed 固定でランダムサブサンプリング。val/test は触らない。

cap=100 で実行: **Tufted_Duck train 728 → 361 chunks（−367）**。他種は全て不変。

### 設計判断

**なぜ「1録音上限」か**:
train.csv の chunks 分布を全種で取って気づいた:
- Tufted_Duck の上位2録音 XC488113 (297 chunks ≈49分) と XC488112 (270 chunks ≈45分) で **Tufted total 728 の 78%** を占めている
- 他種の最大は Mallard XC396538 の 64 chunks、Wigeon XC110737 の 49 chunks など
- 「Tufted_Duck はチャンクが多い種」というより、**「2つの異常に長い録音を含む種」** だった

これは「録音数20本→多様性不足」の問題とは別の、データ収集時の構造的偏り。録音の長さが均一でない以上、chunks ベースで均す＝1録音上限を設けるのが筋。

**なぜ cap=100 か**:
- cap=30 だと Mallard (top1=64) や Wigeon (top1=49) も削れて変数が増える → 単独評価にならない
- cap=50 だと Mallard だけ軽く削れる → 変数2つ
- **cap=100 だと Tufted のみ介入**。Mallard 64, Wigeon 49 は影響なし → 完全な単独評価
- cap=100 で Tufted total 728→361 となり、Mallard 390 と並ぶ。class-imbalance（chunks ベース）はほぼ解消

**なぜ録音追加（run07 の状態）を残したまま上限を入れるか**:
- run07 で「録音追加 + チャンク追加（副作用）」が打ち消し合うのが分かった
- run08 で「録音追加状態のまま chunks だけ均す」と、録音多様性向上の正の効果だけ残す形になる
- これは run07 と run08 の差分が **純粋にチャンク不均衡是正の効果** になる

### 実装

`cap_train_chunks.py`:
- `groupby(['species', 'xc_id'])` で xc_id 単位に分け、cap を超える行を `random.Random(seed).sample` でクリップ
- `--dry-run` で差分のみプレビュー
- 上書き保存（git で履歴は残る）

確認:
```
Tufted_Duck after cap: 361 chunks, 29 records
top10:
   100  XC488113-2018-07-29  ← 297 から clip
   100  XC488112-2018-07-31  ← 270 から clip
    27  XC730568-Fuligule    ← 影響なし
    （以下中央値前後は全部影響なし）
records preserved: 29/29
```

長尺2録音だけが clip され、録音数は全保持。Tufted の **「異常な2録音による嵩増し分」が消えた状態**。

### 仮説の置き方の工夫

run07 で「H4 不成立（他種への波及）」が出た反省を活かし、run08 では H6 で「cap は他種データを変えていないので H6 達成は当然視」と明記したうえで、それでも学習ダイナミクスの間接波及で他種が悪化する可能性を残している。

「Tufted の過剰予測抑制 → Eurasian_Teal が改善」が運動学的に成立するはずの仮説（H5）も別途立てた。run07 では Tufted と予測した27件中 Eurasian_Teal が14件含まれていたので、Tufted を予測しなくなれば Teal が正しく予測される件数が増えるはず。

### 次にやる

- run08(pre) コミット → 学習 → test 評価 → 結果コミット
- 結果に応じて run09 を分岐: cap=50 を試す / class-weighted loss に切り替え / 別軸へ

---

## 2026-05-24 run08: Tufted chunks 半減でも precision 0.269 で完全同値 — 仮説が崩れた

### やったこと

Tufted_Duck の train chunks を cap=100 で 728→361（−50%）に削減し、他種は不変のまま学習・test 評価。事前登録で H1〜H7 を固定。

### 結果

- val f1 0.8381 (epoch 6) / **test f1 0.8203** / acc 0.864
- run05 比: val −0.002 / **test −0.018（悪化）** / run07 比 test −0.007
- 仮説検証: H3 / H5 / H7 のみ成立、主仮説 H1 / H2 / H4 はいずれも不成立
- 種別: Tufted **±0.000**（run05 と完全同値）、Eurasian_Teal +0.015、Northern_Pintail +0.033、Northern_Shoveler −0.077、Mallard −0.069、Wigeon −0.049

### 衝撃のなぜそうなったか

**Tufted_Duck の予測パターンが run05 と統計的にほぼ識別不能**:

| | run05 | run08 |
|---|---|---|
| Tufted と予測 total | 26 | 26 |
| Tufted 正解 | 7 | 7 |
| Teal 誤吸引 | 14 | 13 |
| Wigeon 誤吸引 | 5 | 4 |
| Goldeneye 誤吸引 | 0 | 2 |
| precision | 0.269 | 0.269 |

train chunks を **−50%** 削っても Tufted の precision が小数第3位まで完全一致した。誤吸引の構成もほぼ同じ。つまり run08 の Tufted モデルは **run05 と同じ決定境界を学んでいる**。

### 真の原因への仮説

chunks 不均衡説が崩れた以上、別の説明が必要:

1. **音響的類似性**: Tufted_Duck と Eurasian_Teal / Eurasian_Wigeon の鳴き声が AST の表現空間で区別困難。同じ Aythini / Anatinae で類似した call を持つ可能性
2. **モデル容量の上限**: AST-base（86M params）の表現力で区別できない細かい音響差。AST-Large や別アーキテクチャでないと識別できない
3. **train data の質**: Tufted の29録音の中身が「Tufted 単独」ではなく Teal や Wigeon の声が背景に混じっている可能性。run07 で追加した quality=B 録音にこの混入があるかも

どれが真かは混同行列の音声を実際に聞いてみるか、分光図を並べて見ないと分からない。

### 学び（重い方法論の知見）

- **「不均衡 = 過剰予測」仮説の検証可能性**: chunks を半減して precision が動かなかったことは、不均衡仮説への極めて強い反証。**1 run で仮説そのものを潰せた**のは事前登録 + 単独介入の威力。仮説を多重化していたら「他の効果と相殺された」で逃げる余地が残った
- **「介入していない他種」が動く**: H6 で Mallard −0.069 / Wigeon −0.049。「Tufted の chunks しか変えていない」のに他種が大幅に動いた = 学習ダイナミクスは局所介入でも全体に波及する。「介入対象以外は固定」と仮定するモデルは現実と合わない
- **小サンプル種の数字振れ**: Pintail (n=28) +0.033、Shoveler (n=14) +0.031、Goldeneye (n=64) −0.021。test n が小さい種ほど run 間の F1 変動が大きい。これは「介入の真の効果」と「乱数の振れ」を切り分けにくくする。今後は **複数 seed で平均を取る** ことも検討すべきかも
- **6 run 撃って 0.85 を超えられないのは構造説**: ハイパラ4 + データ2 = 6 run、全部 run05 を超えられない。「カモ類は AST-base でこの辺りが上限」という構造仮説が説得力を持ってきた。次は「データやハイパラを変える」ではなく **「モデル容量」や「特徴表現」を変える** べき段階かもしれない

### 次にやる（方向性の見直し）

事前登録の分岐は「cap=50 か class-weighted loss」だったが、chunks 半減で precision が動かなかった以上、cap=50 の期待値は極めて低い。方針再考:

1. **混同パターンの音響的根拠を調べる**: Tufted と予測されて Teal だった13件の分光図を並べる。実際に音響的に紛らわしいなら chunks 介入では永遠に解決しない
2. **run09 候補: class-weighted loss** — chunks の数でなく損失の重みで介入。期待値は低めだがコストも低い
3. **run10 候補: AST-Large** — モデル容量を上げる。VRAM 8GB との戦い
4. **run11 候補: モデル選択方式の改善** — `load_best_model_at_end` の val 単点ピーク依存を見直す。これは独立軸の改善
5. preprocess.py の `_extract_xc_id` バグ修正は引き続き TODO

最初に 1 をやる方が筋。介入で動かない理由が「データの限界」か「モデルの限界」かで打ち手が全く変わる。

---

## 2026-05-24 分析タスク: 誤分類の真因は「Tufted の長尺2録音による決定境界の歪み」

### やったこと

run08 の predictions.csv に対して `src/bird_fine/analysis/confusion_audio.py` を新規実装。誤分類されたチャンクの メルスペクトログラム を正解チャンクと並べる + xc_id 別の集中度を集計 + 問題録音の Xeno-canto メタデータを精査、の3段階で原因を追跡。

### 発見の連鎖

**Step 1: 誤分類は録音単位で異常集中していた**
- Tufted と予測されて Teal だった 13件のうち **12件が同じ XC197026 から**
- Tufted と予測されて Wigeon だった 4件は **全て XC349677 から**
- 一方 Tufted の正解 7件は 3録音に分散、Teal の正解 67件は 8録音に分散

「種同士が音響的に似ている」なら誤分類は録音間で分散するはず。この集中度は「特定録音の問題」を強く示唆。

**Step 2: 問題録音は「学習データに無いタイプの音」だった**
- XC197026: メス + **6羽の幼鳥**。"The short whistling call is juvenile"
- XC349677: **オスとメスの掛け合い** ("interspecific calls between males and females")

**Step 3: train data の stage 分布に重大な偏り**
- Eurasian_Teal train 42録音のうち **stage=juvenile は 1件のみ**
- test には juvenile 録音が 1件入っている (XC197026 自身)
- AST は「カモの juvenile call」をほぼ学んでいない

**Step 4: 仮説の更新（chunks 不均衡説の完全否定）**
- run08 で「chunks 半減でも precision が動かなかった」事実と整合する仮説:
- **Tufted_Duck の長尺2録音 XC488112 (45分) / XC488113 (49分) が「カモ全般のデフォルト call」のような幅広い音響パターンを学習させている**
- 学習データに無い音（juvenile call、複数個体の鳴き合い）は、最も音響範囲の広い Tufted_Duck クラスに吸引される
- cap=100 ではランダムサブサンプリングなので XC488112/XC488113 の代表チャンクは残り、多様な音響パターンも保持される → precision が動かない理由

### なぜこの調査が機能したか

- run08 の **「chunks 半減で precision が完全同値」という強い反証** が出ていたから、別の説明を真剣に探した
- 分光図を「グループ別に並べる」だけでなく、**xc_id 別の集中度（録音単位の頻度）** を一緒に出した。これがなければ「特定録音への集中」は気づけなかった
- 集中していると分かった時点で、**メタデータの type / sex / stage / rmk** を確認 → 「学習データに無いタイプの音」が浮かび上がった

### 学び（重い方法論の知見）

- **「同じ種の誤分類」は均一に発生していると無意識に仮定していた**。だが実際は「特定録音」「特定 stage」「特定状況」に偏る。これは Xeno-canto のような **「ユーザー投稿型データセット」固有の偏り**。データ取得時に type/stage/sex のメタデータでバランスを取らないと再発する
- **メルスペクトログラムは「種」より「録音状況」を強く反映する**: 長尺フィールド録音は環境音や複数個体の鳴き合いを含むため、その「種の典型的な call」とは異なる音響パターンを大量に含む。チャンク分割すると、その「典型じゃない音」が学習データの中で大きな比率を占めることになる
- **chunks 数で均しても「音の多様性」は均せない**: cap=100 のランダムサブサンプリングは平均的に多様性を保つ。「Tufted の決定境界が広い」のは chunks の量ではなく「その種を代表する音響パターンの多様性」の問題
- **分析タスクを 1 run と数えるべきか**: 学習せず docs に分析結果だけ残す段階を「分析タスク」として位置付け、experiments.md に run 番号と並列して書いた。今後も「学習せず検証する」フェーズが必要な局面で再利用したい

### 次にやる: run09(pre) で XC488112 / XC488113 を train から完全削除

`src/bird_fine/data/exclude_train_recordings.py` を新規実装。`--xc-ids XC488112 XC488113` で削除。Tufted_Duck train chunks: 361 → 161（-200）。録音数 29→27。

仮説検証ポイント:
- H2 主: Tufted precision ≥0.55（run08 0.269 から大幅改善するか）
- H4: XC197026 の13チャンクが Tufted ではなく Teal と予測されるか（run08 では 12/13 が Tufted）
- H5: XC349677 の4チャンクが Wigeon と予測されるか（run08 では 4/4 が Tufted）

H2 + H4 + H5 が同時成立すれば、「長尺2録音が決定境界を歪めていた」仮説が確証する。

---
