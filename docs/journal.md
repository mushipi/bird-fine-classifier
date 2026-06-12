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

## 2026-05-24 run09: 仮説確証 + 介入が強すぎた — 真因は特定できたが「適正解」ではない

### やったこと

run09(pre) で立てた通り、XC488112 / XC488113 を train から完全削除（Tufted chunks 361→161）。学習・test 評価。

### 結果

- val f1 0.8436 (epoch 5) / **test f1 0.8098** / acc 0.870
- 全体 f1 は run08 から −0.010 / run05 から −0.028 でさらに悪化
- ただし **問題録音 XC197026 / XC349677 の Tufted 過剰予測は劇的に解消**

### 仮説検証のクライマックス: 問題録音の予測変化 (run05→09)

**XC197026 (Eurasian_Teal juvenile, n=13)**:
- run05: Tufted 13 → run07: 12 → run08: 12 → **run09: 4**（Pochard 6, Teal 3）

**XC349677 (Eurasian_Wigeon duet, n=6)**:
- run05: Tufted 5 → run07: 2 → run08: 4 → **run09: 0**（Mallard 4, Wigeon 2）

run08 で chunks 半減（cap=100）が効かなかったのに、**run09 の完全削除で劇的に変化**した。これは「chunks 不均衡」では解決しなかった現象を、「特定録音の完全削除」で動かせたことを意味する。**「Tufted の長尺2録音が決定境界を歪めていた」仮説の確証**。

### なぜそうなったか（仮説の核）

XC488112 (45分) と XC488113 (49分) という長尺フィールド録音は、Tufted_Duck だけでなく **複数の状況・複数個体・環境音** を含んでいた可能性が高い。チャンク分割（10秒）でその「Tufted 以外の音響パターン」が大量に Tufted_Duck ラベルとして学習され、AST が「Tufted_Duck = 多様なカモのデフォルト音」と覚えてしまった。

この仮説は cap=100 で動かなかった理由とも整合する: ランダムサブサンプリングでは「多様な音響パターン」が比率で残るため、決定境界の広さは変わらない。**完全削除して初めて学習対象から消える**。

### なぜ全体 f1 は悪化したか（仮説確証と副作用は独立）

主仮説は確証されたが test f1 は悪化（0.838 → 0.810）。理由は3つの独立した副作用:

1. **Tufted の学習データ不足**: chunks 161（他種 144〜390）まで減らしたら Tufted の **「本物の音響特徴」も学習しきれない**。Tufted の test 10件中 7件が他種に流出（recall 0.700→0.300）。test n=10 で正解 7→3 = F1 -0.073
2. **distribution shift は Tufted 削除では解決しない**: XC197026 の半数（6/13）が Common_Pochard に流れた。「train に juvenile タグ録音が 1件しかない」問題は、Tufted の決定境界を狭めただけでは「学習データに無い音」を正解にできない。受け皿が Tufted から Pochard に変わっただけ
3. **小サンプル種の振れ**: Shoveler n=14 で F1 +0.034、Pintail n=28 で F1 −0.016 など、test n が小さい種で run 間振れが大きい

### 学び（重い方法論の知見）

- **「数値仮説の不成立」と「仮説の意図の達成」を分けて評価する重要性**: H2「precision ≥0.55」は数値で不成立だったが、Tufted 予測総数 26→9 で過剰予測の意図は明確に達成。H4「XC197026 Teal 予測 ≥7」も数値不成立も、Tufted 予測 12→4 で「Tufted から離れる」意図は達成。事前登録した数値が外れても、**仮説の質的方向は検証できる**
- **介入の用量反応 (dose-response)**: cap=100 (Tufted chunks 728→361) → 効果ゼロ、完全削除 (chunks 0) → 効きすぎ。中間点（cap=30/50 で chunks 200程度）が適正解の可能性。「効くか/効かないか」の二値でなく **用量レンジで効き方を測る** べき
- **問題の独立性に気づく**: 「Tufted 過剰予測」と「distribution shift」は同じ症状（XC197026 が誤分類される）として現れていたが、**根本原因は独立**。Tufted を削っても juvenile 音は Pochard に流れる。**1 つの介入で複数の独立問題を同時解決はできない**。問題ごとに別の介入を順次重ねる必要
- **run08 の反証 → 分析 → run09 で確証 の流れが機能**: 「chunks 半減で動かない」という強い反証が出たから、別の説明（特定録音）を真剣に探した。**反証は仮説リファインの最良のシグナル**。事前登録した H2 が大外れだったことが、結果的に正しい次の仮説に導いた
- **「現行ベスト run05」の正体**: run05 は XC488112/XC488113 を train に含んでいた状態で test f1 0.838 を達成。これは「過剰予測を含む状態」の天井。 **過剰予測を削ると Tufted 学習も削れる** トレードオフがある以上、run05 を超えるには「両立する解」が必要

### 次にやる

- **run10(pre): 長尺2録音の中間 cap を探る** — 完全削除でなく cap=30 や cap=50 で部分的に残す。XC488112 と XC488113 だけに cap を適用する `cap_specific_recordings.py` あるいは `cap_train_chunks` の拡張で対応
  - cap=30: XC488112 270→30, XC488113 297→30 → Tufted total 161 + 30 + 30 = ~221 chunks
  - cap=50: Tufted total ~261 chunks
- **run11(pre) 候補: Teal juvenile 録音追加** — distribution shift の独立対処。Xeno-canto で `stage:juvenile` の Eurasian_Teal を検索して追加
- 順序: run10 で中間点を見つけてから run11、並行は変数増えすぎ

### 副次的な良い発見

Tufted の test 正解 3件（XC303149, XC476421, XC760407）はいずれも **短尺録音**から。長尺2録音を削除しても **「本来の Tufted_Duck の鳴き声」は他の27録音から学べている**。これは「Tufted 27 録音で十分に種特徴を捉えられる」可能性を示唆。あとは「過剰予測しないバランス」を見つけるだけ。

---

## 2026-05-31 Ubuntu移行 + 3sチャンクへのパラダイム変更

### やったこと

開発環境を Windows から Ubuntu（同一マシンのデュアルブート）に移行。あわせて BirdNet のソースコード（`kahst/BirdNET-Analyzer`）を確認し、チャンク長の設計上の問題を発見したため 10s→3s への変更を決定。run10 として事前登録した。

### 発見: BirdNet は 3秒窓で処理している

BirdNet-Analyzer の `birdnet_analyzer/model.py` に `keras.Input(shape=(144000,))` とあり、48kHz × 3秒 = 144000 サンプルが確認できた。`audio.py` の `split_signal()` もデフォルト `seconds=3.0`。

本プロジェクトは「BirdNet が カモ類 を検出した 3秒音声を受け取って8種に細分類する」設計のはずなのに、**学習時は 10秒チャンクを使っていた**。推論時には 7秒分のゼロパディングが発生する設計不整合だった。

### 旧 run10 計画（cap=30/50）との関係

run09 終了時点で「next: run10 は長尺2録音の中間 cap を探る」としていたが、それより上位の前処理設計の問題を先に直すべきと判断した。cap 実験は 3s チャンク体制が安定してから再検討する。

### Ubuntu 移行で対応した問題

Windows 側で生成された `data/splits/*.csv` の `file_path` 列がバックスラッシュ区切り（`data\processed\...`）だったため Linux で動かない。対応:
1. CSV を sed で一括修正（`\` → `/`）
2. `dataset.py` と `confusion_audio.py` の `Path / row["file_path"]` に `.replace("\\", "/")` を追加（防御的修正）

### コード変更内容

- `config.yaml`: `chunk_duration_sec` 10.0→3.0、`min_chunk_duration_sec` 3.0→1.0、`feature_extractor_max_length: 304` を追加
- `dataset.py`: `build_datasets()` に `max_length` 引数を追加、`ASTFeatureExtractor.from_pretrained()` に渡す
- `train.py`: `build_datasets()` 呼び出し時に `model_cfg` から `feature_extractor_max_length` を読んで渡す

### max_length=304 の根拠

ASTFeatureExtractor は 16kHz / 25ms窓 / 10ms hop でメルスペクトログラムを作る。3秒音声のフレーム数 ≈ (48000-400)/160+1 ≈ 298。AST の慣例（10s=1024, 1s=128 ≒ 102.4フレーム/秒）から 3s ≈ 307。実装上の標準値として 304 を採用。

### 次にやる

`uv run python -m bird_fine.data.preprocess` → `split` → `train --dry-run` → `train` の順で実行。

## 2026-05-31 run10 完了: 3sチャンク移行の結果と次のアクション

### やったこと

Ubuntu 移行後、3s チャーク移行（BirdNet pipeline alignment）を run10 として実行。XC488112/XC488113 を chunks_index.csv 段階で全splits から除外し re-split した上で学習。

### 結果

- val f1 0.806 (epoch 6) / **test f1 0.782**（run05 比 −0.056）
- **Tufted_Duck F1: 0.389 → 0.738（+0.349）** — 最大の成果
- Goldeneye（−0.201）・Pintail（−0.227）・Shoveler（−0.230）が大幅悪化

### なぜ Tufted が改善したか

XC488112/XC488113 の完全除外（run09 で仮説確証済み）に加え、3s チャンクで train 477 chunks（run09 の 161 より多い）を確保できた。「過剰予測の解消」と「適正な学習量」の両立が初めて達成された。

### なぜ全体 f1 は低下したか

1. **AST 位置埋め込みの適応コスト**: 10s→3s で位置埋め込みを 1214→350 に線形補間。事前学習の文脈（10s）から離れた分、汎化が落ちた
2. **train チャンク数が少ない種の悪化**: Pintail 305/Shoveler 243 は run05 と同水準だが、3s チャンクは 1 チャンクあたりの情報量が少ない（10s の 1/3）→ 実質的な学習情報量は減少
3. **test n の変化**: 3s 再分割で test n が変わり（例: Goldeneye 64→199）比較が難しい面もある

### 技術的な落とし穴と修正

位置埋め込みリサイズ後に `model.config.max_length` を更新していなかったため、保存 config.json（max_length=1024）と実際の重み（350次元）が不一致。evaluate.py でロード時にエラー。
→ `model.config.max_length = max_length` を train.py に追加し、既存チェックポイントの config.json も手動修正。

### 次にやる

run10 で Tufted F1 が改善した一方、全体は低下。2つの方向性が考えられる:

1. **run11: ハイパラ調整（lr, patience）で全体 f1 を底上げ** — 3s チャーク体制を維持したまま学習を安定させる
2. **run11: Pintail/Shoveler などデータ不足種に録音追加** — チャンク数の少ない種の弱点を補う

AST の 3s 適応には複数 epoch が必要な可能性があり、lr を下げて patience を増やす（より長く学習させる）のが有力。

### 追記: XC488112/XC488113 の全splits除外 と re-split (2026-05-31)

preprocess → split 後に Tufted_Duck val が 1032 chunks と異常に多いことを発見。XC488113（49分）が val に丸ごと入っていた。`load_best_model_at_end` がこれを基準にモデルを選ぶと run03 と同じ選択バイアスが再発する。

対処: chunks_index.csv から XC488112（899 chunks）・XC488113（989 chunks）を除外し re-split。Tufted_Duck val が 1032→76 に正常化。run09 の知見（「この2録音が問題」）と一貫した処置。

また dry-run で位置埋め込みの次元ミスマッチ (`tensor a=350 vs b=1214`) が判明。`resize_position_embeddings()` を train.py に実装し、事前学習済み1214次元の埋め込みを線形補間で350次元にリサイズしてから学習する方式とした。

---

## 2026-05-31 run11 設計：Double Dipping の罠と "other" クラス戦略

### やったこと

run10 の OOD 評価（AUROC softmax=0.838 / energy=0.894）を受けて、「"other" クラスを追加して再学習」する run11 の設計を行い実装した。

### 最重要の設計判断：Sampler と Loss の役割分担

当初「WeightedRandomSampler で均等化 + Loss に N_duck/N_other の重みを掛ける」と提案したが、これは **Double Dipping**（二重補正）だと指摘を受けた。Sampler で既にバッチ内比率が均等になっているのに、さらにデータ数ベースの Loss 重みを掛けると "other" の勾配が爆発して8種の決定境界が破壊される。

**採用した設計：**
- Sampler: WeightedRandomSampler（全9クラスを均等にサンプリング）
- Loss: CE weight = [1.0, ..., 1.0, 1.5]（固定値。データ数ベースは捨てる）

### SpecAugment の条件付き適用

run06 で「8種にSpecAugment → 境界が壊れて test f1 -0.028」を確認済み。一方で "other" クラスは過学習防止のために波形の多様性が必要。→ `if label == other_label_id` の条件分岐で **"other" のみ適用**。実装コストが低い割に理論と実利が完璧に一致する解。

### "other" チャンク数の設計（過学習防止）

per-species 50chunk 上限では 350chunk 程度 → Sampler での複製が多く丸暗記リスク。recording 単位の train/eval 分割 + per-recording cap=30 のみにして上限を外し、1220chunk を確保。複製回数は各チャンク ~0.5回/epoch と低水準に抑えた。

### metric_for_best_model の変更

9クラス全体の f1_macro をモデル選択基準にすると「other が高くて8種が壊れたモデル」が選ばれるリスク。→ `f1_macro_8class`（8種のみの F1 マクロ平均）を選択基準に変更。other_recall はログに出すが選択には使わない。

### 次にやる

run11 の学習・評価完了後、OOD eval で AUROC と energy threshold を再測定する。

## 2026-05-31 Outlier Exposure としての割り切り — 第4の選択肢への収束

### 経緯

run11 の結果（AUROC energy 0.932 / other_recall 0.09）を受けて、OOD 対策の最終方針を決定した。

当初の選択肢は「backbone 凍結して "other" 決定境界を強制する」か「energy パイプラインを構築する」の二択だったが、正しいフレーミングはどちらでもなく——

**第4の選択肢: Outlier Exposure としての割り切り**

run11 でやったことは Hendrycks et al. (2018) の Outlier Exposure そのもの。「OOD データを学習に晒して energy 空間を calibrate する」ことが目的であり、"other" を正確に分類することは目的ではなかった。other_recall=0.09 は失敗ではなく「エネルギー空間が分離された結果として softmax 空間では境界が引けない」という構造的帰結。AUROC 0.932 が成功の証拠。

### 決定したアーキテクチャ

```
推論時:
  logits → energy_score = T * logsumexp(logits / T)  (T=1.0)
  energy_score < energy_threshold → reject ("unknown duck species")
  energy_score >= energy_threshold → argmax(logits[:8]) → 種名
```

- 閾値は `species_taxonomy.yaml` の `energy_threshold: 10.35` で管理
- 温度は `energy_temperature: 1.0`（将来 Temperature Scaling を試す際は config だけ変える）
- `predict.py` に energy gate を追加することで完成

### なぜシステム側で解決するか

- AUROC 0.932 は実運用に耐えるレベル
- "other" をモデル側で分類しようとすると「閉世界仮定で開世界の問題を解く」矛盾を抱える
- energy ゲートはモデルと独立しているため、閾値だけ調整すれば FPR/TPR のトレードオフを変えられる
- run11 のモデル重みを変えずに運用設計を完成させられる

### energy スコアの符号規約（コード上の注意）

- ood_eval.py の convention: `score = logsumexp(logits)` → 高い = in-distribution
- 閾値 10.35 はこの convention で calibrate されている
- predict.py でも同じ convention を使う。論文の E(x) = -logsumexp(logits)（負値）とは符号が逆なので混同注意

### 次にやる

`predict.py` に energy gate を実装して動作確認し、コミットする。

---

## 2026-06-02 メタデータ駆動キュレーション基盤 — 耳で聞かずに長尺録音と shift を自動検出

### 経緯

AudioMAE/SSAM へのバックボーン差し替え検討（`docs/model_comparison_audiomae_ssam.md`）の中で「ボトルネックはアーキでなくデータキュレーション」と結論したのを受け、「データキュレーションは耳で聞く以外に手法があるか?」という問いを立てた。手法を棚卸し（メタデータ / 信号処理 / 埋め込み / モデルベース）した上で、最小コストの第一段＝**Xeno-canto メタデータの復元**に着手した。

### やったこと

`src/bird_fine/data/enrich_metadata.py` を実装。`data/raw/{Species}/metadata.csv`（DL時に保存済み・454録音）を結合し、split CSV に録音メタ（type/sex/stage/q/length_sec 等）を付与。`data/splits/*_enriched.csv` と `outputs/curation/recordings.csv` を出力する。

### 発見1: split がメタを捨てていた / file-name 照合で100%復元

現行の `data/splits/*.csv` は species/xc_id/chunk_index/file_path/duration_sec/source_file しか持たず、type/stage/length 等を保持していなかった。juvenile shift（run09）や長尺録音を「耳で」発見せざるを得なかった一因。

`xc_id` から XC番号を抽出して metadata と結合（大多数）＋ **XC番号が無い古い録音（33本）は source_file を metadata の `file-name` 列と正規化照合**することで、**8種は全 split で 100% カバー（未一致0本）を API を一切叩かずに達成**した。

### 発見2: API再取得は不要だった

当初の全体カバー率（train 75.9%）は低く見えたが、未一致の大半（XC番号あり・metadata未収録の103録音）は**全て "other" クラス**で、8種分析には無関係。8種本体は最初からほぼ完備していた。`xcapi_runs.json` は ID リストのみで生メタを含まず補完に使えないことも確認。"other" のメタが必要になった時だけ API 取得すれば足りる。

### 結果: 耳で聞かずに新しいキュレーション候補を自動検出

**(1) 長尺×冗長チャンクの偏り** — train の録音あたり chunks は中央値8に対し、`Mallard XC396538`=**214ch**(10:41) / `XC717935`=194ch(9:42) / `Eurasian_Wigeon XC780288`=161ch(8:03) が突出。run09 で問題化した「特定録音が学習を支配する」構造が Mallard/Wigeon にも存在。`length_sec>10分` のフラグだけで検出できた（除外済みの XC488112=44:55 / XC488113=49:25 もこの基準で即判別できることを後追い確認）。

**(2) distribution shift の定量化** — train vs test の stage 構成比クロス集計で:
- **Tufted_Duck: juvenile / "adult,juvenile" が test に各1本あるのに train は両方ゼロ**（train は adult のみ）。run10 で Tufted F1 が改善した後も残る新発見の shift。
- Eurasian_Teal: juvenile が train 1本のみ（run分析の「Teal juvenile 1件」をメタで裏付け）。test の uncertain 2本に対し train ゼロ。

### 学び

- 「データキュレーション」は主観的な傾聴に限らず、メタデータ照合で体系化・自動化できる。第一段（メタデータ復元）が最小工数で最大効果——run09 で数時間かけて耳で特定した juvenile shift / 長尺録音が、`stage` と `length_sec` のフラグで即座に出る。
- 見かけのカバー率（75.9%）に飛びつかず「未一致の中身」を見たことで、無駄な API 再取得（全部 other）を回避できた。費用対効果は対象を分解して初めて見える。

### 次にやる

run12 候補（メタデータ分析が示すレバー、いずれも「特定録音のキュレーション」軸）:
1. **Mallard/Wigeon の長尺録音に中間 cap** — run09 終了時に積み残した「長尺の中間 cap」を 3s 体制で実施。チャンク不均衡の是正。
2. **Tufted / Teal の juvenile shift 対処** — train に juvenile 録音を追加し test との分布ギャップを埋める（独立軸）。

「1 run 1 変更」の原則に従い、まず (1) か (2) のどちらかを単独で。

### 追記: run12 検証分析 — Tufted juvenile shift を確証（同日）

run を回す前に、run09 で機能した「分析→run」の流れに倣い、run11 の test 予測（`outputs/eval_20260531_231427/predictions.csv`）を `test_enriched.csv` と順序結合して Tufted_Duck の誤分類を stage 別に分析した。

**結果（仮説の完全確証）**:

| stage | chunks | recall |
|---|---|---|
| adult | 10 | **1.000** |
| adult, juvenile | 21 | 0.810 |
| (未記載) | 22 | 0.818 |
| **juvenile** | 17 | **0.000** |

- Tufted F1 低下（0.703）の主因は **juvenile distribution shift** で確定。juvenile 録音 **XC667403(17ch) が全て Eurasian_Wigeon に誤分類**（recall 0.000）。
- adult は recall 1.000。モデルは Tufted の adult を完璧に識別できており、**juvenile の音響パターンを学習していないだけ**（train に Tufted juvenile が0本）。
- precision 誤り（過剰予測）は13件に減少し、もはや主問題ではない。run05/08 時代の「Tufted 過剰予測」から、ボトルネックが recall 側（juvenile 未学習）へ移行した。
- run09 の XC197026（Teal juvenile）と同じ「特定録音 × distribution shift」構造が Tufted で再現。

**なぜメタデータ基盤が効いたか**: stage 列が無ければ「Tufted の誤りが juvenile に集中」とは切り分けられなかった。本日構築した enrich_metadata.py の stage 付与があって初めて、誤分類 24 件のうち 17 件が単一 juvenile 録音に由来すると即座に分かった。

### 次にやる（更新）

run12 = **train に Tufted juvenile 録音を追加**（test の XC667403/XC667392 を除外し汚染防止、run07 の教訓で cap 併用）。
事前登録の予測値を立てるには Tufted juvenile の入手可能性確認が先決——`download.py --metadata-only` で Xeno-canto の在庫を調べる（stage=juvenile は希少。メタ全体で11本のみ）。在庫次第で run12 が成立するか決まる。

### 追記: Tufted juvenile は Xeno-canto では枯渇 — 他公開ソースへ展開（同日）

run12 の前提（juvenile 録音の入手可能性）を Xeno-canto API で確認した結果、**データ駆動（XC追加）では解決不能**と確定した。

**在庫調査（read-only・保存なし）**:
- `en:"Tufted Duck" stage:juvenile` / `gen:Aythya sp:fuligula stage:juvenile` のいずれでも**世界で2件のみ**、しかも両方 test 既存（XC667403 / XC667392）。英名・学名で結果は完全同一でクエリの取りこぼしではない。
- type/remarks まで広げても追加候補は2件（XC668878 / XC577931）だが、いずれも q=C かつ nestling(雛)で、test の juvenile call とは発達段階が異なり代替にならない。
- Tufted Duck 全300件の stage 分布: adult 62 / 未記載 207 / juvenile 1 / adult,juvenile 1 / uncertain 28 / adult,nestling 1。

**YouTube 案の検討と保留**: 候補に挙がったが、(1) データ源異質性が run 比較を壊す、(2) 録音単位 shift の本質（test の XC667403 への汎化保証が弱い）、(3) 種同定の確度、(4) ライセンス、の4点で本流の run には不適と判断。PoC 扱いにする。

**方針: 他の公開鳥類音声ソースを検討**。候補は Macaulay Library（age 記入率が高く有望／再配布制限）、iNaturalist（CC選択可）、GBIF（横断・ただし XC を集約するため重複排除必須）、Animal Sound Archive Berlin、British Library Wildlife。条件は **CC0/CC-BY 優先**と **XC-ID 照合による重複排除**、学術フィールド録音同士で異質性を抑えること。まず**無認証で叩ける GBIF + iNaturalist** から Tufted juvenile 音源の在庫を確認する。

**学び**: メタデータ基盤（stage 列）があったからこそ「juvenile が枯渇している」を定量的に確定でき、効果の出ない録音追加 run（run07 の二の舞）を回避できた。データキュレーションの検証は「追加すべきデータが存在するか」の在庫確認まで含めて初めて完結する。

### 次にやる（更新2）

GBIF + iNaturalist の公開 API で「Aythya fuligula × 音声付き × juvenile」の在庫・ライセンス・XC重複を確認。実在すれば run12（他ソース juvenile 追加）が成立、無ければ augmentation か他弱点種（Shoveler/Goldeneye）へ軸を移す。

### 追記: 無認証3ソースで Tufted juvenile 枯渇を確定（同日）

GBIF / iNaturalist の公開 API（無認証・read-only）で在庫確認した:
- **GBIF**: 音声付き Tufted で `lifeStage=Juvenile` は **1件のみ**。occurrenceID が `data.biodiversitydata.nl/xeno-canto/observation/XC667403`、media が xeno-canto.org、creator が Stichting Xeno-canto → **XC667403 の重複で test 既存**。GBIF は Xeno-canto を集約しているため新規性ゼロ。
- **iNaturalist**: Life Stage 注釈=Juvenile かつ音声ありは **0件**（音声付き観察は44件あるが juvenile タグは皆無）。

**結論**: 無認証で叩ける主要3ソース（Xeno-canto / GBIF / iNaturalist）すべてで、train に追加できる新規 Tufted juvenile 音源は **0本**。juvenile call の録音が世界的に極めて希少という分野共通の事実が3ソースで裏付けられた。**データ追加で juvenile shift を埋める路線は行き止まり**。

**決定**: Macaulay Library（認証要・再配布制限）と augmentation（疑似 juvenile 合成）は将来の別 PoC に回す。run12 は **他弱点種（Shoveler 0.738 / Goldeneye 0.743）を同じ stage 別分析で切り分け、データで埋められる確実なレバーを探す**方向に確定。juvenile shift は「データ不在による解決不能問題」として確定記録した。

### 次にやる（更新3）

Shoveler / Goldeneye の run11 test 誤分類を enriched メタで stage 別に分析し、Tufted と同じ juvenile/特定録音 shift か、別要因（録音多様性不足・音響類似）かを切り分ける。確実に埋められるレバーがあれば run12 として事前登録。

### 追記: 他弱点種の切り分け — Goldeneye は「表現力ボトルネック」と判明（同日）

**Shoveler/Goldeneye は juvenile shift ではない**（stage はほぼ未記載/uncertain）。Tufted の juvenile 枯渇問題とは別物。

- **Northern_Shoveler (recall 0.686)**: 誤分類が複数録音に薄く分散、誤予測先も Pochard/Teal/Mallard とバラバラ。**分散型で単一介入が効きにくい**。
- **Common_Goldeneye (recall 0.623)**: **`song`(求愛ディスプレイ音)が Eurasian_Teal に一方向流出**。song の正解59/誤54、誤りは主に Teal へ。Teal precision=0.749 の誤吸込65件中 Goldeneye が42件で突出。`display call`(正32/誤2)や `call` は当たるのに `song` だけ詰む。

**埋め込み距離による切り分け（run11=v11 の AST 最終層 mean pool, cos類似）**:

| 群 | n | →GE重心 | →Teal重心 | Teal寄り割合 |
|---|---|---|---|---|
| 誤分類 GE song(→Teal) | 41 | 0.900 | **0.912** | **68%** |
| 正解 GE song | 89 | **0.940** | 0.853 | 0% |

重心間 cos(train GE song, train Teal song)=**0.880**（両者の song が埋め込み上ほぼ重なる）。

**結論: (a) 本質的な音響類似で確定**。正解 song は train GE song に近い（Teal寄り0%）が、**誤分類 song は68%が train Teal song の方に近く、埋め込み空間で実際に Teal 領域に入り込んでいる**。train Goldeneye song は275chunk と十分あるのに分離できない → **データ追加では解決せず、現行 AST の表現力がボトルネック**。（0.900 vs 0.912 は僅差で「GE song と Teal song がほぼ重なる難境界」だが、データで埋まらない点は明確。）

**メタ含意（重要）**: 一連の分析で弱点が2種類に分離した。
- **データ不在型**（Tufted juvenile）: 世界に2録音で枯渇 → data/aug/別ソース全滅、解決不能
- **表現力不足型**（Goldeneye song）: Teal song と音響類似でデータ十分 → **バックボーン強化が本命の解**

`model_comparison_audiomae_ssam.md` の「ボトルネックはアーキでなくデータキュレーション」という結論に対し、**Goldeneye は明確な反例**（アーキ強化が本命の種が実在）。最初のドキュメントで第一候補に挙げた AudioMAE 評価が、今度は定量的根拠付きで再浮上した。

### 次にやる（更新4）

run12 = **AudioMAE 全体fine-tune による表現力評価**を事前登録する（model_comparison ドキュメントの Go/No-Go チェックリストに沿う）。期待: Goldeneye song ↔ Teal song の分離向上。データ系レバー（Tufted juvenile / 録音追加）は尽きたため、アーキ軸へ移行。

## 2026-06-03 ドメイン整合 — 「幼鳥は冬の日本に不要」を検証し評価セットを修正

### 経緯

juvenile データ枯渇を延々追っていたが、ユーザーの「**そもそも幼鳥は冬の日本のモニタリングに不要では?**」という上流の問いを検証。これが一連の juvenile 議論を根本から正した。

### 検証結果（決定的）

- test の Tufted juvenile 録音 XC667403/667392 は **2021-08-05・フランス**＝繁殖期・繁殖地の雛(begging call)。冬の日本では遭遇しない音響パターン。
- ドメインフィルタ別に run11 を再評価（既存 predictions）:
  - stage=juvenile/nestling 除外 → f1_macro_8 0.798→**0.808**、Tufted **0.703→0.767(+0.064)**
  - 月で繁殖期(5-8月)除外 → 0.764 / Tufted 0.480（adult call を巻き込み悪化）→ **stage ベースが正しい**
- **さらに大きな発見**: train/test とも **日本録音0**（test 0/73・train 0/313）、繁殖期録音が test 18/train 92。「Japan→worldwide フォールバック」の帰結で、run01〜11 の test f1 は厳密には「冬の日本での性能」を測れていなかった。

### 対処

`filter_domain.py` を実装（`enrich_metadata` の照合ロジックを再利用）。**val/test から juvenile/nestling を除外**（val 1358→1285・test 1154→1038）。train は学習の音響多様性のため不変（ユーザー確定方針）。旧 split は `*.bak` 退避。run12 以降の baseline は **test v2（f1_macro_8class 0.808）**。

### 学び

- 「データをどう埋めるか」の前に「**その評価対象はドメインとして正しいか**」を問うべきだった。juvenile のデータ枯渇（XC/GBIF/iNat 全滅）を数時間追ったが、真の答えは「そもそも評価に入れるべきでない」だった。
- メタデータ基盤（stage/date/cnt）があったから、ドメイン外を録音単位で即特定でき、評価を定量的に正せた。

### 次にやる

対象種拡張（軸2）。ood_tier1 のカモ科のうち**データ十分な4種**（Gadwall/ウミアイサ/カルガモ/カワアイサ）を target 昇格し run13 として事前登録（8→12種）。下位3種(トモエ/ヨシ/スズ)は録音不足で見送り。Goldeneye の表現力ボトルネック(run12 AudioMAE)は対象種拡張後に再評価。

## 2026-06-04 run13 準備: 対象種拡張 8→12種 を実装（split まで）

### やったこと

4種を target 化し、split・label_map・config を 12種体制に再構築。run13 として事前登録（学習は未着手）。

### データ収集と2つの落とし穴

1. **カルガモが2本しか取れない** — `download.py` の `fallback_worldwide` は「指定国で**0件**のとき」のみ worldwide 検索する設計。カルガモは Japan に2件ヒットしたため worldwide(16本)に行かなかった。`_download(country=None)` を直接呼んで worldwide 追加収集し16本確保。**学び: fallback は0件時のみ。少数ヒット国があると worldwide を取り逃す。**
2. **label_map から other が欠落** — split.py / ad-hoc 再生成は `chunks_index.csv`（other を含まない）から label_map を作るため、other が抜けた。`build_datasets` は `label_map.csv` を直接読み、`other_label_id = label_map.get("other")` で OOD 重み付けを決めるため、欠落すると other 処理が無効化される。**other=12 を手動追加して解消。**

### split 戦略（既存8種を公平比較するため）

- split.py の通常モードは全種再分割で既存8種の割り当てが変わり、`--preserve-existing` は新種を全て train に入れてしまう。どちらも不適。
- → **既存8種 split（test v2）を完全保持し、新4種だけ `split_by_recording`(seed=42, 70/15/15) で分割して追記**。filter_domain を再適用し新4種の juvenile/nestling も除外（test 1299→1261）。これで run11↔run13 の既存8種比較が公平。

### init は pretrained

12種化で label_id のソート順が変わる（Common_Merganser 挿入等）ため、run11(v11) の9クラスヘッド重みを正しくマップできない。`init_from=""` で pretrained から初期化する。

### 次にやる

run13 学習 → 13クラス eval。H1（既存8種 f1_macro_8 ≥ 0.77）/ H2（新4種 macro 0.40〜0.60）/ H3（OOD AUROC ≥ 0.88）を検証。学習前に `train.py --dry-run` で VRAM・テンソル形状（13クラス・pretrained 初期化・位置埋め込み 304）を確認する。

### 結果: 対象種拡張は「半分成功」（同日）

- **H1 ほぼ不成立**: 既存8種 f1_macro_8 = 0.769（baseline run11 test v2 0.808 から −0.039、予測下限 0.77 を僅か割れ）。**H2 域内だが二極化**: 新4種 0.449（Gadwall 0.823 / ウミアイサ 0.746 ＝成功、カワアイサ 0.227 / **カルガモ 0.000 ＝失敗**）。**H3 成立**: OOD AUROC energy 0.902。
- **真因は同属相互混同ではなく Mallard の「吸引ハブ」化**。カルガモ27chは18→Mallard で自種0、カワアイサも Mallard/Goldeneye へ。pred=Mallard に Teal/Gadwall/カルガモ/カワアイサが流入し Mallard precision 低下 → 既存8種を −0.039 押し下げた。
- **学び**: (1) 「録音数が効く」を再確認（成功2種=録音30/17、失敗2種=録音11/8）。(2) カルガモは Anas 属で Mallard に酷似し、録音11でも多数派バイアスに負け全滅 → **録音追加(worldwide 16本上限)では限界。Goldeneye song と同じ「表現力/多数派バイアス」の壁**。(3) 種拡張は「データが揃う種だけ」が鉄則。
- **次にやる**: 12種そのまま採用は見送り。候補 ①Gadwall/ウミアイサのみ昇格(10種)・カルガモ/カワアイサは other へ戻す ②Mallard 過剰予測対策(CE重み/sampler) で12種再学習 ③run14 AudioMAE で表現力を上げ音響類似を分離。`species_taxonomy.yaml` の v13 切替は採用判断後。

## 2026-06-04 run14: focal loss 失敗 — カルガモの壁は分類層でなく表現力

### やったこと

run13 でカルガモ 0.000(Mallard吸引)。素 AST で test カルガモ 67% がカルガモ寄りだったため
「分類層の多数派バイアス」と見立て、focal loss(gamma=2.0)で難サンプル強調を試した（loss のみ1変更）。

### 結果（仮説の反証）

- **カルガモ F1=0.000 のまま（H1 不成立）**。既存8種 0.761(run13 0.769)・12種 0.646(run13 0.662) と微減。
- focal は Mallard precision を改善(F1 0.627→0.674)したが、その分 ウミアイサ(−0.103)/カワアイサ(−0.057)が犠牲。新4種全体は悪化。

### なぜ反証されたか

素 pretrained の「67% カルガモ寄り」は **pretrained 限定**。fine-tune すると CE でも focal でも、極小マージン（train カルガモ↔Mallard 重心 cos 0.982、test 0.917 vs 0.915）が Mallard 多数派に押し潰される。**focal=分類層を直接是正する最強手でも 0% → 分類層では救えない＝表現力の壁が本質**。

### 学び

- **「データ追加（run07/13）」「分類層調整（run14 focal）」のどちらでも近縁カモの壁を破れなかった**。残る軸は表現力（AudioMAE）かデータ品質（中国語斑嘴鸭録音の質）。
- Goldeneye song（run11 分析）と カルガモ（run13/14）が同じ「表現力の壁」に収束。AST の汎用表現は fine-grained 近縁差を分離しきれない、という一貫した診断。
- 切り分けの順序（データ→分類層→表現力）が機能した。focal の失敗は「表現力が要る」ことの確証であり、無駄ではない。

### 次にやる

①run15 = AudioMAE（表現力軸）で Goldeneye song + カルガモ の分離を検証、または ②カルガモ/カワアイサを other に戻し成功2種(Gadwall/ウミアイサ)のみ昇格＝10種運用で確実に前進。run14 は不採用、12種ベストは run13(v13)。

## 2026-06-04 表現軸 probe 検証: AudioMAE は総合AST以下・SSAMは断念

### やったこと

カルガモ/Goldeneye song の「表現力の壁」に対し、自己教師あり表現(AudioMAE/SSAM)を linear probe で検証。

### AudioMAE 実装の落とし穴（再利用知見）

- timm で `hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m` をロード。
- **前処理が AST と全く別**: kaldi.fbank(htk_compat, hanning, 128mel) + 正規化 (x-(-4.268))/(4.569*2)、1024frame固定。ASTFeatureExtractor 流用は不可（埋め込み崩壊 std~0.01）。
- **3s音声はゼロpadでなく10sタイル**（pad 71%が定数支配で崩壊）。
- **特徴は CLS でなく patch mean pool**。global_pool=token の CLS は MAE で定数的＝崩壊。
- 生特徴の重心比較は MAE型に不利（生encoderは分類向けでない）→ linear probe に切替。

### 結果（probe vs probe で公平比較）

| | カルガモ F1 | 既存8種 | 12種 |
|---|---|---|---|
| AudioMAE probe | **0.140** | 0.528 | 0.422 |
| AST probe | 0.000 | 0.631 | 0.472 |
| run13 AST FT | 0.000 | 0.769 | 0.662 |

- **総合は AST が上**（教師あり AudioSet が分類向け）。AudioMAE 全面採用の根拠は弱い。
- **カルガモだけ AudioMAE のみ非ゼロ(0.140)**。AST が原理的に潰す種を自己教師あり表現は微細に保持＝芽はあるが小さい。
- **SSAM は使える port が無く断念**（mamba-ssm ビルド+公式コード移植で AudioMAE の10倍超コスト）。

### 学び

- **表現軸に銀の弾丸なし**。データ追加(run07/13)・分類層(run14 focal)・表現力(AudioMAE/SSAM)を一通り試し、近縁カモ(カルガモ↔Mallard, Goldeneye↔Teal song)は現行リソースで分離困難と確定。
- linear probe は「fine-tune コスト前の表現力判定」に有効。生特徴比較が MAE型に不適だった失敗から probe へ切替えたのが効いた。
- 検証の網羅（データ→分類層→表現力）が完了。これ以上は「近縁ペアの受容」が現実解。

### 次にやる

実用化: カルガモ/カワアイサを other に戻し、成功2種(Gadwall/ウミアイサ)昇格の **10種運用** で確定（run15）。カルガモ/Goldeneye song は分離困難な近縁ペアとして受容。

## 2026-06-05 run15: 10種運用で確定 — 失敗種除去で既存種が回復

### 結果

カルガモ/カワアイサを other へ差戻し、Gadwall/ウミアイサ昇格の10種で確定:
- 既存8種 f1_macro_8 **0.786**(run13 0.769 から +0.017 回復, H1成立)
- 10種 f1_macro_10 **0.783**(run13 12種 0.662 を大きく上回る, H3成立)
- OOD AUROC energy **0.899**(維持, 閾値 6.73)

### なぜ既存種が回復したか

run13 で カルガモ/カワアイサが Mallard に吸われ Mallard precision が落ち、それが既存8種全体を −0.039 押し下げていた。2種を除くと吸引ハブが消え、**Mallard 0.627→0.664 / Goldeneye 0.679→0.765 / Shoveler +0.054** と回復。「失敗種を抱えるコスト」が既存種に波及していたことの裏返し。

### 学び（一連の探索の総括）

ユーザーの「幼鳥は冬の日本に不要では?」から始まった探索の結論:
- **ドメイン整合(juvenile除外)** = 効いた(Tufted +0.064)
- **対象種拡張** = データが揃う種だけ成功(Gadwall/ウミアイサ)。カルガモ/カワアイサは Mallard と分離困難
- **分類層(focal)・表現力(AudioMAE/SSAM)** = 近縁ペアの壁を破れず
- → **実用解は「データが揃う種のみ昇格＋失敗種は other で受容」**。10種(8→10)が堅実な成果
- 重要な一般原則: **失敗種を target に残すと既存種まで巻き添えにする**。撤退の判断が性能を上げることもある

### 確定した運用構成

カモ10種(マガモ/コガモ/オナガガモ/ハシビロガモ/ヒドリガモ/キンクロハジロ/ホオジロガモ/ホシハジロ/オカヨシガモ/ウミアイサ) + OOD energy gate(AUROC 0.899, threshold 6.73)。species_taxonomy.yaml = v15。

## 2026-06-05 「広くカバー」の査定: 日本の観察頻度で target 候補を仕分け

### やったこと

run15(10種確定)後、ユーザーの「運用上カモ類を広くカバーしたい」を受け、両輪(target拡張+OOD強化)の方向を確認。worldwide 在庫だけで候補を挙げたが、ユーザーの「どれも日本じゃ少ないかも」「ツクシは必要だがアカツクシは引っ張られる」「csv の記述も参考に」の指摘で、**北部九州 iNat 観察頻度(obs_count)で査定**し直した(`sync_species_master.py` 実行、bbox 32-34N/128.5-132E)。

### 査定結果（北部九州で観察されるカモ系、obs_count順）

- 既存 target: マガモ195/ヒドリガモ178/ホシハジロ88/コガモ86/オナガガモ60/キンクロ36/ハシビロ28/Gadwall20/ウミアイサ17/ホオジロガモ6
- **未登録だが観察される**: **ツクシガモ Common Shelduck 42(candidate)** / オシドリ16 / ミコアイサ9 / シマアジ5
- ood: カルガモ131 / ヨシガモ23 / トモエ18 / スズガモ11 / カワアイサ2

### 重要な気づき

- **worldwide 在庫 ≠ 日本の観察実態**。worldwide で大量のアカツクシガモ(124本)・シマアジ(81)・オシドリ(64)も、北部九州 obs では圏外〜低。ユーザーの直感が正しい。**査定は obs_count を主軸にすべき**。
- **ツクシガモ(obs 42)が target 最優先候補**。ハシビロ(28)・キンクロ(36)より観察され、形態・鳴き声が独特。しかも**近縁のアカツクシガモが日本にほぼいない＝混同相手不在で分離しやすい**(カルガモ↔Mallardの逆の好条件)。
- **最大の運用課題: カルガモ(obs 131・観察3位)を識別できない**。「マガモ or カルガモ」は日本のカモ観察の最頻ペアなのに run15 で分離困難と確定し OOD 止まり。ここが弱点。

### 次にやる（次セッション）

両輪の具体化:
1. **target 追加: ツクシガモ Common Shelduck**(Xeno-canto A品質31本)を収集→11種化。独特種なので run15 的な既存種巻き添えは起きにくい見込み。オシドリ(16)も次点候補。
2. **OOD 強化**: カルガモ/ミコアイサ/シマアジ等を OOD tier1 に整理し「カモ類」検知を広げる。
3. カルガモ識別は別軸の難問として継続検討(日本録音入手 or マガモ/カルガモ2値分類器 等)。

## 2026-06-05 総点検: run間の精度比較はほぼ全て統計的に無意味だった

### きっかけ

「セッションを総点検、根本的な間違いがないか」の依頼で評価を多角的に再検証した。

### 発見1: 指標の計算法で結論が逆転する

既存8種 f1_macro を3通りで計算したら値も順位も変わった:
- chunk単位・全データ(FP含む): run13 0.769 / run15 0.786（私が報告した「+0.017回復」）
- chunk単位・8種抽出: run13 0.798 / run15 0.792（−0.006、逆）
- 録音単位: run13 0.909 / run15 0.827（−0.082、大きく逆）

chunk単位は同一録音内の一部誤分類をペナルティし**性能を過小評価**（run15 10種 0.783→録音単位 0.842）。
クラス数の違う(9/11/13)モデルで「既存8種f1」を比べるのも誤予測先が変わり交絡していた。

### 発見2: 録音単位 bootstrap で run間差は全て有意でない

`compare_runs_ci.py` を作り、run11/13/15 を同一の既存8種 test(69録音)で録音単位クラスタ
bootstrap 比較した:
- run11 0.878[0.762,0.947] / run13 0.909[0.808,0.965] / run15 0.827[0.710,0.908]
- **全ペア差の95%CIが0を含む**（run15-run13 −0.082[−0.183,+0.018] 等）。CI幅±0.1。

→ **test 69録音では run間の微差を測る解像度が無い**。点推定の順位が指標で逆転するのは
「全部ノイズ」の証拠。

### 根本的な反省

- セッション中の run比較（「種拡張で既存低下」「10種で回復」「focal失敗−0.008」等）は
  **ことごとく統計的に無意味な差を効果と誤認**していた。run03 で「val f1スパイクの選択バイアス」
  を学んだのに同じ罠（小サンプルのノイズを効果と誤認）に再びはまった。
- 皮肉にも `docs/bootstrap_ci.py`（既存）の冒頭が「chunk単位の再標本化はCIを過小評価。正しくは
  録音を塊ごと再標本化」と明記していた。それを見落として chunk単位の点推定で全 run を比較した。

### 何が生き残るか

- 揺るがない: カルガモ F1=0.000（明確）、失敗種除去で f1_macro が桁で動く、カルガモ↔Mallall 混同
  （定性的）、juvenile はドメイン外（生態的妥当）、obs_count ベースの種査定。
- 崩れた: run11/13/14/15 の精度ランキング、±0.02〜0.08 の全比較、test v2 の +0.064 も要再検証。

### 確立したルール（CLAUDE.md / experiments.md に記録）

1. run比較は **録音単位 + bootstrap CI**（`compare_runs_ci.py`）。chunk単位点推定で優劣を断定しない。
2. CI が重なる差（≈±0.05未満）は「差なし」。大きな構造的差と定性的所見のみで意思決定。
3. **test の録音数拡大が最優先課題**（現状69録音では解像度不足）。

### 次にやる

新機能・追加 run より先に **評価系の是正**: (a) test 録音数を増やす（種ごと最低十数録音→数十）
(b) 今後の全 run を compare_runs_ci で CI 付き評価。これ無しの精度主張は信頼しない。

## 2026-06-06 再評価2: focal「失敗」は撤回、モデル選定は運任せと判明

### やったこと

compare_runs_ci に run10/run14 を追加し、run10〜15 を同一の既存8種 test(69録音)で録音単位 bootstrap 再評価。

### 結果（録音単位 既存8種 macro-F1）

run10 0.884 / run11 0.878 / run13 0.909 / **run14 0.934(最高)** / run15 0.827(最低)。
ペア差は10通り中**有意は run15-run14(-0.107)の1つだけ**(多重比較の偽陽性が濃厚)。他は全て CI が0を含む。

### 覆った判断

- **「focal は失敗(-0.016)」は誤り**。録音単位で run14 が5run中最高、run13と差なし。chunk単位の誤判定だった。ただし**カルガモ救済失敗(F1=0.000)は構造的に変わらず**＝focal の主目的は未達。
- **「other追加で+0.011改善」も無効**(run11-run10 差なし)。
- run10〜15 の既存8種は**統計的に区別できない**ことが確定。

### モデル選定への重大な含意

- **v15 は単一シードで録音単位最低(0.827, CI[0.71,0.91])**。同設定でも early stop/シードで大きく振れる。**v15 は「たまたま悪い引き」の可能性**があり、run14(0.934)が「良い引き」だったのかもしれない。
- → **「確定版」は撤回**。v15 は暫定運用モデル。experiments.md の run14「失敗」/run15「確定」記述と species_taxonomy.yaml に注記済み。
- **単発でモデルを作り直しても、また別の運任せの1個ができるだけ**。複数シード学習＋CI選定が要るが、それも test 69録音では選定不能。

### 結論と次にやる（A: test拡大から）

このプロジェクトの本質的な行き詰まりは **test の録音数不足(各種4〜14録音)で評価解像度が無いこと**。新機能・追加runより先に **(A) test 録音数の拡大** に着手する。順序: test拡大 → 複数シード学習 → compare_runs_ci で CI 付き選定。

## 2026-06-11〜12 Perch蒸留→運用昇格→OOD監査→複合クラス（系統転換の総括）

AST頭打ち打開の別系統(Perch凍結埋め込み+プローブ)を起点に運用モデルを刷新。全詳細→ docs/perch_kd_report.md。

- **正則化スイープ→P1 foundation比較**: Perchプローブのwd/Cは無反応(正則化軸枯れ)。Perch>>BirdAVES、線形系天井≈0.83。
- **Perch→AST 知識蒸留**: 教師=Perch複数arm平均(train OOF)。素CNNで「KDが効く」を統計確認(base0.45→KD0.60,+0.148有意)。AST単発は+0.025非有意も、多seed(3/3一貫)+soupで base-soup0.813→**KD-soup0.871,+0.058有意**→運用stage2昇格。
- **regime訂正**: 現splitsはv15学習時と別データ(commit 4cea661で全面改訂,test81→231録音)。実体10種(other/カルガモ/カワアイサはOOD)。
- **OOD/FP監査+閾値再キャリブレ**: 旧2.948はchunk導出でpredict(録音平均)と不整合→真カモ32%誤棄却。録音単位再導出で**2.717**(真カモ保持0.90)。AST(0.79)>Perch(0.76)=ゲートにPerch不要。非カモは解決、対象外カモが難所。
- **カルガモ壁=本質的**: Perch2.0本体でもカルガモ/マガモ分離不能(0/30)→精度は死に筋。**複合クラス(マガモ/カルガモ,slash)出力**で受容(再学習ゼロ)。混同行列で10種内に他の混同ペア無し確認。
- **git一本化**: 分岐main統合・GitHub同期。push はGT105リレー(mainPC token無効)。
- **残**: data/ood_processed再生成(陳腐化), test拡大(弱種の振れ=評価解像度の本丸)。

## 2026-06-12 test拡大: download マルチグレードバグ発見 → 弱種を実拡大

### きっかけ
弱種(ヒドリ n=7 / ウミアイサ n=11)の評価解像度を上げる test拡大に着手。品質A→B緩和で増えるか確認。

### 重大バグ: quality "A B" が常に0件
`download.py: _build_query` が quality="A B" を `q:"A B"`(不正な結合文字列)として投げ、Xeno-canto は常時0件を返す。
→ 一旦「弱種はデータ枯渇」と**誤結論**。実は `q:A` / `q:B` を別々に投げれば在庫あり(ウミアイサ42/キンクロ80件等)。
修正: `_download` をグレード毎クエリ→XCID統合(dedup)に。単一gradeは後方互換。

### 収集結果(worldwide A B, exclude-existing): 部分的だが本命に当たり
- ヒドリガモ raw 44→220 (+176, 最弱種が5倍), マガモ 100→290 (+190)
- ウミアイサ/キンクロ/ホオジロ/ホシハジロ: 0増 (worldwide A B も既収集 = 真の天井)
→ 2/6で大当たり。残4種は世界的にデータ天井で、これ以上は増やせない。

### 再split: test 231→289録音 (ヒドリ 7→35, マガモ 16→46)
split.py が OOD種(カルガモ/カワアイサ)も拾うので filter で除去し10種regime維持。

### 学び
- 「データ枯渇」結論はクエリバグ由来。**外部API引数の仕様(grade単一 vs 結合)を疑うべき**だった。
- worldwide緩和は弱種を救うがドメイン混入 = 最終CIで良し悪し判定(無条件昇格しない)。
- データ天井は種ごとに違う(ヒドリ豊富/ウミアイサ枯渇)。

### 次
Stage4 再学習(base+KD×3seed→soup, ~6h)中。Stage5でKD再現・弱種CI縮小・昇格判定。
