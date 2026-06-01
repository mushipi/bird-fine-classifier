# Troubleshooting

よくあるトラブルと対処法。実行環境は Ubuntu / bash。

---

## セットアップ系

### Q. `uv sync` で torch が CPU 版になる

`pyproject.toml` の `[tool.uv.sources]` で `pytorch-cu124` インデックス指定済み。
確認:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# True 12.4 が出れば OK
```

### Q. `XENO_CANTO_API_KEY が未設定` エラー

`.env` がプロジェクトルートにあるか確認。なければ作成:

```
XENO_CANTO_API_KEY=your_key_here
```

APIキーは https://xeno-canto.org/account/api で取得。

### Q. `ModuleNotFoundError: bird_fine`

必ず `uv run` 経由で実行する:

```bash
uv run python -m bird_fine.data.download   # OK
python -m bird_fine.data.download          # NG
```

---

## データ収集系

### Q. 種によって録音が 0 件

- `config.yaml` の `fallback_worldwide: true` を確認
- 品質条件を緩和: `quality: "A B"`
- `species_master.csv` の `en_birdnet` / Xeno-canto 表記が正しいか確認
- それでも不足なら YouTube 補完（`docs/data_guide.md` 参照）

### Q. やり直したい / 再 DL

xcapi は `data/raw/{Species}/xcapi_runs.json` で DL 済み ID を管理:

```bash
rm -rf data/raw/Mallard   # 特定種
rm -rf data/raw           # 全部
```

### Q. OOD データの音声が前処理でスキップされる

`download_ood.py` は xcapi のサブディレクトリ構造（`{Species}/{Scientific_name}/`）に
ファイルを置く。前処理は `rglob` で再帰検索するので問題ないが、
`--metadata-only` だけでは音声がDLされないケースがある。本DLを実行すること:

```bash
uv run python -m bird_fine.data.download_ood --tiers 3   # 該当 tier を本DL
```

---

## 前処理系

### Q. `librosa.load` で音声が読めない

| 原因 | 対処 |
|---|---|
| mp3 破損 | 該当ファイル削除 → 再 DL |
| ffmpeg 不足 | `sudo apt install ffmpeg` |

### Q. Windows で作った CSV が Linux で動かない

Windows 生成の `data/splits/*.csv` は `file_path` がバックスラッシュ区切り。
Linux では sed で一括変換:

```bash
sed -i 's|\\|/|g' data/splits/*.csv
```

`dataset.py` / `confusion_audio.py` は `.replace("\\", "/")` で防御済みだが、
新規 CSV を Windows から持ち込んだ場合は確認すること。

---

## 学習系

### Q. CUDA out of memory (OOM)

対処の優先順:

1. `--batch-size 2`
2. `gradient_accumulation_steps` を上げて実効 batch を維持
3. `gradient_checkpointing: true` を確認
4. `fp16: true` を確認
5. `nvidia-smi` で他プロセスの VRAM 使用を確認

### Q. 位置埋め込みの次元ミスマッチエラー

```
RuntimeError: The size of tensor a (350) must match tensor b (1214)
```

3s チャンク（max_length=304）に対し、事前学習モデルが 10s 用の位置埋め込み
（1214 次元）を持つために発生する。`train.py` の `resize_position_embeddings()`
が呼ばれているか確認。`config.yaml` の `model.feature_extractor_max_length: 304` が必要。

### Q. evaluate.py で位置埋め込みエラー

```
MISMATCH: position_embeddings ckpt torch.Size([1, 350, 768]) vs model [1, 1214, 768]
```

保存済みモデルの `config.json` に `max_length` が正しく書かれていない。
`train.py` は `model.config.max_length = max_length` で保存時に記録するが、
古いチェックポイントは手動修正が必要:

```bash
uv run python -c "
import json
p = 'models/ast-duck-v10/config.json'
d = json.load(open(p)); d['max_length'] = 304
json.dump(d, open(p,'w'), indent=2)
"
```

### Q. Outlier Exposure で 8 種の精度が壊れる

Double Dipping（Sampler + Loss 重みの二重補正）が原因の可能性。
`config.yaml` の `other_class.loss_alpha` は固定値のみ。
データ数ベースの重みを Loss に掛けていないか `train.py` の `DuckTrainer.compute_loss` を確認。

### Q. OE 学習で "other" ばかり予測する

`loss_alpha` が高すぎる。1.5 から始めて、8 種の `f1_macro_8class` が
下がるなら 1.2 に下げる。

### Q. `metric_for_best_model` が見つからない

OE 学習時は `f1_macro_8class` を指定するが、これは `make_compute_metrics()`
が "other" クラスを含む場合のみ出力する。8 種学習時は `f1_macro` を使う。

---

## 評価・推論系

### Q. accuracy/F1 が異常に高い（> 0.95）

leakage の可能性。split が録音 ID 単位か確認:

```bash
uv run python -c "import pandas as pd; t=set(pd.read_csv('data/splits/train.csv')['xc_id']); e=set(pd.read_csv('data/splits/test.csv')['xc_id']); print('overlap:', len(t&e))"
# overlap: 0 が正しい
```

### Q. energy gate が全部弾く / 全部通す

`species_taxonomy.yaml` の `energy_threshold` を確認:

- 全部弾く → 閾値が高すぎる。`ood_eval.py` で再算出
- 全部通す → 閾値が低すぎる or `null`（gate 無効）

energy スコアの符号規約に注意（高い = in-distribution）。
`predict.py` と `ood_eval.py` で同じ convention（`logsumexp(logits)`、正値）を使う。

### Q. 推論結果が全部同じ種

| 原因 | 対処 |
|---|---|
| label_map.csv が学習時と違う | `models/{run}/label_map.csv` を確認 |
| 入力が短すぎ | min_chunk_duration_sec（1s）未満は空 |

### Q. CPU 強制推論したい

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m bird_fine.inference.predict --audio file.wav
```

---

## ONNX / デプロイ系

### Q. ONNX エクスポートで TracerWarning

```
TracerWarning: Converting a tensor to a Python boolean ...
is_causal = query.shape[2] > 1 and ...
```

SDPA の `is_causal` 判定が入力形状依存でトレース時に定数化される。
3s チャンク固定（304 フレーム）運用では無害。入力長を変える場合のみ要再検証。

### Q. ONNX と PyTorch の出力がずれる

`export_onnx.py` の検証で max_abs_diff を確認:

```bash
uv run python -m bird_fine.inference.export_onnx
# batch=1/2 とも diff < 1e-4 なら OK
```

差が大きい場合は FP16 変換が原因のことが多い。FP32 で再エクスポートして切り分ける。

---

## 環境系（Ubuntu 移行関連）

### Q. Windows パーティションのデータにアクセスしたい

```bash
sudo mount -t ntfs3 -o ro /dev/nvme1n1p3 /mnt/win
```

### Q. TensorBoard が表示されない

```bash
# ポート競合時
uv run tensorboard --logdir models/ast-duck-v11/runs --port 6007
```

### Q. BirdProject（TF）の GPU が認識されない

`tensorflow[and-cuda]` 同梱の CUDA ライブラリパスを `LD_LIBRARY_PATH` に
追加する必要がある（`.bashrc` で設定済み）。新しいシェルで:

```bash
source ~/.bashrc
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## デバッグ Tips

### `--dry-run` を活用

```bash
uv run python -m bird_fine.training.train --dry-run
```

各種 6 チャンクのサブセットで 1 epoch 完走するか確認。OOM/import/Dataset バグが即判明。
**注意**: dry-run は層化サンプリング（`groupby("species").head(6)`）。
flat な head だと先頭種に偏って自明に高精度が出る（run03 で踏んだ罠）。

### 各種マスタ・設定の整合性確認

```bash
uv run python -c "
import pandas as pd, yaml
m = pd.read_csv('data/species_master.csv')
print(m['status'].value_counts())
print('groups:', m['group'].unique())
"
```

---

## それでも解決しない時

1. エラー全文を確認
2. `dry-run` で再現するか
3. `uv pip list` でバージョン確認
4. `nvidia-smi` の出力確認
5. `docs/journal.md` で過去の同様トラブルを検索
