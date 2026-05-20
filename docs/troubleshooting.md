# Troubleshooting

よくあるトラブルと対処法。

---

## セットアップ系

### Q. `uv sync` でtorchが入らない / CPU版になる

`pyproject.toml` の `[tool.uv.sources]` で `pytorch-cu124` インデックス指定済み。
動作しない場合は手動で：

```powershell
uv add torch torchaudio --index https://download.pytorch.org/whl/cu124
```

確認：
```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# True 12.4 が出ればOK
```

### Q. `XENO_CANTO_API_KEY が未設定` エラー

`.env` ファイルがプロジェクトルートにあるか確認：

```powershell
Get-Content .env
# XENO_CANTO_API_KEY=xxxxxxxx と出ればOK
```

無ければ作成：
```env
XENO_CANTO_API_KEY=your_key_here
```

APIキーは https://xeno-canto.org/account/api で取得。

### Q. `ModuleNotFoundError: bird_fine` が出る

uv環境外で動かしている可能性。必ず `uv run` 経由で：

```powershell
uv run python -m bird_fine.data.download
# ❌ python -m bird_fine.data.download  ←これだと動かない
```

---

## データ収集系

### Q. 種によって録音が0件

カモ類は地域差があるため、種によっては日本国内録音がない場合あり。
config.yamlでフォールバック設定を確認：

```yaml
download:
  fallback_worldwide: true   # ← Japanで0件ならworldwideで再検索
```

それでも0件なら：
- 品質条件を緩和: `quality: "A B"`
- 英名のスペル確認（Xeno-cantoの正確な表記が必要）

### Q. ダウンロードが途中で止まる / タイムアウト

xcapiは1音声30秒タイムアウト。原因と対処：

| 原因 | 対処 |
|---|---|
| ネット不安定 | 再実行（DL済みはスキップされる） |
| 大量同時DL | 種を分けて実行: `--species Mallard` |
| Xeno-canto側Rate Limit | しばらく待って再実行 |

### Q. 重複DLしたい / やり直したい

xcapiは `data/raw/{Species}/xcapi_runs.json` でDL済みIDを管理。
完全に再DLしたいなら：

```powershell
# 特定種だけ削除
Remove-Item -Recurse -Force data\raw\Mallard

# 全部やり直し
Remove-Item -Recurse -Force data\raw
```

---

## 前処理系

### Q. `librosa.load` で音声が読めない

エラー例: `EOFError`, `soundfile.LibsndfileError`

| 原因 | 対処 |
|---|---|
| mp3ファイル破損 | 該当ファイルを削除して再DL |
| ffmpegが必要 | `winget install ffmpeg` |
| 特殊コーデック | mp3/wav以外は元データから除外 |

### Q. チャンク数が想定より少ない

`config.yaml` の `min_chunk_duration_sec` を確認：

- デフォルト3秒 → 3秒未満の録音はスキップ
- 短い録音を活かしたいなら1秒に下げる（ただし情報量低下に注意）

### Q. ディスク容量が足りない

- raw音声: 種あたり数百MB（mp3）
- processed: raw × 約3倍（wav非圧縮、16kHz）

合計 5〜10GB 想定。outputディレクトリを別ドライブに変更したいなら config.yaml の `output_dir` を絶対パスで指定。

---

## 学習系

### Q. CUDA out of memory (OOM)

エラー例:
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate XX MiB
```

**対処の優先順**（上から試す）：

1. **batch_size を下げる**: `--batch-size 2`
2. **gradient_accumulation_steps を上げる** で実効batchを維持：
   ```yaml
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 8   # effective = 16
   ```
3. **gradient_checkpointing を確認**: config.yamlで `true` か
4. **fp16 を確認**: config.yamlで `true` か
5. **eval batch size も下げる**: `per_device_eval_batch_size: 4`
6. **DataLoader workers を 0 に**: Windowsはデフォルト0（既定）
7. **不要なプロセスを終了**: `nvidia-smi` で他プロセスがVRAM使ってないか確認

それでもダメなら：
- メルスペクトログラムの時間長を短縮（10秒 → 5秒、ただし精度低下リスク）

### Q. 学習が進まない（loss下がらない）

| 症状 | 原因候補 | 対処 |
|---|---|---|
| loss = NaN | lr高すぎ / fp16不安定 | lrを下げる(1e-5)、fp16 → fp32 |
| loss振動 | lr高すぎ | warmup増やす、lr下げる |
| loss一定 | データ問題 | label_map.csv、Dataset確認 |
| acc = 1/num_classes 付近 | 学習されてない | データ確認、層が固定されてないか |

### Q. 学習が遅すぎる

| 状態 | 確認 |
|---|---|
| GPU使ってない | `nvidia-smi` で確認、TrueなのにCPUなら環境問題 |
| CPU100%でGPU空き | DataLoader bottleneck、`num_workers=0` ならWindowsの制約 |
| 1 step が異常に遅い | fp16が無効化されてないか、grad_checkpointing過剰か |

### Q. EarlyStoppingで早く止まりすぎる

```yaml
early_stopping_patience: 5  # 3 → 5 に増やす
```

ただし、本当に過学習が進んでいる場合は止めた方が良い。TensorBoardで val_loss を見て判断。

### Q. 「load_best_model_at_end」がエラー

```
ValueError: --load_best_model_at_end requires the save and eval strategy to match
```

config.yamlで揃える：
```yaml
eval_strategy: "epoch"
save_strategy: "epoch"
```

---

## 評価・推論系

### Q. 評価のaccuracy/F1が異常に高い（> 0.95）

要注意。leakageの可能性：

1. **splitが録音ID単位か確認**
   - `data/splits/{train,test}.csv` で同じ `xc_id` が両方に出ていないか
   ```powershell
   uv run python -c "import pandas as pd; t=set(pd.read_csv('data/splits/train.csv')['xc_id']); e=set(pd.read_csv('data/splits/test.csv')['xc_id']); print('overlap:', len(t&e))"
   # overlap: 0 が正しい
   ```

2. **同じ録音者・場所バイアス**
   - 一部の録音者が大量に同一種を投稿 → モデルが「録音の癖」を学習
   - metadata.csvのrec(録音者)列を見て分布チェック

### Q. 推論結果が全部同じ種になる

| 原因 | 対処 |
|---|---|
| label_map.csvが学習時と違う | `models/ast-duck/label_map.csv` を確認 |
| モデルが学習されてない | val accが高いか再確認 |
| 入力音声が短すぎ | 3秒以上必要 |

### Q. 推論時にCUDA OOM

推論はバッチ1で動くはずだが、長すぎる音声をチャンク分割した結果メモリ不足になることも。
`predict.py` の chunk処理を1チャンクずつに直すか、CPUで推論：

```powershell
# 強制CPU推論
$env:CUDA_VISIBLE_DEVICES=""
uv run python -m bird_fine.inference.predict --audio path/to/file.wav
```

---

## 環境系

### Q. PowerShellで `&&` が動かない

Windows PowerShell 5.1 は `&&` 非対応。代わりに：

```powershell
# NG: cmd1 && cmd2
# OK: cmd1; if ($?) { cmd2 }
uv run python -m bird_fine.data.download; if ($?) { uv run python -m bird_fine.data.preprocess }
```

### Q. パス区切り `/` vs `\`

PowerShellでも Python内部でも `/` `\` 両対応。pathlib.Pathを使えば自動解決。
**.envや設定ファイルに絶対パス書く時は二重バックスラッシュ** に注意：

```yaml
# NG（Windowsで escape interpretation）
output_dir: "C:\Users\..."

# OK
output_dir: "C:/Users/..."
# または
output_dir: "C:\\Users\\..."
```

### Q. TensorBoardが表示されない

```powershell
# プロセスが残ってないか確認
Get-Process | Where-Object { $_.Name -like "*tensorboard*" }

# ポート競合
uv run tensorboard --logdir models/ast-duck/runs --port 6007
```

---

## デバッグTips

### `--dry-run` を活用

```powershell
uv run python -m bird_fine.training.train --dry-run
```

これで1 epoch・50サンプルで完走するか確認。OOM/import error/Dataset bugなどが即座に判明。

### ログレベル上げる

学習時のverbose出力：
```yaml
logging_steps: 5   # 20 → 5 に下げる
```

### Python REPLで部分検証

```powershell
uv run python
>>> from bird_fine.data.dataset import build_datasets
>>> from pathlib import Path
>>> train_ds, val_ds, test_ds, lm = build_datasets(Path("data/splits"), "MIT/ast-finetuned-audioset-10-10-0.4593")
>>> sample = train_ds[0]
>>> sample["input_values"].shape   # (1024, 128) 程度ならOK
```

---

## それでも解決しない時

1. **エラー全文をコピー**してissueに記録
2. `dry-run` で同じエラーが出るか確認
3. `uv pip list` で依存関係のバージョン確認
4. `nvidia-smi` の出力を確認
5. transformers / torch のバージョン互換を疑う
