"""ONNX エクスポート検証スクリプト。

run11 モデルが ONNX としてクリーンに書き出せるかを最小構成で確認する。
以下の3点を検証する:

  1. torch.onnx.export が通るか
  2. onnxruntime で PyTorch と同じ出力が出るか（最大絶対誤差 < 1e-4）
  3. 動的バッチサイズが機能するか（batch=1 と batch=2 で確認）

使い方:
    uv run python -m bird_fine.inference.export_onnx
    uv run python -m bird_fine.inference.export_onnx --model-dir models/ast-duck-v11
    uv run python -m bird_fine.inference.export_onnx --output models/ast-duck-v11/model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import ASTForAudioClassification

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_onnx(
    model_dir: Path,
    output_path: Path,
    opset: int = 17,
) -> bool:
    """ONNX エクスポートと数値検証を実行。成功なら True を返す。"""
    import onnx
    import onnxruntime as ort

    config = load_config()
    model_cfg = config["model"]
    max_length = int(model_cfg.get("feature_extractor_max_length", 1024))

    # --- モデルロード ---
    print(f"[LOAD] {model_dir}")
    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    model.eval()
    num_labels = model.config.num_labels
    print(f"  num_labels={num_labels}  max_length={max_length}")

    # --- ダミー入力 (batch=1, time=304, freq=128) ---
    dummy = torch.randn(1, max_length, 128)

    # --- PyTorch での基準出力を取得 ---
    with torch.no_grad():
        pt_logits = model(input_values=dummy).logits.numpy()
    print(f"  PyTorch logits shape: {pt_logits.shape}")

    # --- ONNX エクスポート ---
    print(f"\n[EXPORT] opset={opset} → {output_path}")
    try:
        torch.onnx.export(
            model,
            {"input_values": dummy},
            str(output_path),
            opset_version=opset,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        print("  export: OK")
    except Exception as e:
        print(f"  [FAIL] export エラー: {e}")
        return False

    # --- ONNX モデルの整合性チェック ---
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  onnx.checker: OK")
    except Exception as e:
        print(f"  [FAIL] onnx.checker エラー: {e}")
        return False

    # --- onnxruntime で数値一致確認 ---
    print("\n[VERIFY] onnxruntime との数値一致確認")
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

    # batch=1
    ort_logits = sess.run(["logits"], {"input_values": dummy.numpy()})[0]
    diff = np.abs(pt_logits - ort_logits).max()
    status = "OK" if diff < 1e-4 else "WARN"
    print(f"  batch=1: max_abs_diff={diff:.2e}  [{status}]")
    if diff >= 1e-4:
        print(f"  [WARN] 差が大きい。FP16 変換後に再確認を推奨")

    # batch=2（動的バッチ確認）
    dummy2 = torch.randn(2, max_length, 128)
    with torch.no_grad():
        pt2 = model(input_values=dummy2).logits.numpy()
    ort2 = sess.run(["logits"], {"input_values": dummy2.numpy()})[0]
    diff2 = np.abs(pt2 - ort2).max()
    status2 = "OK" if diff2 < 1e-4 else "WARN"
    print(f"  batch=2: max_abs_diff={diff2:.2e}  [{status2}]")

    # energy スコアの一致確認（onnxruntime 側で計算）
    logits_t = torch.tensor(ort_logits)
    energy_ort = float(torch.logsumexp(logits_t, dim=-1).item())
    logits_pt = torch.tensor(pt_logits)
    energy_pt = float(torch.logsumexp(logits_pt, dim=-1).item())
    ediff = abs(energy_pt - energy_ort)
    print(f"  energy score: pt={energy_pt:.4f}  ort={energy_ort:.4f}  diff={ediff:.2e}  [{'OK' if ediff < 1e-3 else 'WARN'}]")

    # --- ファイルサイズ ---
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n[SIZE] {output_path.name}: {size_mb:.1f} MB")
    if size_mb > 400:
        print("  [NOTE] 400MB超。Jetson Nanoのメモリに注意（FP16化で約半分になる）")

    all_ok = diff < 1e-4 and diff2 < 1e-4 and ediff < 1e-3
    print(f"\n{'[PASS] ONNX エクスポート検証完了' if all_ok else '[WARN] 一部確認が必要'}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    config = load_config()
    train_cfg = config["training"]
    model_dir = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / train_cfg["output_dir"]

    if not model_dir.exists():
        print(f"[ERROR] {model_dir} が見つからない")
        sys.exit(1)

    output_path = Path(args.output) if args.output else model_dir / "model.onnx"
    success = verify_onnx(model_dir, output_path, opset=args.opset)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
