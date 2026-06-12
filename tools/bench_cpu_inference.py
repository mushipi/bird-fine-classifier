"""GT105(CPU) での AST KD-soup 推論ベンチ（運用機デプロイ前提検証）。

モデルを1回ロードし、3秒チャンクの推論 latency / throughput / メモリ を実測。
.venv-cpu/bin/python tools/bench_cpu_inference.py --model models/ast-duck-C-kd-soup --chunks bench_chunks
"""
from __future__ import annotations

import argparse
import glob
import resource
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KB→MB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ast-duck-C-kd-soup")
    ap.add_argument("--chunks", default="bench_chunks")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    pretrained = cfg["model"]["pretrained"]
    max_length = int(cfg["model"].get("feature_extractor_max_length", 1024))
    print(f"[env] torch {torch.__version__} cpu={not torch.cuda.is_available()} "
          f"threads={torch.get_num_threads()}  RSS={rss_mb():.0f}MB", flush=True)

    t0 = time.perf_counter()
    fe = ASTFeatureExtractor.from_pretrained(pretrained, max_length=max_length)
    model = ASTForAudioClassification.from_pretrained(str(ROOT / args.model)).eval()
    load_s = time.perf_counter() - t0
    print(f"[load] model+FE = {load_s:.1f}s  RSS={rss_mb():.0f}MB  "
          f"classes={model.config.num_labels}", flush=True)

    wavs = sorted(glob.glob(str(ROOT / args.chunks / "*.wav")))[:30]
    sr = fe.sampling_rate
    feats = []
    t0 = time.perf_counter()
    for w in wavs:
        a, _ = librosa.load(w, sr=sr, mono=True)
        feats.append(fe(a, sampling_rate=sr, return_tensors="pt")["input_values"])
    fe_s = (time.perf_counter() - t0) / max(len(wavs), 1)
    print(f"[data] {len(wavs)} chunks, 前処理(FE+load) {fe_s*1000:.0f}ms/chunk", flush=True)

    X = torch.cat(feats, 0)  # (N, T, F)

    def bench(threads: int):
        torch.set_num_threads(threads)
        with torch.no_grad():
            _ = model(input_values=X[:1])  # warmup
            # 単発 latency
            ts = []
            for _ in range(args.reps):
                for i in range(len(X)):
                    t = time.perf_counter()
                    model(input_values=X[i:i + 1])
                    ts.append(time.perf_counter() - t)
            single_ms = np.median(ts) * 1000
            # バッチ throughput
            t = time.perf_counter()
            model(input_values=X)
            batch_s = time.perf_counter() - t
        return single_ms, len(X) / batch_s

    print(f"\n  {'threads':>8s} {'単発latency':>14s} {'batch throughput':>18s}", flush=True)
    for th in (1, 4, 16):
        s_ms, thr = bench(th)
        print(f"  {th:>8d} {s_ms:>11.0f}ms {thr:>13.1f} ch/s", flush=True)

    print(f"\n[peak] RSS={rss_mb():.0f}MB", flush=True)


if __name__ == "__main__":
    main()
