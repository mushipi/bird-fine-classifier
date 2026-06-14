"""OOD/偽陽性 監査: AST(運用) と Perch(教師) を同じ OOD 集合で同じ土俵比較。

測るもの:
  - OOD AUROC（max-softmax / energy）を AST vs Perch で比較（高い=OODをよく弾ける）
  - 種別/tier別リーク率（運用閾値で OOD が in-dist と誤受理される率→何に化けるか。カルガモ注目）
  - 録音単位の動作点（chunk閾値を録音平均で再評価）+ tier別 E2E 偽陽性率

in-dist = test.csv（10カモ）, OOD = data/ood_processed（Perch埋め込みは data/embeddings/perch_ood/ood.npz）。

実行（本体.venv, GPU）:
  .venv/bin/python tools/ood_fp_audit.py --ast-model models/ast-duck-C-kd-soup --ast-threshold 2.948
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "probe_sweep"))
from bird_fine.data.dataset import build_datasets, collate_fn  # noqa: E402
from bird_fine.embeddings import io_utils  # noqa: E402
from bird_fine.training.train_probe import standardize_apply, standardize_fit  # noqa: E402
from bird_fine.inference.predict import compute_energy  # noqa: E402
import run_sweep as rs  # noqa: E402
from transformers import ASTForAudioClassification  # noqa: E402


def rec_id(path_or_xc: str) -> str:
    return re.sub(r"_chunk\d+\.wav$", "", str(path_or_xc).split("/")[-1])


def energy_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return np.array([compute_energy(row, T) for row in logits])


def auroc(score_in: np.ndarray, score_ood: np.ndarray) -> float:
    """in-dist=1 / OOD=0 を score が高いほど in-dist と分離できるか。"""
    y = np.r_[np.ones(len(score_in)), np.zeros(len(score_ood))]
    s = np.r_[score_in, score_ood]
    return float(roc_auc_score(y, s))


# ---------- AST: in-dist(test) と OOD(wav) を推論して logits ----------
def ast_logits_indist(model, ds, device):
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)
    out = []
    with torch.no_grad():
        for b in loader:
            out.append(model(input_values=b["input_values"].to(device)).logits.cpu().numpy())
    return np.concatenate(out)


def ast_logits_ood(model, fe, wavs, device, sr):
    import soundfile as sf
    out = []
    with torch.no_grad():
        for w in wavs:
            a, s = sf.read(str(w), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            iv = fe(a, sampling_rate=sr, return_tensors="pt")["input_values"].to(device)
            out.append(model(input_values=iv).logits.cpu().numpy()[0])
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ast-model", default="models/ast-duck-C-kd-soup")
    ap.add_argument("--ast-threshold", type=float, default=2.948)
    ap.add_argument("--perch-emb", default="data/embeddings/perch")
    ap.add_argument("--ood-emb", default="data/embeddings/perch_ood/ood.npz")
    ap.add_argument("--ood-root", default="data/ood_processed")
    ap.add_argument("--duck-order", default="data/embeddings/teacher_proba/duck_order.csv")
    ap.add_argument("--splits-dir", default=None,
                    help="in-dist test の splits（省略時 data/splits）。群別は data/splits-<group>")
    ap.add_argument("--in-family", default=None,
                    help="群の科（同科他種OOD判定用）。省略時は target種の科を master から自動導出")
    ap.add_argument("--config", default=None,
                    help="設定ファイル（省略時 config.yaml）。群別は config-<group>.yaml")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(args.config).read_text() if args.config
                         else (PROJECT_ROOT / "config.yaml").read_text())
    duck_order = pd.read_csv(PROJECT_ROOT / args.duck_order)["species"].tolist()
    splits_dir = (PROJECT_ROOT / args.splits_dir) if args.splits_dir else (PROJECT_ROOT / "data" / "splits")

    # ===== AST =====
    print(f"[AST] load + 推論（in-dist test={splits_dir.name} + OOD）", flush=True)
    _, _, test_ds, _ = build_datasets(
        splits_dir=splits_dir, pretrained=cfg["model"]["pretrained"],
        project_root=PROJECT_ROOT, max_length=int(cfg["model"].get("feature_extractor_max_length", 1024)),
        duck_order=duck_order)
    from transformers import ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained(cfg["model"]["pretrained"],
                                             max_length=int(cfg["model"].get("feature_extractor_max_length", 1024)))
    ast = ASTForAudioClassification.from_pretrained(str(PROJECT_ROOT / args.ast_model)).to(device).eval()

    ast_in_logits = ast_logits_indist(ast, test_ds, device)
    ood_wavs = sorted((PROJECT_ROOT / args.ood_root).rglob("*.wav"))
    ast_ood_logits = ast_logits_ood(ast, fe, ood_wavs, device, fe.sampling_rate)
    ood_tier = np.array([w.relative_to(PROJECT_ROOT / args.ood_root).parts[0] for w in ood_wavs])
    ood_sp = np.array([w.relative_to(PROJECT_ROOT / args.ood_root).parts[1] for w in ood_wavs])
    ood_rec = np.array([rec_id(str(w)) for w in ood_wavs])
    in_rec = test_ds.df["xc_id"].astype(str).to_numpy()

    # 陳腐化対策: ood_processed に残る「今や標的の種」(Gadwall/ウミアイサ等)を真OODから除外
    keep = ~np.isin(ood_sp, duck_order)
    n_drop = int((~keep).sum())
    print(f"[filter] OOD から標的種を除外: {n_drop} chunks（{sorted(set(ood_sp[~keep]))}）→ 真OOD {int(keep.sum())}", flush=True)
    ast_ood_logits = ast_ood_logits[keep]; ood_tier = ood_tier[keep]; ood_sp = ood_sp[keep]; ood_rec = ood_rec[keep]
    ood_keep_mask = keep  # Perch 側にも同じ順序で適用

    ast_in_e = energy_np(ast_in_logits); ast_ood_e = energy_np(ast_ood_logits)
    ast_in_m = _softmax(ast_in_logits).max(1); ast_ood_m = _softmax(ast_ood_logits).max(1)
    ast_ood_pred = np.array(duck_order)[ast_ood_logits.argmax(1)]

    # ===== Perch（教師: 複数arm平均 / energyは linear logits）。埋め込み未整備の群は skip（AST単独監査）=====
    perch_test_npz = PROJECT_ROOT / args.perch_emb / "test.npz"
    perch_ood_npz = PROJECT_ROOT / args.ood_emb
    do_perch = perch_test_npz.exists() and perch_ood_npz.exists()
    perch_in_m = perch_ood_m = perch_in_e = perch_ood_e = None
    if not do_perch:
        print(f"[Perch] skip（{args.perch_emb}/test.npz か {args.ood_emb} が無い）→ AST 単独で監査", flush=True)
    else:
        print("[Perch] プローブ学習 + 採点", flush=True)
        tr = io_utils.load_embeddings(PROJECT_ROOT / args.perch_emb / "train.npz")
        te = io_utils.load_embeddings(perch_test_npz)
        ood = np.load(perch_ood_npz, allow_pickle=True)
        mean, std = standardize_fit(tr["X"]); ytr = tr["y"]; nc = len(tr["label_names"])
        Xtr = standardize_apply(tr["X"], mean, std)
        Xte = standardize_apply(te["X"], mean, std)
        Xood = standardize_apply(ood["X"], mean, std)
        operative = sorted(np.unique(ytr).tolist())
        va = io_utils.load_embeddings(PROJECT_ROOT / args.perch_emb / "val.npz")
        Xva = standardize_apply(va["X"], mean, std)
        lin, _ = rs.train_torch("linear", Xtr, ytr, Xva, va["y"], nc, operative, 0.0, device)
        mlp, _ = rs.train_torch("mlp", Xtr, ytr, Xva, va["y"], nc, operative, 5e-3, device)
        clf = rs.fit_sklearn(Xtr, ytr, 0.01)

        def perch_score(X):
            proba = (rs.torch_proba(lin, X, device) + rs.torch_proba(mlp, X, device)
                     + rs.sklearn_proba_full(clf, X, nc)) / 3.0
            lin_logits = lin(torch.from_numpy(X).to(device)).detach().cpu().numpy()  # energy用
            return proba, lin_logits

        p_in, le_in = perch_score(Xte)
        p_ood, le_ood = perch_score(Xood)
        # AST と同じ「真OOD」マスクを適用（標的種除外）
        p_ood = p_ood[ood_keep_mask]; le_ood = le_ood[ood_keep_mask]
        perch_in_m = p_in[:, operative].max(1); perch_ood_m = p_ood[:, operative].max(1)
        perch_in_e = energy_np(le_in[:, operative]); perch_ood_e = energy_np(le_ood[:, operative])

    # ===== 比較1: OOD AUROC =====
    print("\n===== OOD AUROC（高い=OODをよく弾ける, in-dist test vs OOD全体）=====", flush=True)
    print(f"  {'model':8s} {'max-softmax':>12s} {'energy':>10s}", flush=True)
    print(f"  {'AST':8s} {auroc(ast_in_m, ast_ood_m):12.4f} {auroc(ast_in_e, ast_ood_e):10.4f}", flush=True)
    if do_perch:
        print(f"  {'Perch':8s} {auroc(perch_in_m, perch_ood_m):12.4f} {auroc(perch_in_e, perch_ood_e):10.4f}", flush=True)

    # ===== 比較2: AST 運用閾値での種別リーク率（chunk & 録音単位）=====
    th = args.ast_threshold
    print(f"\n===== AST 種別リーク率 @ energy>={th}（OOD種が誤受理される率）=====", flush=True)
    print(f"  {'tier/species':32s} {'chunk_leak':>10s} {'rec_leak':>9s} {'化け先top':>16s}", flush=True)
    df = pd.DataFrame({"tier": ood_tier, "sp": ood_sp, "rec": ood_rec,
                       "e": ast_ood_e, "pred": ast_ood_pred})
    for (tier, sp), g in df.groupby(["tier", "sp"]):
        chunk_leak = float((g["e"] >= th).mean())
        rec = g.groupby("rec")["e"].mean()
        rec_leak = float((rec >= th).mean())
        top = g[g["e"] >= th]["pred"].value_counts()
        topstr = top.index[0] if len(top) else "-"
        if chunk_leak > 0.001:
            print(f"  {tier+'/'+sp:32.32s} {chunk_leak:10.3f} {rec_leak:9.3f} {topstr:>16.16s}", flush=True)

    # ===== 比較3: tier別 E2E 偽陽性率（録音単位 受理率）=====
    print(f"\n===== tier別 E2E 偽陽性率（録音単位 mean energy >= {th} で受理=FP）=====", flush=True)
    for tier in ["tier1", "tier2", "tier3"]:
        m = ood_tier == tier
        rec = pd.DataFrame({"rec": ood_rec[m], "e": ast_ood_e[m]}).groupby("rec")["e"].mean()
        print(f"  {tier}: 録音{len(rec)}  FP率(受理)={float((rec>=th).mean()):.3f}", flush=True)
    # ===== 録音単位の閾値再導出（在群最優先: 非同科OOD基準・保持>=0.90）=====
    # OOD を family で「同科他種(in_family) / 非同科」に分割
    sm = pd.read_csv(PROJECT_ROOT / "data" / "species_master.csv")
    norm = lambda s: str(s).replace(" ", "_")
    fam_map = {}
    for _, r in sm.iterrows():
        for col in ("en_inat", "en_birdnet"):
            if pd.notna(r[col]):
                fam_map[norm(r[col])] = r["family"]
    # 群の科: --in-family 指定 or target種(duck_order)の科を多数決で自動導出
    if args.in_family:
        in_family = args.in_family
    else:
        fams = [fam_map.get(norm(s)) for s in duck_order if fam_map.get(norm(s))]
        in_family = max(set(fams), key=fams.count) if fams else "Anatidae"
    print(f"[CFG] in_family（同科判定）= {in_family}", flush=True)
    ood_fam = np.array([fam_map.get(s, "?") for s in ood_sp])
    n_unk = int((ood_fam == "?").sum())
    if n_unk:
        print(f"[warn] family 未突合 {n_unk} chunks（{sorted(set(ood_sp[ood_fam=='?']))}）→ 非同科扱い", flush=True)

    rin = pd.DataFrame({"rec": in_rec, "e": ast_in_e}).groupby("rec")["e"].mean().to_numpy()
    oodf = pd.DataFrame({"rec": ood_rec, "e": ast_ood_e, "fam": ood_fam})
    rec_ood = oodf.groupby("rec").agg(e=("e", "mean"), fam=("fam", "first"))
    r_anas = rec_ood[rec_ood["fam"] == in_family]["e"].to_numpy()   # 同科他種OOD
    r_non = rec_ood[rec_ood["fam"] != in_family]["e"].to_numpy()    # 非同科OOD

    def _au(a, b):
        return auroc(a, b) if len(a) and len(b) else float("nan")

    print(f"\n===== 録音単位 動作点（在群最優先: 非同科OOD基準・保持>=0.90）=====", flush=True)
    print(f"  録音数: 在群test={len(rin)}  同科他種OOD={len(r_anas)}  非同科OOD={len(r_non)}", flush=True)
    print(f"  録音単位AUROC(energy): 在群 vs 非同科={_au(rin, r_non):.4f}  "
          f"在群 vs 同科他種={_au(rin, r_anas):.4f}", flush=True)
    def at(t):
        return float((rin >= t).mean()), float((r_non >= t).mean()), float((r_anas >= t).mean())

    def thr_for_retention(target):
        grid = np.linspace(rin.min(), rin.max(), 800)
        cand = [t for t in grid if (rin >= t).mean() >= target]
        return float(max(cand)) if cand else float(grid[0])

    # Youden 最適点（真カモ保持 - 非カモ類FP を最大化）
    grid = np.linspace(min(rin.min(), r_non.min()), max(rin.max(), r_non.max()), 800)
    j = [(float((rin >= t).mean()) - float((r_non >= t).mean()), t) for t in grid]
    youden_t = float(max(j)[1])

    print(f"  {'動作点':14s} {'閾値':>7s} {'在群保持':>9s} {'非同科FP':>9s} {'同科他種漏れ':>11s}", flush=True)
    rows = [("現行", th), ("保持>=0.95", thr_for_retention(0.95)), ("保持>=0.90", thr_for_retention(0.90)),
            ("保持>=0.85", thr_for_retention(0.85)), ("保持>=0.80", thr_for_retention(0.80)),
            ("Youden最適", youden_t)]
    for name, t in rows:
        hold, fp_non, leak = at(t)
        print(f"  {name:14s} {t:7.3f} {hold:9.3f} {fp_non:9.3f} {leak:11.3f}", flush=True)
    print(f"\n  ※在群保持と非同科FPはトレードオフ。同科他種漏れは層②(分類)課題として受容。", flush=True)


def _softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    main()
