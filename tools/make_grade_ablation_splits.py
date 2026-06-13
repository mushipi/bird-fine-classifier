"""grade アブレーション用スプリット生成（標準フロー資産）。

ある群について、1種(--ablate)の Xeno-canto grade を A / A+B / A+B+C と振った3スプリットを出力する。
**val/test は全アームで完全共通**（ablate種は grade-A から取り、grade-B/C は train にのみ入れる）。
これで「grade 緩和が train に効くか」を、評価を歪めずに切り出せる。

grade は data/raw/<種>/metadata.csv の file-name を xcapi と同じサニタイズで source_file に突き合わせて取得（100%結合）。

実行:
  .venv/bin/python tools/make_grade_ablation_splits.py --config config-crow.yaml \
      --ablate Large-billed_Crow --out-prefix data/splits-crow
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sanitize(s: str) -> str:
    for c in '<>:"/\\|?*':
        s = s.replace(c, "_")
    return s.strip(". ")


def grade_map(species_dir: Path) -> dict[str, str]:
    """source_file basename -> grade（metadata.csv の file-name をサニタイズして突合）。"""
    md = pd.read_csv(species_dir / "metadata.csv")
    return {sanitize(str(r["file-name"])): str(r["q"]) for _, r in md.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ablate", required=True, help="grade を振る種（processed ディレクトリ名）")
    ap.add_argument("--out-prefix", required=True, help="出力 splits ディレクトリの接頭辞")
    ap.add_argument("--arms", nargs="+", default=["A", "AB", "ABC"])
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    pp = cfg["preprocessing"]
    seed = int(pp["random_seed"])
    vr, tr = float(pp["val_ratio"]), float(pp["test_ratio"])
    raw_dir = ROOT / cfg["download"]["output_dir"]
    targets = [s["en"].replace(" ", "_").replace("-", "-") for s in cfg["target_species"]]
    # processed のディレクトリ名は en をそのまま（スペース→アンダースコアは download 側命名と一致）
    targets = [s["en"].replace(" ", "_") for s in cfg["target_species"]]

    ci = pd.read_csv(ROOT / pp["processed_dir"] / "chunks_index.csv")
    ci = ci[ci["species"].isin(targets)].copy()

    # ablate 種に grade を付与
    gmap = grade_map(raw_dir / args.ablate)
    ci["grade"] = ci.apply(
        lambda r: gmap.get(str(r["source_file"])) if r["species"] == args.ablate else "AB",
        axis=1,
    )
    rng = random.Random(seed)

    # 録音キー = source_file（種内ユニーク）。録音単位で分割。
    val_keys, test_keys = set(), set()
    train_pool: dict[str, list[tuple]] = {a: [] for a in args.arms}  # arm -> [(species, rec)]

    for sp, sub in ci.groupby("species"):
        recs = sub.drop_duplicates("source_file")[["source_file", "grade"]].values.tolist()
        if sp == args.ablate:
            A = sorted([r for r, g in recs if g == "A"])
            B = sorted([r for r, g in recs if g == "B"])
            C = sorted([r for r, g in recs if g == "C"])
            rng.shuffle(A)
            n_val = max(1, int(len(A) * vr))
            n_test = max(1, int(len(A) * tr))
            val_keys |= {(sp, r) for r in A[:n_val]}
            test_keys |= {(sp, r) for r in A[n_val:n_val + n_test]}
            trainA = A[n_val + n_test:]
            pools = {"A": trainA, "AB": trainA + B, "ABC": trainA + B + C}
            for a in args.arms:
                train_pool[a] += [(sp, r) for r in pools[a]]
        else:
            ids = sorted([r for r, _ in recs])
            rng.shuffle(ids)
            n = len(ids)
            n_val = max(1, int(n * vr))
            n_test = max(1, int(n * tr))
            val_keys |= {(sp, r) for r in ids[:n_val]}
            test_keys |= {(sp, r) for r in ids[n_val:n_val + n_test]}
            tr_ids = ids[n_val + n_test:]
            for a in args.arms:
                train_pool[a] += [(sp, r) for r in tr_ids]

    label_map = pd.DataFrame(
        {"species": targets, "label_id": list(range(len(targets)))}
    )
    key = lambda df: set(zip(df["species"], df["source_file"]))
    ci_key = ci.assign(_k=list(zip(ci["species"], ci["source_file"])))
    val_df = ci[ci_key["_k"].isin(val_keys)]
    test_df = ci[ci_key["_k"].isin(test_keys)]

    cols = ["species", "xc_id", "chunk_index", "file_path", "duration_sec", "source_file"]
    for a in args.arms:
        out = ROOT / f"{args.out_prefix}-{a}"
        out.mkdir(parents=True, exist_ok=True)
        train_df = ci[ci_key["_k"].isin(set(train_pool[a]))]
        train_df[cols].to_csv(out / "train.csv", index=False)
        val_df[cols].to_csv(out / "val.csv", index=False)
        test_df[cols].to_csv(out / "test.csv", index=False)
        label_map.to_csv(out / "label_map.csv", index=False)
        nrec = lambda d: d.drop_duplicates("source_file").groupby("species").size().to_dict()
        print(f"[{a}] {out.name}: train録音={train_df.drop_duplicates('source_file').shape[0]} "
              f"val={val_df.drop_duplicates('source_file').shape[0]} test={test_df.drop_duplicates('source_file').shape[0]}")
        print(f"     train種別録音: {nrec(train_df)}")


if __name__ == "__main__":
    main()
