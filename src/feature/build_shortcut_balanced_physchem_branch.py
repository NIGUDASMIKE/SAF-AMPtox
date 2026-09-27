from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data--final" / "feature_physchem"
OUT_ROOT = REPO_ROOT / "data--final" / "feature_physchem_shortcut_balanced"
SOURCE_SUBDIR = "union_final_groups"
OUT_SUBDIR = "shortcut_balanced_union"
META_COLS = {"fasta_id", "sequence", "length", "label", "label_id", "task_label", "task", "split", "source", "database"}

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "objective": "binary",
    "n_jobs": -1,
    "verbosity": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shortcut-balanced CCD feature-selection branch without touching the final mainline matrices."
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--source-subdir", default=SOURCE_SUBDIR)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--out-subdir", default=OUT_SUBDIR)
    parser.add_argument("--top-k-per-task", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 29, 47, 101])
    parser.add_argument("--sampling-seed", type=int, default=20260614)
    parser.add_argument("--tox-bins", type=int, default=20)
    return parser.parse_args()


def matrix_path(root: Path, task: str, subdir: str, split: str) -> Path:
    return root / task / subdir / f"{subdir}_{task}_{split}.parquet"


def hard_split(task: str) -> str:
    return "test_hard_amp" if task == "amp" else "test_hard_tox"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLS]


def sample_n(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return frame.iloc[0:0].copy()
    if frame.shape[0] <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed)


def amp_balanced_ids(frame: pd.DataFrame, seed: int) -> tuple[list[str], dict[str, object]]:
    work = frame[["fasta_id", "sequence", "label_id"]].copy()
    work["_starts_m"] = work["sequence"].astype(str).str.startswith("M")
    selected: list[pd.DataFrame] = []
    strata: list[dict[str, object]] = []
    for starts_m, stratum in work.groupby("_starts_m"):
        pos = stratum[stratum["label_id"] == 1]
        neg = stratum[stratum["label_id"] == 0]
        n = min(pos.shape[0], neg.shape[0])
        selected.append(sample_n(pos, n, seed + int(starts_m) + 11))
        selected.append(sample_n(neg, n, seed + int(starts_m) + 29))
        strata.append(
            {
                "starts_with_M": bool(starts_m),
                "positive_available": int(pos.shape[0]),
                "negative_available": int(neg.shape[0]),
                "selected_per_class": int(n),
            }
        )
    out = pd.concat(selected, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    report = {
        "n_total": int(out.shape[0]),
        "n_positive": int((out["label_id"] == 1).sum()),
        "n_negative": int((out["label_id"] == 0).sum()),
        "positive_m_start": float(out.loc[out["label_id"] == 1, "_starts_m"].mean()),
        "negative_m_start": float(out.loc[out["label_id"] == 0, "_starts_m"].mean()),
        "strata": strata,
    }
    return out["fasta_id"].astype(str).tolist(), report


def tox_balanced_ids(frame: pd.DataFrame, bins: int, seed: int) -> tuple[list[str], dict[str, object]]:
    work = frame[["fasta_id", "sequence", "label_id"]].copy()
    c_count = work["sequence"].astype(str).str.count("C")
    work["_c_count"] = c_count
    work["_c_freq"] = c_count / work["sequence"].astype(str).str.len().clip(lower=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    work["_c_bin"] = -1
    nonzero = work["_c_freq"] > 0
    work.loc[nonzero, "_c_bin"] = pd.cut(
        work.loc[nonzero, "_c_freq"],
        bins=edges,
        include_lowest=False,
        labels=False,
    ).astype("Int64")
    selected: list[pd.DataFrame] = []
    bin_report: list[dict[str, object]] = []
    for bin_id, stratum in work.groupby("_c_bin", dropna=False):
        pos = stratum[stratum["label_id"] == 1]
        neg = stratum[stratum["label_id"] == 0]
        n = min(pos.shape[0], neg.shape[0])
        if n > 0:
            selected.append(sample_n(pos, n, seed + int(bin_id) + 101))
            selected.append(sample_n(neg, n, seed + int(bin_id) + 409))
        bin_report.append(
            {
                "bin": int(bin_id) if pd.notna(bin_id) else -1,
                "positive_available": int(pos.shape[0]),
                "negative_available": int(neg.shape[0]),
                "selected_per_class": int(n),
            }
        )
    out = pd.concat(selected, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = out[out["label_id"] == 1]
    neg = out[out["label_id"] == 0]
    report = {
        "bins": int(bins),
        "n_total": int(out.shape[0]),
        "n_positive": int(pos.shape[0]),
        "n_negative": int(neg.shape[0]),
        "positive_contains_C": float((pos["_c_count"] > 0).mean()),
        "negative_contains_C": float((neg["_c_count"] > 0).mean()),
        "positive_mean_C_frequency": float(pos["_c_freq"].mean()),
        "negative_mean_C_frequency": float(neg["_c_freq"].mean()),
        "bin_report": bin_report,
    }
    return out["fasta_id"].astype(str).tolist(), report


def balanced_ids(task: str, frame: pd.DataFrame, split: str, args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    seed = args.sampling_seed + (0 if split == "train" else 100_000)
    if task == "amp":
        return amp_balanced_ids(frame, seed)
    return tox_balanced_ids(frame, bins=args.tox_bins, seed=seed)


def mean_importance(frame: pd.DataFrame, features: list[str], seeds: list[int]) -> pd.Series:
    x = frame.loc[:, features].astype(np.float32)
    y = frame["label_id"].to_numpy(dtype=np.int64)
    gains = []
    for seed in seeds:
        params = dict(LGBM_PARAMS)
        params["random_state"] = seed
        model = LGBMClassifier(**params)
        model.fit(x, y)
        gains.append(pd.Series(model.booster_.feature_importance(importance_type="gain"), index=features))
    return pd.concat(gains, axis=1).mean(axis=1)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    task_selected: dict[str, list[str]] = {}
    task_reports: dict[str, object] = {}
    all_feature_order: list[str] | None = None

    for task in ["amp", "tox"]:
        train = pd.read_parquet(matrix_path(source_root, task, args.source_subdir, "train"))
        val = pd.read_parquet(matrix_path(source_root, task, args.source_subdir, "val"))
        features = feature_columns(train)
        if all_feature_order is None:
            all_feature_order = features
        train_ids, train_report = balanced_ids(task, train, "train", args)
        val_ids, val_report = balanced_ids(task, val, "val", args)

        train_balanced = train[train["fasta_id"].astype(str).isin(set(train_ids))].copy()
        importance = mean_importance(train_balanced, features, args.seeds)
        selected = importance.sort_values(ascending=False).head(args.top_k_per_task).index.tolist()
        task_selected[task] = selected
        task_reports[task] = {
            "train_balancing": train_report,
            "val_balancing": val_report,
            "top_k_per_task": args.top_k_per_task,
            "selected_features": selected,
            "top20_importance": importance.sort_values(ascending=False).head(20).to_dict(),
        }

    assert all_feature_order is not None
    union_set = set(task_selected["amp"]).union(task_selected["tox"])
    union_features = [feature for feature in all_feature_order if feature in union_set]
    (report_dir / "shortcut_balanced_compact_feature_union.json").write_text(
        json.dumps(union_features, indent=2),
        encoding="utf-8",
    )
    (report_dir / "shortcut_balanced_task_feature_selection.json").write_text(
        json.dumps(task_reports, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"task": task, "feature": feature, "selected": True} for task, feats in task_selected.items() for feature in feats]
    ).to_csv(report_dir / "shortcut_balanced_task_selected_features.csv", index=False)

    split_map = {
        "train": "train",
        "val": "val",
        "test": "test",
        "test_hard_amp": "test_hard_amp",
        "test_hard_tox": "test_hard_tox",
    }
    branch_report: dict[str, object] = {
        "source_root": str(source_root),
        "source_subdir": args.source_subdir,
        "out_root": str(out_root),
        "out_subdir": args.out_subdir,
        "seeds": args.seeds,
        "sampling_seed": args.sampling_seed,
        "tox_bins": args.tox_bins,
        "n_union_features": len(union_features),
        "task_reports": task_reports,
    }

    for task in ["amp", "tox"]:
        task_out = out_root / task / args.out_subdir
        task_out.mkdir(parents=True, exist_ok=True)
        train_full = pd.read_parquet(matrix_path(source_root, task, args.source_subdir, "train"))
        val_full = pd.read_parquet(matrix_path(source_root, task, args.source_subdir, "val"))
        train_ids, _ = balanced_ids(task, train_full, "train", args)
        val_ids, _ = balanced_ids(task, val_full, "val", args)
        id_map = {"train": set(train_ids), "val": set(val_ids)}
        for split in ["train", "val", "test", hard_split(task)]:
            frame = pd.read_parquet(matrix_path(source_root, task, args.source_subdir, split))
            if split in id_map:
                frame = frame[frame["fasta_id"].astype(str).isin(id_map[split])].copy()
            meta_cols = [column for column in frame.columns if column in META_COLS]
            out = frame.loc[:, [*meta_cols, *union_features]].copy()
            out_path = task_out / f"{args.out_subdir}_{task}_{split}.parquet"
            out.to_parquet(out_path, index=False)
            print(f"[WRITE] {task}/{split}: {out.shape} -> {out_path}")

    (report_dir / "shortcut_balanced_branch_manifest.json").write_text(
        json.dumps(branch_report, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] union features: {len(union_features)}")
    print(f"[DONE] reports -> {report_dir}")


if __name__ == "__main__":
    main()
