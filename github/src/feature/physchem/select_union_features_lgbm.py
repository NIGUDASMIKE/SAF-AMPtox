from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_table, report_path, write_table
from config import FEATURE_OUTPUT_ROOT, LGBM_PARAMS, TOPK_CANDIDATES


METADATA_COLUMNS = {
    "fasta_id",
    "sequence",
    "length",
    "label",
    "label_id",
    "task_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-grained LightGBM feature selection on union matrices.")
    parser.add_argument("--tasks", nargs="+", default=["amp", "tox"], choices=["amp", "tox"])
    parser.add_argument("--union-subdir", default="union")
    parser.add_argument("--topk-candidates", nargs="+", type=int, default=list(TOPK_CANDIDATES))
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--report-prefix", default="lightgbm_union_selection")
    parser.add_argument("--compact-roc-tolerance", type=float, default=0.0)
    parser.add_argument("--compact-pr-tolerance", type=float, default=0.0)
    return parser.parse_args()


def union_path(task: str, split: str, union_subdir: str) -> Path:
    return FEATURE_OUTPUT_ROOT / task / union_subdir / f"{union_subdir}_{task}_{split}.parquet"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def build_model(random_state: int) -> LGBMClassifier:
    params = dict(LGBM_PARAMS)
    params["random_state"] = random_state
    params["verbosity"] = -1
    return LGBMClassifier(**params)


def evaluate_split(model: LGBMClassifier, x: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    y_prob = model.predict_proba(x)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }


def load_union(task: str, split: str, union_subdir: str) -> pd.DataFrame:
    return read_table(union_path(task, split, union_subdir))


def save_task_golden_tables(task: str, splits: dict[str, pd.DataFrame], selected_features: list[str]) -> dict[str, str]:
    out_dir = FEATURE_OUTPUT_ROOT / task / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    keep_columns = [column for column in splits["train"].columns if column in METADATA_COLUMNS] + selected_features
    for split, frame in splits.items():
        out_path = out_dir / f"golden_{task}_{split}.parquet"
        write_table(frame.loc[:, keep_columns], out_path)
        saved[split] = str(out_path)
    return saved


def save_shared_union_tables(task: str, splits: dict[str, pd.DataFrame], shared_features: list[str]) -> dict[str, str]:
    out_dir = FEATURE_OUTPUT_ROOT / task / "golden_shared_union"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    keep_columns = [column for column in splits["train"].columns if column in METADATA_COLUMNS] + shared_features
    for split, frame in splits.items():
        out_path = out_dir / f"golden_shared_union_{task}_{split}.parquet"
        write_table(frame.loc[:, keep_columns], out_path)
        saved[split] = str(out_path)
    return saved


def main() -> None:
    args = parse_args()
    all_importance_rows: list[dict[str, float | int | str]] = []
    all_sweep_rows: list[dict[str, float | int | str]] = []
    task_manifests: dict[str, dict[str, object]] = {}
    selected_feature_union: set[str] = set()
    cached_splits: dict[str, dict[str, pd.DataFrame]] = {}

    for task in args.tasks:
        splits = {
            split: load_union(task, split, args.union_subdir)
            for split in ("train", "val", "test")
        }
        cached_splits[task] = splits
        columns = feature_columns(splits["train"])
        x_train = splits["train"].loc[:, columns].astype(np.float32)
        y_train = splits["train"]["label_id"].to_numpy(dtype=np.int64)
        x_val = splits["val"].loc[:, columns].astype(np.float32)
        y_val = splits["val"]["label_id"].to_numpy(dtype=np.int64)
        x_test = splits["test"].loc[:, columns].astype(np.float32)
        y_test = splits["test"]["label_id"].to_numpy(dtype=np.int64)

        base_model = build_model(args.random_state)
        base_model.fit(x_train, y_train)
        booster = base_model.booster_
        importance_values = booster.feature_importance(importance_type="gain")
        importance_names = booster.feature_name()

        importance_df = pd.DataFrame({
            "task": task,
            "feature_name": importance_names,
            "importance_gain": importance_values,
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1)
        importance_path = report_path(f"{args.report_prefix}_{task}_importance.csv")
        importance_df.to_csv(importance_path, index=False)

        ranked_features = importance_df["feature_name"].tolist()
        task_best = None
        task_sweep_rows: list[dict[str, float | int | str]] = []
        for topk in sorted(set(args.topk_candidates)):
            k = min(topk, len(ranked_features))
            selected = ranked_features[:k]
            model = build_model(args.random_state)
            model.fit(x_train.loc[:, selected], y_train)
            val_metrics = evaluate_split(model, x_val.loc[:, selected], y_val)
            test_metrics = evaluate_split(model, x_test.loc[:, selected], y_test)
            row = {
                "task": task,
                "topk": k,
                "n_available_features": len(ranked_features),
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1": test_metrics["f1"],
            }
            task_sweep_rows.append(row)
            if task_best is None or (row["val_roc_auc"], row["val_pr_auc"]) > (task_best["val_roc_auc"], task_best["val_pr_auc"]):
                task_best = row

        sweep_df = pd.DataFrame(task_sweep_rows).sort_values("topk")
        sweep_path = report_path(f"{args.report_prefix}_{task}_topk_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)

        assert task_best is not None
        if args.compact_roc_tolerance > 0.0 or args.compact_pr_tolerance > 0.0:
            eligible = [
                row for row in task_sweep_rows
                if row["val_roc_auc"] >= task_best["val_roc_auc"] - args.compact_roc_tolerance
                and row["val_pr_auc"] >= task_best["val_pr_auc"] - args.compact_pr_tolerance
            ]
            task_best = sorted(eligible, key=lambda row: (row["topk"], -row["val_roc_auc"], -row["val_pr_auc"]))[0]
        best_k = int(task_best["topk"])
        task_selected_features = ranked_features[:best_k]
        selected_feature_union.update(task_selected_features)

        saved_tables = save_task_golden_tables(task, splits, task_selected_features)

        all_importance_rows.extend(importance_df.to_dict("records"))
        all_sweep_rows.extend(task_sweep_rows)
        task_manifests[task] = {
            "importance_csv": str(importance_path),
            "topk_sweep_csv": str(sweep_path),
            "best_topk": best_k,
            "compact_roc_tolerance": args.compact_roc_tolerance,
            "compact_pr_tolerance": args.compact_pr_tolerance,
            "best_val_metrics": {
                "roc_auc": float(task_best["val_roc_auc"]),
                "pr_auc": float(task_best["val_pr_auc"]),
                "balanced_accuracy": float(task_best["val_balanced_accuracy"]),
                "f1": float(task_best["val_f1"]),
            },
            "best_test_metrics": {
                "roc_auc": float(task_best["test_roc_auc"]),
                "pr_auc": float(task_best["test_pr_auc"]),
                "balanced_accuracy": float(task_best["test_balanced_accuracy"]),
                "f1": float(task_best["test_f1"]),
            },
            "selected_features": task_selected_features,
            "golden_tables": saved_tables,
        }
        print(f"[OK] {task} best top-k={best_k} val ROC-AUC={task_best['val_roc_auc']:.4f} test ROC-AUC={task_best['test_roc_auc']:.4f}")

    union_feature_list = sorted(selected_feature_union)
    union_feature_path = report_path(f"{args.report_prefix}_feature_union.json")
    union_feature_path.write_text(json.dumps(union_feature_list, indent=2), encoding="utf-8")

    shared_union_metrics: dict[str, object] = {}
    for task in args.tasks:
        splits = cached_splits[task]
        x_train = splits["train"].loc[:, union_feature_list].astype(np.float32)
        y_train = splits["train"]["label_id"].to_numpy(dtype=np.int64)
        x_val = splits["val"].loc[:, union_feature_list].astype(np.float32)
        y_val = splits["val"]["label_id"].to_numpy(dtype=np.int64)
        x_test = splits["test"].loc[:, union_feature_list].astype(np.float32)
        y_test = splits["test"]["label_id"].to_numpy(dtype=np.int64)
        model = build_model(args.random_state)
        model.fit(x_train, y_train)
        val_metrics = evaluate_split(model, x_val, y_val)
        test_metrics = evaluate_split(model, x_test, y_test)
        shared_paths = save_shared_union_tables(task, splits, union_feature_list)
        shared_union_metrics[task] = {
            "n_shared_features": len(union_feature_list),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "shared_union_tables": shared_paths,
        }
        print(f"[OK] {task} shared-union({len(union_feature_list)}) val ROC-AUC={val_metrics['roc_auc']:.4f} test ROC-AUC={test_metrics['roc_auc']:.4f}")

    combined_manifest = {
        "union_subdir": args.union_subdir,
        "topk_candidates": list(args.topk_candidates),
        "random_state": args.random_state,
        "tasks": task_manifests,
        "selected_union_feature_count": len(union_feature_list),
        "selected_union_features_json": str(union_feature_path),
        "shared_union_metrics": shared_union_metrics,
    }
    manifest_path = report_path(f"{args.report_prefix}_manifest.json")
    manifest_path.write_text(json.dumps(combined_manifest, indent=2), encoding="utf-8")

    all_importance_df = pd.DataFrame(all_importance_rows)
    all_importance_df.to_csv(report_path(f"{args.report_prefix}_all_importance.csv"), index=False)
    all_sweep_df = pd.DataFrame(all_sweep_rows)
    all_sweep_df.to_csv(report_path(f"{args.report_prefix}_all_topk_sweep.csv"), index=False)

    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
