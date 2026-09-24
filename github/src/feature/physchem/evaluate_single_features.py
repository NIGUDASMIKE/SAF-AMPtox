from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import feature_columns, feature_path, read_table, report_path
from config import DEFAULT_FEATURE_GROUPS, DEFAULT_MODEL, DEFAULT_TASKS, LGBM_PARAMS, RANDOM_SEEDS, RF_PARAMS
from feature_registry import ready_feature_groups, resolve_feature_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate feature groups on validation splits.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=list(DEFAULT_TASKS))
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS))
    parser.add_argument("--all-ready", action="store_true")
    parser.add_argument("--model", choices=("lightgbm", "random_forest"), default=DEFAULT_MODEL)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RANDOM_SEEDS))
    parser.add_argument("--report-prefix", default="single_feature_results")
    return parser.parse_args()


def load_feature_split(task: str, group: str, split: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    df = read_table(feature_path(task, group, split))
    columns = feature_columns(df)
    x = df.loc[:, columns].astype(np.float32)
    y = df["label_id"].to_numpy(dtype=np.int64)
    return x, y, len(columns)


def build_model(model_name: str, seed: int):
    if model_name == "lightgbm":
        params = dict(LGBM_PARAMS)
        params["random_state"] = seed
        params["verbosity"] = -1
        return LGBMClassifier(**params)
    params = dict(RF_PARAMS)
    params["random_state"] = seed
    return RandomForestClassifier(**params)


def evaluate_once(model_name: str, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_eval: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    model = build_model(model_name, seed)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_eval)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_eval, y_prob)),
        "pr_auc": float(average_precision_score(y_eval, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "f1": float(f1_score(y_eval, y_pred)),
    }


def summarize_metric(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"mean_{prefix}": float(array.mean()),
        f"std_{prefix}": float(array.std(ddof=0)),
        f"min_{prefix}": float(array.min()),
        f"max_{prefix}": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    groups = ready_feature_groups() if args.all_ready else resolve_feature_groups(args.groups)
    summary_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[dict[str, float | int | str]] = []

    for task in args.tasks:
        for group in groups:
            x_train, y_train, n_features = load_feature_split(task, group, args.train_split)
            x_eval, y_eval, _ = load_feature_split(task, group, args.eval_split)

            metrics_by_seed = []
            for seed in args.seeds:
                metrics = evaluate_once(args.model, seed, x_train, y_train, x_eval, y_eval)
                metrics_by_seed.append(metrics)
                detail_rows.append({
                    "task": task,
                    "feature_group": group,
                    "model_type": args.model,
                    "seed": seed,
                    "n_features": n_features,
                    **metrics,
                })

            row: dict[str, float | int | str] = {
                "task": task,
                "feature_group": group,
                "model_type": args.model,
                "n_features": n_features,
                "train_n": int(len(y_train)),
                "eval_n": int(len(y_eval)),
            }
            for metric_name in ("roc_auc", "pr_auc", "balanced_accuracy", "f1"):
                row.update(summarize_metric([metrics[metric_name] for metrics in metrics_by_seed], metric_name))
            summary_rows.append(row)
            print(f"[OK] {task}/{group} mean ROC-AUC={row['mean_roc_auc']:.4f} PR-AUC={row['mean_pr_auc']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["task", "mean_roc_auc"], ascending=[True, False])
    detail_df = pd.DataFrame(detail_rows).sort_values(["task", "feature_group", "seed"])

    summary_path = report_path(f"{args.report_prefix}_summary.csv")
    detail_path = report_path(f"{args.report_prefix}_detail.csv")
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    manifest = {
        "model_type": args.model,
        "tasks": args.tasks,
        "groups": groups,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "seeds": list(args.seeds),
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
    }
    manifest_path = report_path(f"{args.report_prefix}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[DONE] summary -> {summary_path}")
    print(f"[DONE] detail -> {detail_path}")


if __name__ == "__main__":
    main()
