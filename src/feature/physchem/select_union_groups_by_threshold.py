from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select feature groups by absolute thresholds instead of fixed top-K.")
    parser.add_argument("--summary-csv", default=str(report_path("single_feature_results_summary.csv")))
    parser.add_argument("--amp-roc-threshold", type=float, default=None)
    parser.add_argument("--tox-roc-threshold", type=float, default=None)
    parser.add_argument("--amp-pr-threshold", type=float, default=None)
    parser.add_argument("--tox-pr-threshold", type=float, default=None)
    parser.add_argument("--max-roc-std", type=float, default=0.03)
    parser.add_argument("--auto-quantile", type=float, default=None)
    parser.add_argument("--min-roc", type=float, default=0.80)
    parser.add_argument("--min-pr", type=float, default=0.80)
    parser.add_argument("--report-prefix", default="selected_groups_union")
    parser.add_argument("--suggest-only", action="store_true")
    return parser.parse_args()


def describe_distribution(df: pd.DataFrame, metric: str) -> dict[str, float]:
    series = df[metric].astype(float)
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def select_task_groups(
    df: pd.DataFrame,
    roc_threshold: float | None,
    pr_threshold: float | None,
    max_roc_std: float,
) -> pd.DataFrame:
    selected = df.copy()
    if roc_threshold is not None:
        selected = selected[selected["mean_roc_auc"] >= roc_threshold]
    if pr_threshold is not None:
        selected = selected[selected["mean_pr_auc"] >= pr_threshold]
    selected = selected[selected["std_roc_auc"] <= max_roc_std]
    return selected.sort_values(["mean_roc_auc", "mean_pr_auc"], ascending=[False, False])


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)

    amp_df = summary_df[summary_df["task"] == "amp"].copy()
    tox_df = summary_df[summary_df["task"] == "tox"].copy()

    distribution_report = {
        "amp": {
            "roc_auc": describe_distribution(amp_df, "mean_roc_auc"),
            "pr_auc": describe_distribution(amp_df, "mean_pr_auc"),
        },
        "tox": {
            "roc_auc": describe_distribution(tox_df, "mean_roc_auc"),
            "pr_auc": describe_distribution(tox_df, "mean_pr_auc"),
        },
    }

    if args.auto_quantile is not None:
        args.amp_roc_threshold = max(args.min_roc, float(amp_df["mean_roc_auc"].quantile(args.auto_quantile)))
        args.tox_roc_threshold = max(args.min_roc, float(tox_df["mean_roc_auc"].quantile(args.auto_quantile)))
        args.amp_pr_threshold = max(args.min_pr, float(amp_df["mean_pr_auc"].quantile(args.auto_quantile)))
        args.tox_pr_threshold = max(args.min_pr, float(tox_df["mean_pr_auc"].quantile(args.auto_quantile)))

    distribution_path = report_path(f"{args.report_prefix}_distribution_summary.json")
    distribution_path.write_text(json.dumps(distribution_report, indent=2), encoding="utf-8")

    if args.suggest_only:
        print(json.dumps(distribution_report, indent=2))
        print(f"[DONE] wrote distribution summary -> {distribution_path}")
        return

    amp_selected = select_task_groups(
        amp_df,
        roc_threshold=args.amp_roc_threshold,
        pr_threshold=args.amp_pr_threshold,
        max_roc_std=args.max_roc_std,
    )
    tox_selected = select_task_groups(
        tox_df,
        roc_threshold=args.tox_roc_threshold,
        pr_threshold=args.tox_pr_threshold,
        max_roc_std=args.max_roc_std,
    )

    union_groups = sorted(set(amp_selected["feature_group"]).union(tox_selected["feature_group"]))
    union_df = summary_df[summary_df["feature_group"].isin(union_groups)].copy()
    union_df["selected_for_amp"] = union_df["feature_group"].isin(set(amp_selected["feature_group"]))
    union_df["selected_for_tox"] = union_df["feature_group"].isin(set(tox_selected["feature_group"]))
    union_df = union_df.sort_values(["feature_group", "task"])

    union_csv_path = report_path(f"{args.report_prefix}.csv")
    union_df.to_csv(union_csv_path, index=False)

    manifest = {
        "summary_csv": str(Path(args.summary_csv).resolve()),
        "distribution_summary_json": str(distribution_path),
        "amp_roc_threshold": args.amp_roc_threshold,
        "tox_roc_threshold": args.tox_roc_threshold,
        "amp_pr_threshold": args.amp_pr_threshold,
        "tox_pr_threshold": args.tox_pr_threshold,
        "max_roc_std": args.max_roc_std,
        "selected_amp_groups": amp_selected["feature_group"].tolist(),
        "selected_tox_groups": tox_selected["feature_group"].tolist(),
        "selected_union_groups": union_groups,
    }
    manifest_path = report_path(f"{args.report_prefix}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[DONE] union groups -> {union_csv_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
