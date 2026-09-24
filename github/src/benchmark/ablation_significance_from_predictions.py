from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_ORDER = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]
MODEL_LABELS = {
    "ccd_mlp": "CCD-only",
    "esm_mlp": "ESM-2-only",
    "concat_mlp": "Direct Concat",
    "cross_attention_residual": "Residual CA",
}
METRICS = ["sensitivity", "specificity", "accuracy", "f1", "mcc", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "sensitivity": "SEN",
    "specificity": "SPE",
    "accuracy": "Acc",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast paired significance tests from cached hard-test predictions.")
    parser.add_argument(
        "--prediction-dir",
        default=str(REPO_ROOT / "data--final" / "statistical_tests" / "selected_residual_ablation" / "predictions"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data--final" / "statistical_tests" / "selected_residual_ablation"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def metric_value(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, metric: str) -> float:
    y = y.astype(np.int64)
    pred = pred.astype(np.int64)
    if metric == "roc_auc":
        return float(roc_auc_score(y, prob))
    if metric == "pr_auc":
        return float(average_precision_score(y, prob))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    if metric == "sensitivity":
        return float(tp / (tp + fn)) if (tp + fn) else 0.0
    if metric == "specificity":
        return float(tn / (tn + fp)) if (tn + fp) else 0.0
    raise ValueError(metric)


def stratified_bootstrap_indices(y: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return [
        np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
        for _ in range(n)
    ]


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_metric_values(
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    metric: str,
    samples: list[np.ndarray],
) -> np.ndarray:
    return np.asarray([metric_value(y[idx], prob[idx], pred[idx], metric) for idx in samples], dtype=np.float64)


def bootstrap_one_sided_p(delta_values: np.ndarray, observed_delta: float) -> float:
    if observed_delta <= 0:
        return 1.0
    return float((np.sum(delta_values <= 0.0) + 1) / (delta_values.size + 1))


def mcnemar_accuracy_p(y: np.ndarray, residual_pred: np.ndarray, comparator_pred: np.ndarray) -> dict[str, float | int]:
    residual_correct = residual_pred == y
    comparator_correct = comparator_pred == y
    residual_only = int(np.sum(residual_correct & ~comparator_correct))
    comparator_only = int(np.sum(~residual_correct & comparator_correct))
    discordant = residual_only + comparator_only
    p = 1.0 if discordant == 0 else float(binomtest(residual_only, discordant, 0.5, alternative="greater").pvalue)
    return {
        "residual_correct_comparator_wrong": residual_only,
        "residual_wrong_comparator_correct": comparator_only,
        "mcnemar_accuracy_p_one_sided": p,
    }


def load_task_predictions(prediction_dir: Path, task: str) -> pd.DataFrame:
    path = prediction_dir / f"{task}_hard_ensemble_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run_task(task: str, df: pd.DataFrame, samples: list[np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["label_id"].to_numpy(dtype=np.int64)
    metric_rows = []
    ci_rows = []
    boot_cache: dict[tuple[str, str], np.ndarray] = {}

    for model in MODEL_ORDER:
        prob = df[f"{model}_prob"].to_numpy(dtype=np.float64)
        pred = df[f"{model}_pred"].to_numpy(dtype=np.int64)
        metric_row = {"task": task, "model": model, "model_label": MODEL_LABELS[model], "n": int(len(y))}
        for metric in METRICS:
            value = metric_value(y, prob, pred, metric)
            boot = bootstrap_metric_values(y, prob, pred, metric, samples)
            boot_cache[(model, metric)] = boot
            low, high = bootstrap_ci(boot)
            metric_row[metric] = value
            ci_rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "value": value,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
        metric_rows.append(metric_row)

    pair_rows = []
    residual_prob = df["cross_attention_residual_prob"].to_numpy(dtype=np.float64)
    residual_pred = df["cross_attention_residual_pred"].to_numpy(dtype=np.int64)
    for comparator in ["concat_mlp", "esm_mlp", "ccd_mlp"]:
        comp_prob = df[f"{comparator}_prob"].to_numpy(dtype=np.float64)
        comp_pred = df[f"{comparator}_pred"].to_numpy(dtype=np.int64)
        for metric in METRICS:
            residual_value = metric_value(y, residual_prob, residual_pred, metric)
            comparator_value = metric_value(y, comp_prob, comp_pred, metric)
            observed_delta = residual_value - comparator_value
            delta_boot = boot_cache[("cross_attention_residual", metric)] - boot_cache[(comparator, metric)]
            low, high = bootstrap_ci(delta_boot)
            row = {
                "task": task,
                "comparison": f"Residual CA vs {MODEL_LABELS[comparator]}",
                "comparator": comparator,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "residual_value": residual_value,
                "comparator_value": comparator_value,
                "delta_residual_minus_comparator": observed_delta,
                "bootstrap_delta_ci_low": low,
                "bootstrap_delta_ci_high": high,
                "paired_bootstrap_p_one_sided_residual_gt_comparator": bootstrap_one_sided_p(delta_boot, observed_delta),
            }
            if metric == "accuracy":
                row.update(mcnemar_accuracy_p(y, residual_pred, comp_pred))
            pair_rows.append(row)

    for metric in METRICS:
        best_model = max(
            [m for m in MODEL_ORDER if m != "cross_attention_residual"],
            key=lambda m: metric_value(
                y,
                df[f"{m}_prob"].to_numpy(dtype=np.float64),
                df[f"{m}_pred"].to_numpy(dtype=np.int64),
                metric,
            ),
        )
        residual_value = metric_value(y, residual_prob, residual_pred, metric)
        comp_prob = df[f"{best_model}_prob"].to_numpy(dtype=np.float64)
        comp_pred = df[f"{best_model}_pred"].to_numpy(dtype=np.int64)
        comparator_value = metric_value(y, comp_prob, comp_pred, metric)
        observed_delta = residual_value - comparator_value
        delta_boot = boot_cache[("cross_attention_residual", metric)] - boot_cache[(best_model, metric)]
        low, high = bootstrap_ci(delta_boot)
        row = {
            "task": task,
            "comparison": f"Residual CA vs best non-Residual ({MODEL_LABELS[best_model]})",
            "comparator": best_model,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "residual_value": residual_value,
            "comparator_value": comparator_value,
            "delta_residual_minus_comparator": observed_delta,
            "bootstrap_delta_ci_low": low,
            "bootstrap_delta_ci_high": high,
            "paired_bootstrap_p_one_sided_residual_gt_comparator": bootstrap_one_sided_p(delta_boot, observed_delta),
        }
        if metric == "accuracy":
            row.update(mcnemar_accuracy_p(y, residual_pred, comp_pred))
        pair_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(ci_rows), pd.DataFrame(pair_rows)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_report(out_dir: Path, metrics: pd.DataFrame, ci: pd.DataFrame, pairs: pd.DataFrame, n_bootstrap: int) -> None:
    lines = [
        "# Hard-test ablation significance report",
        "",
        f"Bootstrap replicates: {n_bootstrap}. P-values are one-sided paired stratified bootstrap tests for Residual CA > comparator.",
        "For accuracy, an additional one-sided exact McNemar/binomial p-value is reported in the CSV.",
        "",
    ]
    for task in ["amp", "tox"]:
        lines.append(f"## {task.upper()} hard-test values with 95% bootstrap CI")
        rows = []
        for model in MODEL_ORDER:
            row = {"Model": MODEL_LABELS[model]}
            for metric in METRICS:
                x = ci[(ci["task"] == task) & (ci["model"] == model) & (ci["metric"] == metric)].iloc[0]
                row[METRIC_LABELS[metric]] = (
                    f"{float(x['value']):.4f} "
                    f"[{float(x['bootstrap_ci_low']):.4f}, {float(x['bootstrap_ci_high']):.4f}]"
                )
            rows.append(row)
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")
        lines.append(f"## {task.upper()} Residual CA pairwise evidence")
        p_rows = []
        selected = pairs[(pairs["task"] == task) & ~pairs["comparison"].str.contains("best non-Residual", regex=False)]
        for comparator in ["concat_mlp", "esm_mlp", "ccd_mlp"]:
            row = {"Comparison": f"Residual CA vs {MODEL_LABELS[comparator]}"}
            for metric in METRICS:
                x = selected[(selected["comparator"] == comparator) & (selected["metric"] == metric)].iloc[0]
                row[METRIC_LABELS[metric]] = (
                    f"Delta={float(x['delta_residual_minus_comparator']):.4f}; "
                    f"p={fmt_p(float(x['paired_bootstrap_p_one_sided_residual_gt_comparator']))}"
                )
            p_rows.append(row)
        lines.append(pd.DataFrame(p_rows).to_markdown(index=False))
        lines.append("")
        best = pairs[(pairs["task"] == task) & pairs["comparison"].str.contains("best non-Residual", regex=False)]
        best_rows = []
        for metric in METRICS:
            x = best[best["metric"] == metric].iloc[0]
            best_rows.append(
                {
                    "Metric": METRIC_LABELS[metric],
                    "Best non-Residual": MODEL_LABELS[str(x["comparator"])],
                    "Residual": f"{float(x['residual_value']):.4f}",
                    "Best non-Residual value": f"{float(x['comparator_value']):.4f}",
                    "Delta": f"{float(x['delta_residual_minus_comparator']):.4f}",
                    "p": fmt_p(float(x["paired_bootstrap_p_one_sided_residual_gt_comparator"])),
                    "Delta 95% CI": (
                        f"[{float(x['bootstrap_delta_ci_low']):.4f}, "
                        f"{float(x['bootstrap_delta_ci_high']):.4f}]"
                    ),
                }
            )
        lines.append(f"## {task.upper()} Residual CA vs metric-wise best non-Residual")
        lines.append(pd.DataFrame(best_rows).to_markdown(index=False))
        lines.append("")
    (out_dir / "hard_test_ablation_statistical_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    metric_frames = []
    ci_frames = []
    pair_frames = []
    for task in ["amp", "tox"]:
        df = load_task_predictions(prediction_dir, task)
        samples = stratified_bootstrap_indices(df["label_id"].to_numpy(dtype=np.int64), args.n_bootstrap, rng)
        metric_df, ci_df, pair_df = run_task(task, df, samples)
        metric_frames.append(metric_df)
        ci_frames.append(ci_df)
        pair_frames.append(pair_df)

    metrics = pd.concat(metric_frames, ignore_index=True)
    ci = pd.concat(ci_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    metrics.to_csv(out_dir / "hard_test_ablation_metrics_recomputed.csv", index=False)
    ci.to_csv(out_dir / "hard_test_ablation_metric_bootstrap_ci.csv", index=False)
    pairs.to_csv(out_dir / "hard_test_ablation_pairwise_significance.csv", index=False)
    write_report(out_dir, metrics, ci, pairs, args.n_bootstrap)
    manifest = {
        "prediction_dir": str(prediction_dir.resolve()),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "p_value_definition": "one-sided paired stratified bootstrap p-value for Residual CA > comparator",
        "accuracy_extra_test": "one-sided exact McNemar/binomial test, stored in pairwise CSV",
    }
    (out_dir / "hard_test_ablation_significance_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] report -> {out_dir / 'hard_test_ablation_statistical_report.md'}")
    print(f"[DONE] pairwise CSV -> {out_dir / 'hard_test_ablation_pairwise_significance.csv'}")


if __name__ == "__main__":
    main()
