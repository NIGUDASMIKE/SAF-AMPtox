from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_FEATURE = REPO_ROOT / "src" / "feature"
if str(SRC_FEATURE) not in sys.path:
    sys.path.insert(0, str(SRC_FEATURE))

from train_fusion_models import build_model, hard_split, load_split_arrays  # noqa: E402


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
    parser = argparse.ArgumentParser(
        description="Paired hard-test significance analysis for selected Residual CA ablations."
    )
    parser.add_argument(
        "--baseline-checkpoint-root",
        default=str(
            REPO_ROOT
            / "data--final"
            / "fusion_models_residual_sweep_v2"
            / "fair_ensemble_ablation_default"
            / "checkpoints"
        ),
    )
    parser.add_argument(
        "--selected-checkpoint-root",
        default=str(
            REPO_ROOT
            / "data--final"
            / "fusion_models_residual_sweep_v2"
            / "selected_validation_ranked_candidate"
            / "checkpoints"
        ),
    )
    parser.add_argument(
        "--baseline-ensemble-table",
        default=str(
            REPO_ROOT
            / "data--final"
            / "fusion_models_residual_sweep_v2"
            / "fair_ensemble_ablation_default"
            / "reports"
            / "fusion_model_ensemble_results.csv"
        ),
    )
    parser.add_argument(
        "--selected-ensemble-table",
        default=str(
            REPO_ROOT
            / "data--final"
            / "fusion_models_residual_sweep_v2"
            / "selected_validation_ranked_candidate"
            / "reports"
            / "selected_validation_ranked_residual_ca_metrics.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data--final" / "statistical_tests" / "selected_residual_ablation"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutation", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_paths(task: str, model: str, baseline_root: Path, selected_root: Path) -> list[Path]:
    root = selected_root if model == "cross_attention_residual" else baseline_root
    paths = sorted(root.glob(f"{task}__{model}__seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints for {task}/{model} in {root}")
    return paths


def scaler_transform(x: np.ndarray, scaler_state: dict) -> np.ndarray:
    mean = np.asarray(scaler_state["mean"], dtype=np.float32)
    std = np.asarray(scaler_state["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def make_model(checkpoint: dict, ccd_dim: int, esm_dim: int) -> torch.nn.Module:
    hyper = checkpoint["model_hyperparameters"]
    args = SimpleNamespace(
        hidden_dim=int(hyper["hidden_dim"]),
        d_model=int(hyper["d_model"]),
        heads=int(hyper["heads"]),
        esm_tokens=int(hyper["esm_tokens"]),
        dropout=float(hyper["dropout"]),
    )
    model = build_model(
        checkpoint["model_name"],
        ccd_dim=ccd_dim,
        esm_dim=esm_dim,
        group_dims=list(checkpoint["group_dims"]),
        args=args,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_arrays_for_checkpoint(task: str, checkpoint: dict):
    sources = checkpoint["feature_sources"]
    return load_split_arrays(
        task=task,
        split=hard_split(task),
        ccd_root=Path(sources["ccd_root"]),
        ccd_union_subdir=sources["ccd_union_subdir"],
        ccd_features=checkpoint["ccd_features"],
        esm_root=Path(sources["esm_root"]),
    )


def predict_checkpoint(arrays, checkpoint: dict, batch_size: int) -> np.ndarray:
    ccd = scaler_transform(arrays.ccd.astype(np.float32), checkpoint["ccd_scaler"])
    esm = scaler_transform(arrays.esm.astype(np.float32), checkpoint["esm_scaler"])
    model = make_model(checkpoint, ccd.shape[1], esm.shape[1])
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.y), batch_size):
            end = start + batch_size
            logits = model(torch.from_numpy(ccd[start:end]), torch.from_numpy(esm[start:end]))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float64)


def ensemble_threshold(task: str, model: str, split: str, baseline_table: pd.DataFrame, selected_table: pd.DataFrame) -> float:
    if model == "cross_attention_residual":
        row = selected_table[
            (selected_table["task"] == task)
            & (selected_table["split"] == split)
            & (selected_table["model"] == "selected_validation_ranked_residual_ca")
        ].iloc[0]
    else:
        row = baseline_table[
            (baseline_table["task"] == task) & (baseline_table["split"] == split) & (baseline_table["model"] == model)
        ].iloc[0]
    return float(row["threshold"])


def metric_value(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, metric: str) -> float:
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
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    raise ValueError(metric)


def all_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    out = {metric: metric_value(y, prob, pred, metric) for metric in METRICS}
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"threshold": threshold, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    return out


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_int = y.astype(int)
    pos = np.flatnonzero(y_int == 1)
    neg = np.flatnonzero(y_int == 0)
    return np.concatenate(
        [
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ]
    )


def bootstrap_metric_ci(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, metric: str, n: int, rng: np.random.Generator):
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = stratified_indices(y, rng)
        values[i] = metric_value(y[idx], prob[idx], pred[idx], metric)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_delta_ci(
    y: np.ndarray,
    prob_a: np.ndarray,
    pred_a: np.ndarray,
    prob_b: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n: int,
    rng: np.random.Generator,
):
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = stratified_indices(y, rng)
        deltas[i] = metric_value(y[idx], prob_a[idx], pred_a[idx], metric) - metric_value(
            y[idx], prob_b[idx], pred_b[idx], metric
        )
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def paired_permutation_pvalue(
    y: np.ndarray,
    prob_a: np.ndarray,
    pred_a: np.ndarray,
    prob_b: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = metric_value(y, prob_a, pred_a, metric) - metric_value(y, prob_b, pred_b, metric)
    if observed <= 0:
        return observed, 1.0
    exceed = 0
    for _ in range(n):
        swap = rng.random(y.shape[0]) < 0.5
        pa = prob_a.copy()
        pb = prob_b.copy()
        ya = pred_a.copy()
        yb = pred_b.copy()
        pa[swap], pb[swap] = pb[swap], pa[swap].copy()
        ya[swap], yb[swap] = yb[swap], ya[swap].copy()
        delta = metric_value(y, pa, ya, metric) - metric_value(y, pb, yb, metric)
        if delta >= observed - 1e-12:
            exceed += 1
    return observed, float((exceed + 1) / (n + 1))


def mcnemar_one_sided_p(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int]:
    a_correct = pred_a == y
    b_correct = pred_b == y
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    total = a_only + b_only
    if total == 0:
        p = 1.0
    else:
        p = float(binomtest(a_only, total, 0.5, alternative="greater").pvalue)
    return {"a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only, "mcnemar_accuracy_p_one_sided": p}


def load_ensemble_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    baseline_root = Path(args.baseline_checkpoint_root)
    selected_root = Path(args.selected_checkpoint_root)
    baseline_table = pd.read_csv(args.baseline_ensemble_table)
    selected_table = pd.read_csv(args.selected_ensemble_table)
    out_rows = []
    bundles: dict[tuple[str, str], dict[str, object]] = {}
    for task in ["amp", "tox"]:
        split = hard_split(task)
        for model in MODEL_ORDER:
            paths = checkpoint_paths(task, model, baseline_root, selected_root)
            checkpoints = [load_checkpoint(path) for path in paths]
            arrays = load_arrays_for_checkpoint(task, checkpoints[0])
            probs = np.vstack([predict_checkpoint(arrays, checkpoint, args.batch_size) for checkpoint in checkpoints])
            prob = probs.mean(axis=0)
            threshold = ensemble_threshold(task, model, split, baseline_table, selected_table)
            pred = (prob >= threshold).astype(np.int64)
            metrics = all_metrics(arrays.y.astype(np.int64), prob, threshold)
            bundles[(task, model)] = {
                "fasta_id": np.asarray(arrays.fasta_id),
                "y": arrays.y.astype(np.int64),
                "prob": prob,
                "pred": pred,
                "threshold": threshold,
                "checkpoint_paths": [str(path.resolve()) for path in paths],
            }
            out_rows.append(
                {
                    "task": task,
                    "split": split,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n_checkpoints": len(paths),
                    "seeds": ",".join(str(load_checkpoint(path)["seed"]) for path in paths),
                    **metrics,
                }
            )
    return pd.DataFrame(out_rows), bundles


def write_prediction_tables(out_dir: Path, bundles: dict[tuple[str, str], dict[str, object]]) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for task in ["amp", "tox"]:
        base = bundles[(task, MODEL_ORDER[0])]
        frame = pd.DataFrame({"fasta_id": base["fasta_id"], "label_id": base["y"]})
        for model in MODEL_ORDER:
            bundle = bundles[(task, model)]
            if not np.array_equal(frame["fasta_id"].astype(str).to_numpy(), bundle["fasta_id"]):
                raise RuntimeError(f"fasta_id mismatch for {task}/{model}")
            frame[f"{model}_prob"] = bundle["prob"]
            frame[f"{model}_pred"] = bundle["pred"]
        frame.to_csv(pred_dir / f"{task}_hard_ensemble_predictions.csv", index=False)


def significance_tables(args: argparse.Namespace, metric_df: pd.DataFrame, bundles: dict[tuple[str, str], dict[str, object]]):
    rng = np.random.default_rng(args.seed)
    model_ci_rows = []
    pair_rows = []
    for task in ["amp", "tox"]:
        residual = bundles[(task, "cross_attention_residual")]
        y = residual["y"]
        for model in MODEL_ORDER:
            bundle = bundles[(task, model)]
            for metric in METRICS:
                ci_low, ci_high = bootstrap_metric_ci(
                    y,
                    bundle["prob"],
                    bundle["pred"],
                    metric,
                    args.n_bootstrap,
                    rng,
                )
                value = float(metric_df[(metric_df["task"] == task) & (metric_df["model"] == model)].iloc[0][metric])
                model_ci_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "value": value,
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                    }
                )

        for comparator in ["concat_mlp", "esm_mlp", "ccd_mlp"]:
            comp = bundles[(task, comparator)]
            if not np.array_equal(residual["fasta_id"], comp["fasta_id"]):
                raise RuntimeError(f"fasta_id mismatch for {task} residual vs {comparator}")
            for metric in METRICS:
                delta, p_perm = paired_permutation_pvalue(
                    y,
                    residual["prob"],
                    residual["pred"],
                    comp["prob"],
                    comp["pred"],
                    metric,
                    args.n_permutation,
                    rng,
                )
                ci_low, ci_high = bootstrap_delta_ci(
                    y,
                    residual["prob"],
                    residual["pred"],
                    comp["prob"],
                    comp["pred"],
                    metric,
                    args.n_bootstrap,
                    rng,
                )
                row = {
                    "task": task,
                    "comparison": f"Residual CA vs {MODEL_LABELS[comparator]}",
                    "comparator": comparator,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "residual_value": metric_value(y, residual["prob"], residual["pred"], metric),
                    "comparator_value": metric_value(y, comp["prob"], comp["pred"], metric),
                    "delta_residual_minus_comparator": delta,
                    "bootstrap_delta_ci_low": ci_low,
                    "bootstrap_delta_ci_high": ci_high,
                    "paired_permutation_p_one_sided_residual_gt_comparator": p_perm,
                }
                if metric == "accuracy":
                    row.update(mcnemar_one_sided_p(y, residual["pred"], comp["pred"]))
                pair_rows.append(row)

        # A metric-wise "best non-residual" comparator is useful for reviewer-facing robustness checks.
        for metric in METRICS:
            non_res = [m for m in MODEL_ORDER if m != "cross_attention_residual"]
            best_model = max(
                non_res,
                key=lambda m: metric_value(y, bundles[(task, m)]["prob"], bundles[(task, m)]["pred"], metric),
            )
            comp = bundles[(task, best_model)]
            delta, p_perm = paired_permutation_pvalue(
                y,
                residual["prob"],
                residual["pred"],
                comp["prob"],
                comp["pred"],
                metric,
                args.n_permutation,
                rng,
            )
            ci_low, ci_high = bootstrap_delta_ci(
                y,
                residual["prob"],
                residual["pred"],
                comp["prob"],
                comp["pred"],
                metric,
                args.n_bootstrap,
                rng,
            )
            row = {
                "task": task,
                "comparison": f"Residual CA vs best non-Residual ({MODEL_LABELS[best_model]})",
                "comparator": best_model,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "residual_value": metric_value(y, residual["prob"], residual["pred"], metric),
                "comparator_value": metric_value(y, comp["prob"], comp["pred"], metric),
                "delta_residual_minus_comparator": delta,
                "bootstrap_delta_ci_low": ci_low,
                "bootstrap_delta_ci_high": ci_high,
                "paired_permutation_p_one_sided_residual_gt_comparator": p_perm,
            }
            if metric == "accuracy":
                row.update(mcnemar_one_sided_p(y, residual["pred"], comp["pred"]))
            pair_rows.append(row)

    return pd.DataFrame(model_ci_rows), pd.DataFrame(pair_rows)


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_markdown_tables(out_dir: Path, metric_df: pd.DataFrame, ci_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    lines = ["# Hard-test ablation statistical report", ""]
    lines.append(
        "P-values are one-sided paired permutation tests for Residual CA > comparator. "
        "Confidence intervals are stratified bootstrap intervals over hard-test samples."
    )
    lines.append("")
    for task in ["amp", "tox"]:
        lines.append(f"## {task.upper()} hard-test metric table")
        task_metric = metric_df[metric_df["task"] == task].copy()
        task_ci = ci_df[ci_df["task"] == task].copy()
        table_rows = []
        for model in MODEL_ORDER:
            row = {"Model": MODEL_LABELS[model]}
            for metric in METRICS:
                value = float(task_metric[task_metric["model"] == model].iloc[0][metric])
                ci_row = task_ci[(task_ci["model"] == model) & (task_ci["metric"] == metric)].iloc[0]
                row[METRIC_LABELS[metric]] = (
                    f"{value:.4f} [{ci_row['bootstrap_ci_low']:.4f}, {ci_row['bootstrap_ci_high']:.4f}]"
                )
            table_rows.append(row)
        lines.append(pd.DataFrame(table_rows).to_markdown(index=False))
        lines.append("")
        lines.append(f"## {task.upper()} Residual CA pairwise p-values")
        task_pair = pair_df[(pair_df["task"] == task) & pair_df["comparison"].str.contains("best non-Residual") == False]
        p_rows = []
        for comp in ["concat_mlp", "esm_mlp", "ccd_mlp"]:
            row = {"Comparison": f"Residual CA vs {MODEL_LABELS[comp]}"}
            for metric in METRICS:
                prow = task_pair[(task_pair["comparator"] == comp) & (task_pair["metric"] == metric)].iloc[0]
                row[METRIC_LABELS[metric]] = (
                    f"Δ={prow['delta_residual_minus_comparator']:.4f}, "
                    f"p={format_p(float(prow['paired_permutation_p_one_sided_residual_gt_comparator']))}"
                )
            p_rows.append(row)
        lines.append(pd.DataFrame(p_rows).to_markdown(index=False))
        lines.append("")
    (out_dir / "hard_test_ablation_statistical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_df, bundles = load_ensemble_predictions(args)
    write_prediction_tables(out_dir, bundles)
    ci_df, pair_df = significance_tables(args, metric_df, bundles)

    metric_df.to_csv(out_dir / "hard_test_ablation_metrics_recomputed.csv", index=False)
    ci_df.to_csv(out_dir / "hard_test_ablation_metric_bootstrap_ci.csv", index=False)
    pair_df.to_csv(out_dir / "hard_test_ablation_pairwise_significance.csv", index=False)
    write_markdown_tables(out_dir, metric_df, ci_df, pair_df)

    manifest = {
        "baseline_checkpoint_root": str(Path(args.baseline_checkpoint_root).resolve()),
        "selected_checkpoint_root": str(Path(args.selected_checkpoint_root).resolve()),
        "baseline_ensemble_table": str(Path(args.baseline_ensemble_table).resolve()),
        "selected_ensemble_table": str(Path(args.selected_ensemble_table).resolve()),
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "seed": args.seed,
        "tests": {
            "paired_permutation": "one-sided test of Residual CA > comparator using paired sample-wise prediction swaps",
            "bootstrap_ci": "stratified bootstrap over positive and negative hard-test samples",
            "mcnemar": "one-sided exact McNemar/binomial test for accuracy only",
        },
    }
    (out_dir / "hard_test_ablation_significance_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] metric table -> {out_dir / 'hard_test_ablation_metrics_recomputed.csv'}")
    print(f"[DONE] pairwise p-values -> {out_dir / 'hard_test_ablation_pairwise_significance.csv'}")
    print(f"[DONE] report -> {out_dir / 'hard_test_ablation_statistical_report.md'}")


if __name__ == "__main__":
    main()
