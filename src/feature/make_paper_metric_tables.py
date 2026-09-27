from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAMES = {
    "concat_mlp": "Concat baseline",
    "cross_attention_residual": "Residual Cross-Attention",
}

SPLIT_NAMES = {
    "test": "Standard test",
    "test_hard_amp": "AMP hard test",
    "test_hard_tox": "TOX hard test",
}

METRICS_MAIN = [
    ("sensitivity", "SEN"),
    ("specificity", "SPE"),
    ("accuracy", "Acc"),
    ("f1", "F1"),
    ("mcc", "MCC"),
    ("roc_auc", "AUC"),
    ("pr_auc", "AUPRC"),
]

METRICS_SUPPLEMENT = [
    ("sensitivity", "SEN"),
    ("specificity", "SPE"),
    ("accuracy", "Acc"),
    ("precision", "Precision"),
    ("npv", "NPV"),
    ("balanced_accuracy", "BA"),
    ("f1", "F1"),
    ("mcc", "MCC"),
    ("roc_auc", "AUC"),
    ("pr_auc", "AUPRC"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready metric tables for Concat vs Residual Cross-Attention.")
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "fusion_model_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "paper_tables"),
    )
    return parser.parse_args()


def mean_std(row: pd.Series, metric: str) -> str:
    return f"{row[f'mean_{metric}']:.4f} +/- {row[f'std_{metric}']:.4f}"


def build_table(summary: pd.DataFrame, metrics: list[tuple[str, str]], bold_best: bool) -> pd.DataFrame:
    work = summary[
        summary["model"].isin(MODEL_NAMES)
        & summary["split"].isin(["test", "test_hard_amp", "test_hard_tox"])
    ].copy()
    work["Task"] = work["task"].str.upper()
    work["Test set"] = work["split"].map(SPLIT_NAMES)
    work["Model"] = work["model"].map(MODEL_NAMES)

    rows: list[dict[str, str]] = []
    for _, row in work.sort_values(["task", "split", "model"]).iterrows():
        out = {
            "Task": row["Task"],
            "Test set": row["Test set"],
            "Model": row["Model"],
        }
        for metric, label in metrics:
            out[label] = mean_std(row, metric)
        rows.append(out)
    table = pd.DataFrame(rows)

    if not bold_best:
        return table

    for (task, split), sub in work.groupby(["Task", "Test set"]):
        for metric, label in metrics:
            best = sub[f"mean_{metric}"].max()
            best_models = set(sub.loc[sub[f"mean_{metric}"] == best, "Model"])
            mask = (table["Task"] == task) & (table["Test set"] == split) & table["Model"].isin(best_models)
            table.loc[mask, label] = table.loc[mask, label].map(lambda value: f"**{value}**")
    return table


def build_delta_table(summary: pd.DataFrame, metrics: list[tuple[str, str]]) -> pd.DataFrame:
    work = summary[
        summary["model"].isin(MODEL_NAMES)
        & summary["split"].isin(["test", "test_hard_amp", "test_hard_tox"])
    ].copy()
    rows = []
    for (task, split), sub in work.groupby(["task", "split"]):
        concat = sub[sub["model"] == "concat_mlp"].iloc[0]
        residual = sub[sub["model"] == "cross_attention_residual"].iloc[0]
        row: dict[str, str | float] = {
            "Task": task.upper(),
            "Test set": SPLIT_NAMES[split],
        }
        for metric, label in metrics:
            delta = float(residual[f"mean_{metric}"] - concat[f"mean_{metric}"])
            row[f"Delta {label}"] = delta
        rows.append(row)
    return pd.DataFrame(rows)


def build_positive_gain_table(delta_table: pd.DataFrame, min_gain: float = 0.0005) -> pd.DataFrame:
    rows = []
    for _, row in delta_table.iterrows():
        for column in delta_table.columns:
            if not column.startswith("Delta "):
                continue
            gain = float(row[column])
            if gain > min_gain:
                rows.append(
                    {
                        "Task": row["Task"],
                        "Test set": row["Test set"],
                        "Metric": column.replace("Delta ", ""),
                        "Residual - Concat": f"+{gain:.4f}",
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(title: str, table: pd.DataFrame, path: Path) -> None:
    text = f"# {title}\n\n" + table.to_markdown(index=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv)

    main_table = build_table(summary, METRICS_MAIN, bold_best=True)
    supplement_table = build_table(summary, METRICS_SUPPLEMENT, bold_best=False)
    delta_table = build_delta_table(summary, METRICS_MAIN)
    positive_gain_table = build_positive_gain_table(delta_table)

    main_table.to_csv(out_dir / "table_main_concat_vs_residual.csv", index=False)
    supplement_table.to_csv(out_dir / "table_supplement_all_metrics.csv", index=False)
    delta_table.to_csv(out_dir / "table_delta_residual_minus_concat.csv", index=False)
    positive_gain_table.to_csv(out_dir / "table_key_gains_residual_vs_concat.csv", index=False)

    write_markdown(
        "Table 1. Concat baseline vs Residual Cross-Attention on standard and hard tests",
        main_table,
        out_dir / "table_main_concat_vs_residual.md",
    )
    write_markdown(
        "Supplementary Table. Full metric panel",
        supplement_table,
        out_dir / "table_supplement_all_metrics.md",
    )
    write_markdown(
        "Delta Table. Residual Cross-Attention minus Concat baseline",
        delta_table,
        out_dir / "table_delta_residual_minus_concat.md",
    )
    write_markdown(
        "Key Gain Table. Positive deltas of Residual Cross-Attention over Concat baseline",
        positive_gain_table,
        out_dir / "table_key_gains_residual_vs_concat.md",
    )

    print(f"[DONE] main table -> {out_dir / 'table_main_concat_vs_residual.md'}")
    print(f"[DONE] supplementary table -> {out_dir / 'table_supplement_all_metrics.md'}")
    print(f"[DONE] delta table -> {out_dir / 'table_delta_residual_minus_concat.md'}")
    print(f"[DONE] key gains table -> {out_dir / 'table_key_gains_residual_vs_concat.md'}")


if __name__ == "__main__":
    main()
