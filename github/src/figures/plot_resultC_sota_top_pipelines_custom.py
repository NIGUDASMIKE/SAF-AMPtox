from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

METRICS = [
    ("macro_sensitivity", "Macro SEN"),
    ("macro_specificity", "Macro SPE"),
    ("macro_accuracy", "Macro Acc"),
    ("macro_f1", "Macro F1"),
    ("macro_mcc", "Macro MCC"),
    ("macro_roc_auc", "Macro ROC-AUC"),
    ("macro_pr_auc", "Macro PR-AUC"),
]

MODEL_ORDER = [
    "AMPlify-balanced + ToxinPred3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML",
    "Macrel-AMP + ToxinPred3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid",
    "AMPlify-imbalanced + ToxinPred3-Hybrid",
    "Macrel-AMP + ToxinPred3-Hybrid",
    "Residual Cross-Attention",
]

LABELS = {
    "AMPlify-balanced + ToxinPred3-ML": "B+ML",
    "AMPlify-imbalanced + ToxinPred3-ML": "I+ML",
    "Macrel-AMP + ToxinPred3-ML": "M+ML",
    "AMPlify-balanced + ToxinPred3-Hybrid": "B+H",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "I+H",
    "Macrel-AMP + ToxinPred3-Hybrid": "M+H",
    "Residual Cross-Attention": "RCA",
}

LABEL_KEY = [
    ("B", "AMPlify-B"),
    ("I", "AMPlify-I"),
    ("M", "Macrel-AMP"),
    ("ML", "ToxinPred3-ML"),
    ("H", "ToxinPred3-Hybrid"),
    ("RCA", "Residual CA"),
]

COLORS = {
    "AMPlify-balanced + ToxinPred3-ML": "#E6B5B3",
    "AMPlify-imbalanced + ToxinPred3-ML": "#F0D3D2",
    "Macrel-AMP + ToxinPred3-ML": "#CFCFCF",
    "AMPlify-balanced + ToxinPred3-Hybrid": "#B7C5D6",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "#D8E0EA",
    "Macrel-AMP + ToxinPred3-Hybrid": "#D8D8D8",
    "Residual Cross-Attention": "#0A0A8C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot seven-panel paired SOTA comparison for a selected result directory.")
    parser.add_argument("--sota-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fig-stem", default="resultC_fig_sota_top_paired_pipelines")
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        }
    )


def compact_label(model: str) -> str:
    return LABELS.get(model, model.replace(" + ", "\n+ "))


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_source(dual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, title in METRICS:
        max_value = float(dual.loc[dual["model"].isin(MODEL_ORDER), metric].max())
        for model in MODEL_ORDER:
            row = dual.loc[dual["model"] == model].iloc[0]
            value = float(row[metric])
            rows.append(
                {
                    "metric": metric,
                    "metric_label": title,
                    "model": model,
                    "plot_label": compact_label(model).replace("\n", " "),
                    "value": value,
                    "is_best_or_tie": bool(np.isclose(value, max_value) or value >= max_value - 5e-4),
                    "category": row["category"],
                }
            )
    return pd.DataFrame(rows)


def draw(dual: pd.DataFrame, source: pd.DataFrame, out_dir: Path, fig_stem: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 5.05), sharey=True)
    axes_flat = axes.ravel()
    for ax, (metric, title), letter in zip(axes_flat[:7], METRICS, list("ABCDEFG"), strict=True):
        values = [float(dual.loc[dual["model"] == model, metric].iloc[0]) for model in MODEL_ORDER]
        max_value = max(values)
        x = np.arange(len(MODEL_ORDER))
        ax.bar(
            x,
            values,
            width=0.74,
            color=[COLORS[model] for model in MODEL_ORDER],
            edgecolor=["black" if value >= max_value - 5e-4 else "white" for value in values],
            linewidth=[1.15 if value >= max_value - 5e-4 else 0.55 for value in values],
            zorder=3,
        )
        ax.set_ylim(0, 1.105)
        ax.set_title(f"{letter}  {title}", fontsize=9.2, fontweight="bold", loc="left", pad=10)
        if ax in (axes_flat[0], axes_flat[4]):
            ax.set_ylabel("Macro score", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([compact_label(model) for model in MODEL_ORDER], rotation=0, ha="center", fontsize=7)
        ax.grid(axis="y", color="#DCE2E7", linestyle="-", linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", pad=2, length=2.5)
        ax.tick_params(axis="y", labelsize=7, length=3.0)
        for xi, value in zip(x, values, strict=True):
            ax.text(xi, value + 0.024, f"{value:.4f}", ha="center", va="bottom", fontsize=6.1, rotation=0)
    note_ax = axes_flat[7]
    note_ax.axis("off")
    note_ax.text(0.02, 0.88, "Label key", ha="left", va="top", fontsize=8, fontweight="bold")
    y0 = 0.74
    for idx, (code, meaning) in enumerate(LABEL_KEY):
        x_text = 0.02 if idx < 3 else 0.50
        y_text = y0 - 0.15 * (idx % 3)
        note_ax.text(x_text, y_text, f"{code}: {meaning}", ha="left", va="center", fontsize=7.1)
    note_ax.text(0.02, 0.18, "Black outline: best/tie-best", ha="left", va="center", fontsize=7.1)
    note_ax.scatter([0.08], [0.05], s=90, marker="s", color=COLORS["Residual Cross-Attention"], edgecolor="black")
    note_ax.text(0.16, 0.05, "Residual CA", va="center", fontsize=7.1)
    note_ax.set_xlim(0, 1)
    note_ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.91, bottom=0.12, wspace=0.25, hspace=0.40)
    save_all(fig, out_dir / fig_stem)
    source.to_csv(out_dir / f"{fig_stem}_source_data.csv", index=False)


def write_tables(dual: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "model",
        "macro_sensitivity",
        "macro_specificity",
        "macro_accuracy",
        "macro_f1",
        "macro_mcc",
        "macro_roc_auc",
        "macro_pr_auc",
        "worst_task_roc_auc",
        "worst_task_pr_auc",
    ]
    table = dual.set_index("model").loc[MODEL_ORDER].reset_index()[cols]
    table.to_csv(out_dir / "table_resultC_sota_top_paired_pipelines.csv", index=False)
    rounded = table.copy()
    for column in cols[1:]:
        rounded[column] = rounded[column].map(lambda value: f"{float(value):.4f}")
    (out_dir / "table_resultC_sota_top_paired_pipelines.md").write_text(
        "# Paired external screening comparison\n\n"
        + rounded.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    setup_style()
    sota_dir = Path(args.sota_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dual = pd.read_csv(sota_dir / "tables" / "sota_dual_screening_metrics.csv")
    missing = sorted(set(MODEL_ORDER) - set(dual["model"]))
    if missing:
        raise RuntimeError(f"Missing paired models: {missing}")
    source = build_source(dual)
    draw(dual, source, out_dir, args.fig_stem)
    write_tables(dual, out_dir)
    print(f"[DONE] figure -> {out_dir / (args.fig_stem + '.svg')}")
    print(f"[DONE] table -> {out_dir / 'table_resultC_sota_top_paired_pipelines.csv'}")


if __name__ == "__main__":
    main()
