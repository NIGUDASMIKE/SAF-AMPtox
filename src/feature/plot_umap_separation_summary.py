from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_ORDER = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]
MODEL_LABELS = {
    "ccd_mlp": "CCD",
    "esm_mlp": "ESM-2",
    "concat_mlp": "Concat",
    "cross_attention_residual": "Residual CA",
}
COLORS = {
    "ccd_mlp": "#8DA0CB",
    "esm_mlp": "#66C2A5",
    "concat_mlp": "#7A8A99",
    "cross_attention_residual": "#C43C39",
}

PLOT_METRICS = [
    ("silhouette", "Silhouette\nhigher better"),
    ("davies_bouldin", "Davies-Bouldin\nlower better"),
    ("fisher_ratio", "Fisher ratio\nhigher better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAP/latent geometry separation summary.")
    parser.add_argument(
        "--summary-csv",
        default=str(REPO_ROOT / "data--final" / "fusion_umap_hard" / "tables" / "umap_separation_metrics_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data--final" / "fusion_umap_hard" / "figures"),
    )
    parser.add_argument(
        "--source-dir",
        default=str(REPO_ROOT / "data--final" / "fusion_umap_hard" / "source_data"),
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2D333B",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    out_dir = Path(args.out_dir)
    source_dir = Path(args.source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv)

    rows = []
    for task in ["amp", "tox"]:
        for metric, label in PLOT_METRICS:
            for model in MODEL_ORDER:
                hit = summary[(summary["task"] == task) & (summary["model"] == model)].iloc[0]
                rows.append(
                    {
                        "task": task.upper(),
                        "metric": metric,
                        "metric_label": label,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "mean": hit[f"mean_{metric}"],
                        "std": hit[f"std_{metric}"],
                    }
                )
    source = pd.DataFrame(rows)
    source.to_csv(source_dir / "fig_umap_geometric_separation_bars_source.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.7))
    letters = list("ABCDEF")
    idx = 0
    for row_idx, task in enumerate(["AMP", "TOX"]):
        for col_idx, (metric, label) in enumerate(PLOT_METRICS):
            ax = axes[row_idx, col_idx]
            sub = source[(source["task"] == task) & (source["metric"] == metric)].set_index("model").loc[MODEL_ORDER]
            x = np.arange(len(MODEL_ORDER))
            ax.bar(
                x,
                sub["mean"].values,
                yerr=sub["std"].values,
                color=[COLORS[model] for model in MODEL_ORDER],
                edgecolor="white",
                linewidth=0.4,
                capsize=2.0,
            )
            for xi, mean, std in zip(x, sub["mean"].values, sub["std"].values, strict=True):
                offset = 0.035 * max(abs(sub["mean"].max()), 1.0)
                ax.text(xi, mean + std + offset, f"{mean:.2f}", ha="center", va="bottom", fontsize=5.8, rotation=90)
            ax.set_title(f"{task} | {label}", pad=7)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=35, ha="right")
            ax.grid(axis="y", color="#D8DEE6", linestyle="--", linewidth=0.5, alpha=0.75)
            ax.set_axisbelow(True)
            ax.text(-0.20, 1.14, letters[idx], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
            idx += 1
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    fig.subplots_adjust(hspace=0.62, wspace=0.40)
    save_figure(fig, "fig_umap_geometric_separation_bars", out_dir)
    print(f"[DONE] {out_dir / 'fig_umap_geometric_separation_bars.svg'}")


if __name__ == "__main__":
    main()
