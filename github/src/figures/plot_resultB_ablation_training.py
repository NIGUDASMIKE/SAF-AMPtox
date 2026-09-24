from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data--final" / "fusion_evidence_package" / "source_data"
OUT_DIR = ROOT / "data--final" / "manuscript_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_SOURCE = SOURCE_DIR / "fig3_curated_four_model_metric_bars_source.csv"
LOSS_SOURCE = SOURCE_DIR / "fig4_training_loss_source.csv"

FIG_BASENAME = OUT_DIR / "fig2_resultB_ablation_training"
PLOT_SOURCE = OUT_DIR / "fig2_resultB_ablation_training_source_data.csv"
LOSS_PLOT_SOURCE = OUT_DIR / "fig2_resultB_ablation_training_loss_source_data.csv"
REPORT_PATH = OUT_DIR / "fig2_resultB_ablation_training_report.md"

MODEL_ORDER = ["CCD", "ESM-2", "Concat", "Residual CA"]
MODEL_COLORS = {
    "CCD": "#8FA3D1",
    "ESM-2": "#63BFA5",
    "Concat": "#7F8C99",
    "Residual CA": "#C73E3A",
}


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
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 140,
        }
    )


def clean_axis(ax: plt.Axes, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color="#DDE3E8", lw=0.5, ls="-", alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3, color="#20242A", labelcolor="#20242A")
    ax.spines["left"].set_color("#20242A")
    ax.spines["bottom"].set_color("#20242A")


def plot_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    metrics: list[str],
    title: str,
    ylim: tuple[float, float],
) -> None:
    panel = df[(df["Task"] == task) & (df["Metric"].isin(metrics))].copy()
    panel["Metric"] = pd.Categorical(panel["Metric"], categories=metrics, ordered=True)
    panel["Model short"] = pd.Categorical(panel["Model short"], categories=MODEL_ORDER, ordered=True)
    panel = panel.sort_values(["Metric", "Model short"])

    centers = np.arange(len(metrics))
    width = 0.17
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))

    metric_best = panel.groupby("Metric", observed=True)["Mean"].max()
    best_tol = 5e-4

    for model, offset in zip(MODEL_ORDER, offsets):
        rows = panel[panel["Model short"] == model]
        values = rows["Mean"].to_numpy()
        sd = rows["SD"].to_numpy()
        edge_colors = []
        line_widths = []
        for metric, value in zip(rows["Metric"].astype(str), values):
            if value >= float(metric_best.loc[metric]) - best_tol:
                edge_colors.append("#20242A")
                line_widths.append(0.95)
            else:
                edge_colors.append("white")
                line_widths.append(0.45)
        bars = ax.bar(
            centers + offset,
            values,
            width,
            color=MODEL_COLORS[model],
            edgecolor=edge_colors,
            linewidth=line_widths,
            label=model,
            zorder=3,
        )
        ax.errorbar(
            centers + offset,
            values,
            yerr=sd,
            fmt="none",
            ecolor="#2F343A",
            elinewidth=0.45,
            capsize=1.4,
            capthick=0.45,
            alpha=0.85,
            zorder=4,
        )
    concat = panel[panel["Model short"] == "Concat"].set_index("Metric")["Mean"]
    residual = panel[panel["Model short"] == "Residual CA"].set_index("Metric")["Mean"]
    for i, metric in enumerate(metrics):
        delta = float((residual.loc[metric] - concat.loc[metric]) * 100)
        y = ylim[1] - 0.018
        color = "#B62F2F" if delta >= 0 else "#53606A"
        ax.text(
            centers[i],
            y,
            f"{delta:+.2f} pp",
            ha="center",
            va="top",
            fontsize=5.0,
            color=color,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(metrics)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Performance score")
    ax.set_title(title, pad=6)
    clean_axis(ax)


def plot_loss(ax: plt.Axes, loss: pd.DataFrame, task: str, title: str, ylim: tuple[float, float]) -> None:
    model_map = {
        "ccd_mlp": "CCD",
        "esm_mlp": "ESM-2",
        "concat_mlp": "Concat",
        "cross_attention_residual": "Residual CA",
    }
    panel = loss[loss["task"] == task].copy()
    for model_key, label in model_map.items():
        rows = panel[panel["model"] == model_key].sort_values("epoch")
        if rows.empty:
            continue
        x = rows["epoch"].to_numpy()
        y = rows["mean_train_loss"].to_numpy()
        sd = rows["std_train_loss"].to_numpy()
        ax.plot(x, y, color=MODEL_COLORS[label], lw=1.35, label=label)
        ax.fill_between(x, y - sd, y + sd, color=MODEL_COLORS[label], alpha=0.08, linewidth=0)

    ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title(title, pad=7)
    clean_axis(ax)


def main() -> None:
    setup_style()
    bars = pd.read_csv(BAR_SOURCE)
    loss = pd.read_csv(LOSS_SOURCE)

    fig = plt.figure(figsize=(7.45, 3.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.62], hspace=0.56, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_grouped_bars(
        ax_a,
        bars,
        "AMP",
        ["SEN", "Acc", "F1", "MCC", "AUPRC"],
        "AMP hard test: detection-oriented endpoints",
        (0.70, 1.005),
    )
    plot_grouped_bars(
        ax_b,
        bars,
        "TOX",
        ["SPE", "Precision", "AUC", "AUPRC"],
        "TOX hard test: safety-oriented endpoints",
        (0.70, 1.005),
    )
    plot_loss(ax_c, loss, "amp", "AMP training loss", (0.0, 0.305))
    plot_loss(ax_d, loss, "tox", "TOX training loss", (0.0, 0.51))

    handles = [
        mpl.patches.Patch(color=MODEL_COLORS[model], label=model)
        for model in MODEL_ORDER
    ]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="#20242A", linewidth=0.95, label="best/tie-best"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=5,
        frameon=False,
        columnspacing=0.90,
        handlelength=1.0,
    )

    for label, ax, y_pos in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d], [1.12, 1.12, 1.24, 1.24]):
        ax.text(
            -0.15,
            y_pos,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            color="#20242A",
        )

    fig.savefig(f"{FIG_BASENAME}.svg", bbox_inches="tight")
    fig.savefig(f"{FIG_BASENAME}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIG_BASENAME}.pdf", bbox_inches="tight")
    plt.close(fig)

    bars.assign(source_panel="A/B hard-test bars").to_csv(PLOT_SOURCE, index=False)
    loss.assign(source_panel="C/D training loss curves").to_csv(LOSS_PLOT_SOURCE, index=False)
    amp_rows = bars[(bars["Task"] == "AMP") & (bars["Test set"] == "AMP hard test")]
    tox_rows = bars[(bars["Task"] == "TOX") & (bars["Test set"] == "TOX hard test")]
    report = f"""# Result B Ablation Figure Report

## Figure contract

Claim: Residual cross-attention improves the relevant hard-test endpoints over direct concatenation under the same CCD and ESM-2 inputs, while training remains stable across tasks.

## Main hard-test comparison

Primary comparator is Direct Concat MLP, because it uses the same CCD and ESM-2 inputs as Residual CA.

AMP hard test endpoints:
{amp_rows[amp_rows["Metric"].isin(["SEN", "Acc", "F1", "MCC", "AUPRC"])][["Metric", "Model short", "Mean", "SD"]].to_markdown(index=False)}

TOX hard test endpoints:
{tox_rows[tox_rows["Metric"].isin(["SPE", "Precision", "AUC", "AUPRC"])][["Metric", "Model short", "Mean", "SD"]].to_markdown(index=False)}

## Interpretation guardrail

Do not claim Residual CA dominates every metric or every unimodal baseline. The defensible main-text claim is that Residual CA improves over Direct Concat on selected task-relevant hard-test endpoints: AMP detection metrics and TOX safety/ranking metrics.

## Outputs

- SVG: `{FIG_BASENAME}.svg`
- PNG: `{FIG_BASENAME}.png`
- PDF: `{FIG_BASENAME}.pdf`
- Bar source data: `{PLOT_SOURCE}`
- Loss source data: `{LOSS_PLOT_SOURCE}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {FIG_BASENAME}.svg")
    print(f"Wrote {FIG_BASENAME}.png")
    print(f"Wrote {FIG_BASENAME}.pdf")
    print(f"Wrote {PLOT_SOURCE}")
    print(f"Wrote {LOSS_PLOT_SOURCE}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
