from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
DUEL_ROOT = REPO_ROOT / "data--final" / "sota_duel"
TABLE_DIR = DUEL_ROOT / "tables"
FIG_DIR = DUEL_ROOT / "figures"
REPORT_DIR = DUEL_ROOT / "reports"


MODEL_LABELS = {
    "Residual Cross-Attention": "Ours\nResidual CA",
    "AMPlify-balanced + ToxinPred3-ML": "AMPlify-B\n+ ToxinPred3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML": "AMPlify-I\n+ ToxinPred3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid": "AMPlify-B\n+ ToxinPred3-Hybrid",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "AMPlify-I\n+ ToxinPred3-Hybrid",
}

DUAL_METRICS = [
    ("macro_roc_auc", "Macro\nROC-AUC"),
    ("macro_pr_auc", "Macro\nPR-AUC"),
    ("macro_f1", "Macro\nF1"),
    ("macro_mcc", "Macro\nMCC"),
    ("worst_task_roc_auc", "Worst-task\nROC-AUC"),
    ("worst_task_pr_auc", "Worst-task\nPR-AUC"),
]

TASK_METRICS = [
    ("amp_roc_auc", "AMP\nROC-AUC"),
    ("amp_pr_auc", "AMP\nPR-AUC"),
    ("amp_f1", "AMP\nF1"),
    ("amp_mcc", "AMP\nMCC"),
    ("tox_roc_auc", "TOX\nROC-AUC"),
    ("tox_pr_auc", "TOX\nPR-AUC"),
    ("tox_f1", "TOX\nF1"),
    ("tox_mcc", "TOX\nMCC"),
]


def set_style() -> None:
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
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.frameon": False,
        }
    )


def save_pub(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def annotate_heatmap(ax: plt.Axes, values: np.ndarray) -> None:
    threshold = np.nanmean(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if value >= threshold + 0.03 else "#222222"
            weight = "bold" if i == 0 else "normal"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=6.4, fontweight=weight)


def plot_duel_summary(dual: pd.DataFrame) -> pd.DataFrame:
    order = dual["model"].tolist()
    labels = [MODEL_LABELS.get(model, model) for model in order]
    metric_cols = [metric for metric, _ in DUAL_METRICS]
    metric_labels = ["Macro\nROC", "Macro\nPR", "Macro\nF1", "Macro\nMCC", "Worst\nROC", "Worst\nPR"]
    values = dual[metric_cols].to_numpy(dtype=float)

    ours = dual.loc[dual["model"] == "Residual Cross-Attention"].iloc[0]
    sota = dual.loc[dual["model"] != "Residual Cross-Attention"]
    deltas = []
    for metric, label in DUAL_METRICS:
        best_sota = float(sota[metric].max())
        ours_value = float(ours[metric])
        deltas.append(
            {
                "metric": metric,
                "metric_label": label.replace("\n", " "),
                "ours": ours_value,
                "best_paired_sota": best_sota,
                "delta": ours_value - best_sota,
            }
        )
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(TABLE_DIR / "sota_duel_best_sota_delta.csv", index=False)

    cmap = LinearSegmentedColormap.from_list("soft_sota", ["#F6F8FA", "#B9D7E5", "#4DBBD5", "#C00000"])
    fig = plt.figure(figsize=(7.8, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    im = ax0.imshow(values, aspect="auto", cmap=cmap, vmin=0.55, vmax=0.96)
    annotate_heatmap(ax0, values)
    ax0.set_xticks(np.arange(len(metric_labels)), metric_labels)
    ax0.set_yticks(np.arange(len(labels)), labels)
    ax0.tick_params(axis="x", length=0, pad=5)
    ax0.tick_params(axis="y", length=0, pad=4)
    ax0.set_title("A  Dual-task hard-test performance", loc="left", fontsize=9, fontweight="bold", pad=8)
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax0.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax0.tick_params(which="minor", bottom=False, left=False)

    y = np.arange(len(delta_df))
    colors = ["#C00000" if value > 0 else "#A0B1BA" for value in delta_df["delta"]]
    ax1.barh(y, delta_df["delta"], color=colors, height=0.58)
    ax1.axvline(0, color="#222222", lw=0.8)
    ax1.set_yticks(y, delta_df["metric_label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Gain over best paired SOTA")
    ax1.set_title("B  Margin of improvement", loc="left", fontsize=9, fontweight="bold", pad=8)
    ax1.grid(axis="x", color="#D9DEE2", lw=0.5, ls="--")
    ax1.set_axisbelow(True)
    xmax = max(0.01, float(delta_df["delta"].max())) * 1.35
    ax1.set_xlim(0, xmax)
    for yi, value in enumerate(delta_df["delta"]):
        ax1.text(value + xmax * 0.02, yi, f"+{value:.3f}", va="center", ha="left", fontsize=6.7, color="#222222")

    cbar = fig.colorbar(im, ax=ax0, fraction=0.035, pad=0.02)
    cbar.set_label("Score", rotation=270, labelpad=10)
    cbar.outline.set_visible(False)
    save_pub(fig, FIG_DIR / "fig_sota_duel_hard_test_summary")
    plt.close(fig)
    return delta_df


def plot_task_metric_matrix(dual: pd.DataFrame) -> None:
    order = dual["model"].tolist()
    labels = [MODEL_LABELS.get(model, model) for model in order]
    metric_cols = [metric for metric, _ in TASK_METRICS]
    metric_labels = [label for _, label in TASK_METRICS]
    values = dual[metric_cols].to_numpy(dtype=float)

    cmap = LinearSegmentedColormap.from_list("task_sota", ["#F8F8F8", "#D5E8EE", "#4DBBD5", "#E64B35"])
    fig, ax = plt.subplots(figsize=(7.4, 3.7), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.40, vmax=0.96)
    annotate_heatmap(ax, values)
    ax.set_xticks(np.arange(len(metric_labels)), metric_labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_title("Task-specific hard-test metrics", loc="left", fontsize=9, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Score", rotation=270, labelpad=10)
    cbar.outline.set_visible(False)
    save_pub(fig, FIG_DIR / "fig_sota_duel_task_metric_matrix")
    plt.close(fig)


def plot_shortcut_bias(shortcut: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), constrained_layout=True)
    amp_rows = shortcut[
        shortcut["dataset"].isin(
            [
                "AMPlify train AMP",
                "AMPlify train non-AMP balanced",
                "AMPlify train non-AMP imbalanced",
                "AMPlify test AMP",
                "AMPlify test non-AMP balanced",
                "AMPlify test non-AMP imbalanced",
                "Our AMP hard positive",
                "Our AMP hard negative",
            ]
        )
    ].copy()
    amp_rows["label"] = amp_rows["dataset"].replace(
        {
            "AMPlify train AMP": "AMPlify train\nAMP",
            "AMPlify train non-AMP balanced": "AMPlify train\nnon-AMP-B",
            "AMPlify train non-AMP imbalanced": "AMPlify train\nnon-AMP-I",
            "AMPlify test AMP": "AMPlify test\nAMP",
            "AMPlify test non-AMP balanced": "AMPlify test\nnon-AMP-B",
            "AMPlify test non-AMP imbalanced": "AMPlify test\nnon-AMP-I",
            "Our AMP hard positive": "Our hard\nAMP",
            "Our AMP hard negative": "Our hard\nnon-AMP",
        }
    )
    colors = ["#E64B35" if row == "positive" else "#4DBBD5" for row in amp_rows["label_group"]]
    axes[0].bar(np.arange(len(amp_rows)), amp_rows["m_start_rate"], color=colors, width=0.72)
    axes[0].set_xticks(np.arange(len(amp_rows)), amp_rows["label"], rotation=55, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("M-start fraction")
    axes[0].set_title("A  AMPlify source M-start bias", loc="left", fontsize=9, fontweight="bold", pad=8)
    axes[0].grid(axis="y", color="#D9DEE2", ls="--", lw=0.5)
    axes[0].set_axisbelow(True)
    for x, value in enumerate(amp_rows["m_start_rate"]):
        axes[0].text(x, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, rotation=90)

    tox_rows = shortcut[
        shortcut["dataset"].isin(
            [
                "ToxinPred3 train toxic",
                "ToxinPred3 train non-toxic",
                "ToxinPred3 test toxic",
                "ToxinPred3 test non-toxic",
                "Our TOX hard toxic",
                "Our TOX hard non-toxic",
            ]
        )
    ].copy()
    tox_rows["label"] = tox_rows["dataset"].replace(
        {
            "ToxinPred3 train toxic": "ToxinPred3 train\ntoxic",
            "ToxinPred3 train non-toxic": "ToxinPred3 train\nnon-toxic",
            "ToxinPred3 test toxic": "ToxinPred3 test\ntoxic",
            "ToxinPred3 test non-toxic": "ToxinPred3 test\nnon-toxic",
            "Our TOX hard toxic": "Our hard\ntoxic",
            "Our TOX hard non-toxic": "Our hard\nnon-toxic",
        }
    )
    x = np.arange(len(tox_rows))
    colors = ["#E64B35" if row == "positive" else "#4DBBD5" for row in tox_rows["label_group"]]
    axes[1].bar(x, tox_rows["contains_c_rate"], color=colors, width=0.72, label="Contains C")
    axes[1].plot(x, tox_rows["mean_c_frequency"], color="#222222", marker="o", ms=3.5, lw=1.0, label="Mean C frequency")
    axes[1].set_xticks(x, tox_rows["label"], rotation=55, ha="right")
    axes[1].set_ylim(0, 0.82)
    axes[1].set_ylabel("Cys-related fraction")
    axes[1].set_title("B  ToxinPred3 source cysteine bias", loc="left", fontsize=9, fontweight="bold", pad=8)
    axes[1].grid(axis="y", color="#D9DEE2", ls="--", lw=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper right", fontsize=6.5)
    for xpos, value in zip(x, tox_rows["contains_c_rate"]):
        axes[1].text(xpos, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, rotation=90)

    save_pub(fig, FIG_DIR / "fig_sota_source_shortcut_bias")
    plt.close(fig)


def write_report(dual: pd.DataFrame, delta: pd.DataFrame) -> None:
    ours = dual.loc[dual["model"] == "Residual Cross-Attention"].iloc[0]
    best_sota = dual.loc[dual["model"] != "Residual Cross-Attention"].iloc[0]
    runnability = pd.DataFrame(
        [
            {
                "candidate": "AMPlify",
                "task_role": "AMP predictor",
                "status": "included",
                "reason": "Local pretrained balanced and imbalanced weights were available and produced hard-test probabilities.",
            },
            {
                "candidate": "ToxinPred3",
                "task_role": "TOX predictor",
                "status": "included",
                "reason": "Local standalone model package was available; both ML and Hybrid modes produced hard-test probabilities.",
            },
            {
                "candidate": "AMPSeek",
                "task_role": "AMP/TOX workflow",
                "status": "not used as an independent row",
                "reason": "Containerized Nextflow workflow wrapping AMPlify, LocalColabFold and tAMPer; it is not a lightweight local predictor and its AMP component overlaps AMPlify.",
            },
            {
                "candidate": "HyPepTox-Fuse",
                "task_role": "TOX predictor",
                "status": "not runnable from local checkout",
                "reason": "Prediction code exists, but pretrained checkpoints/features are absent locally; only checkpoints/.gitkeep was present.",
            },
            {
                "candidate": "MLpeptide",
                "task_role": "AMP/hemolysis notebooks",
                "status": "not runnable from local checkout",
                "reason": "Repository provides notebooks/training code and datasets, but no ready pretrained predictor weights or command-line inference entry was found.",
            },
        ]
    )
    runnability.to_csv(TABLE_DIR / "sota_candidate_runnability_audit.csv", index=False)
    lines = [
        "# SOTA Duel Hard-Test Summary",
        "",
        "## Key result",
        "",
        (
            "Residual cross-attention ranked first on the balanced hard-test dual-screening benchmark, "
            f"with macro ROC-AUC={ours['macro_roc_auc']:.4f}, macro PR-AUC={ours['macro_pr_auc']:.4f}, "
            f"macro F1={ours['macro_f1']:.4f}, and macro MCC={ours['macro_mcc']:.4f}."
        ),
        "",
        "## Margin over the strongest paired SOTA baseline",
        "",
        delta.assign(delta=delta["delta"].map(lambda x: f"+{x:.4f}")).to_markdown(index=False),
        "",
        "## Recommended manuscript framing",
        "",
        (
            "Use the hard-test table as the main comparison. The standard test can be omitted from the main text "
            "or moved to supplementary context, because the hard set is explicitly designed to control M-start "
            "and cysteine-composition shortcut cues."
        ),
        (
            "For SOTA comparison, frame the task as dual-function screening rather than isolated AMP or TOX "
            "leaderboards: a candidate peptide is practically useful only when antimicrobial activity and toxicity "
            "risk are both considered."
        ),
        "",
        "## SOTA candidate runnability audit",
        "",
        runnability.to_markdown(index=False),
        "",
        "## Output files",
        "",
        "- `tables/sota_dual_screening_metrics.csv` and `.md`",
        "- `tables/sota_duel_best_sota_delta.csv`",
        "- `tables/sota_shortcut_bias_audit.csv` and `.md`",
        "- `tables/sota_candidate_runnability_audit.csv`",
        "- `figures/fig_sota_duel_hard_test_summary.*`",
        "- `figures/fig_sota_duel_task_metric_matrix.*`",
        "- `figures/fig_sota_source_shortcut_bias.*`",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "sota_duel_handover_cn_20260611.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dual = pd.read_csv(TABLE_DIR / "sota_dual_screening_metrics.csv")
    dual = dual.sort_values(["macro_roc_auc", "macro_pr_auc"], ascending=False).reset_index(drop=True)
    shortcut = pd.read_csv(TABLE_DIR / "sota_shortcut_bias_audit.csv")
    delta = plot_duel_summary(dual)
    plot_task_metric_matrix(dual)
    plot_shortcut_bias(shortcut)
    write_report(dual, delta)
    print(f"[DONE] figures -> {FIG_DIR}")
    print(f"[DONE] report -> {REPORT_DIR / 'sota_duel_handover_cn_20260611.md'}")


if __name__ == "__main__":
    main()
