from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOTA_DIR = ROOT / "data--final" / "sota_duel"
TABLE_DIR = SOTA_DIR / "tables"
FIG_DIR = SOTA_DIR / "figures"
MANUSCRIPT_DIR = ROOT / "data--final" / "manuscript_figures"

DUAL_CSV = TABLE_DIR / "sota_dual_screening_metrics.csv"
FIG_STEM = MANUSCRIPT_DIR / "resultC_fig_sota_top_paired_pipelines"
SOURCE_CSV = MANUSCRIPT_DIR / "resultC_fig_sota_top_paired_pipelines_source_data.csv"
SOURCE_CSV_TABLE_DIR = TABLE_DIR / "resultC_fig_sota_paired_7metrics_source_data.csv"
TOP_TABLE_CSV = TABLE_DIR / "table_resultC_sota_top_paired_pipelines.csv"
TOP_TABLE_MD = TABLE_DIR / "table_resultC_sota_top_paired_pipelines.md"
AUDIT_CSV = TABLE_DIR / "sota_candidate_runnability_audit_extended.csv"
AUDIT_MD = TABLE_DIR / "sota_candidate_runnability_audit_extended.md"

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
    "AMPlify-balanced + ToxinPred3-ML": "AMPlify-B\n+ TP3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML": "AMPlify-I\n+ TP3-ML",
    "Macrel-AMP + ToxinPred3-ML": "Macrel\n+ TP3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid": "AMPlify-B\n+ TP3-H",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "AMPlify-I\n+ TP3-H",
    "Macrel-AMP + ToxinPred3-Hybrid": "Macrel\n+ TP3-H",
    "Residual Cross-Attention": "Residual\nCA",
}

COLORS = {
    "AMPlify-balanced + ToxinPred3-ML": "#E6B5B3",
    "AMPlify-imbalanced + ToxinPred3-ML": "#F0D3D2",
    "Macrel-AMP + ToxinPred3-ML": "#CFCFCF",
    "AMPlify-balanced + ToxinPred3-Hybrid": "#B7C5D6",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "#D8E0EA",
    "Macrel-AMP + ToxinPred3-Hybrid": "#D8D8D8",
    "Residual Cross-Attention": "#0A0A8C",
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
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def compact_label(model: str) -> str:
    return LABELS.get(model, model.replace(" + ", "\n+ "))


def selected_models() -> list[str]:
    return MODEL_ORDER.copy()


def build_source_data(dual: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, metric_label in METRICS:
        for model in models:
            rec = dual.loc[dual["model"] == model].iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "model": model,
                    "plot_label": compact_label(model).replace("\n", " "),
                    "value": float(rec[metric]),
                    "is_best_or_tie": bool(
                        np.isclose(float(rec[metric]), float(dual.loc[dual["model"].isin(models), metric].max()))
                    ),
                    "category": rec["category"],
                    "selection_rule": "Pre-specified paired pipelines: AMPlify-B/I or Macrel-AMP paired with ToxinPred3-ML/Hybrid; proposed model shown as the rightmost bar.",
                }
            )
    return pd.DataFrame(rows)


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")


def draw_figure(dual: pd.DataFrame, models: list[str], source: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8.9, 4.65), sharey=True)
    axes_flat = axes.ravel()
    letters = list("ABCDEFG")
    for ax, (metric, title), letter in zip(axes_flat[:7], METRICS, letters, strict=True):
        values = [
            float(dual.loc[dual["model"] == model, metric].iloc[0])
            for model in models
        ]
        max_value = max(values)
        x = np.arange(len(models))
        ax.bar(
            x,
            values,
            width=0.74,
            color=[COLORS.get(model, "#BFC6CF") for model in models],
            edgecolor=["black" if np.isclose(value, max_value) else "white" for value in values],
            linewidth=[1.15 if np.isclose(value, max_value) else 0.55 for value in values],
            zorder=3,
        )
        ymax = 1.04
        ax.set_ylim(0, ymax)
        ax.set_title(f"{letter}  {title}", fontsize=8, fontweight="bold", loc="left", pad=4)
        if ax in (axes_flat[0], axes_flat[4]):
            ax.set_ylabel("Macro score", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([compact_label(model) for model in models], rotation=58, ha="right", fontsize=5.25)
        ax.tick_params(axis="y", labelsize=6.2, length=3.0)
        ax.tick_params(axis="x", length=3.0)
        ax.grid(axis="y", color="#DCE2E7", linestyle="-", linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)
        for xi, value in zip(x, values, strict=True):
            ax.text(xi, value + ymax * 0.016, f"{value:.4f}", ha="center", va="bottom", fontsize=4.15, rotation=90)
    note_ax = axes_flat[7]
    note_ax.axis("off")
    note_ax.text(
        0.02,
        0.78,
        "Black outline:\nbest or tie-best\nwithin each metric",
        ha="left",
        va="top",
        fontsize=7,
        linespacing=1.35,
    )
    note_ax.scatter([0.10], [0.34], s=90, marker="s", color=COLORS["Residual Cross-Attention"], edgecolor="black")
    note_ax.text(0.19, 0.34, "Residual CA", va="center", fontsize=7)
    note_ax.set_xlim(0, 1)
    note_ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.94, bottom=0.20, wspace=0.28, hspace=0.54)
    save_all(fig, FIG_STEM)
    save_all(fig, SOTA_DIR / "figures" / "fig_resultC_sota_top_paired_pipelines")
    source.to_csv(SOURCE_CSV, index=False)
    source.to_csv(SOURCE_CSV_TABLE_DIR, index=False)


def write_top_table(dual: pd.DataFrame, models: list[str]) -> None:
    cols = [
        "model",
        "macro_accuracy",
        "macro_specificity",
        "macro_sensitivity",
        "macro_f1",
        "macro_mcc",
        "macro_roc_auc",
        "macro_pr_auc",
        "worst_task_roc_auc",
        "worst_task_pr_auc",
    ]
    table = dual.set_index("model").loc[models].reset_index()[cols]
    table.to_csv(TOP_TABLE_CSV, index=False)
    rounded = table.copy()
    for col in cols[1:]:
        rounded[col] = rounded[col].map(lambda x: f"{float(x):.4f}")
    TOP_TABLE_MD.write_text(
        "# Top paired external pipelines on shortcut-controlled hard tests\n\n"
        "Rows are the pre-specified paired pipelines used in the seven-panel external screening comparison, with the proposed model shown as the final row. "
        "Macro metrics average AMP hard-test and TOX hard-test results with equal task weight.\n\n"
        + rounded.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    jpa = pd.DataFrame(
        {
            "Model": [compact_label(model).replace("\n", " ") for model in models],
            "SEN": [float(dual.loc[dual["model"] == model, "macro_sensitivity"].iloc[0]) for model in models],
            "SPE": [float(dual.loc[dual["model"] == model, "macro_specificity"].iloc[0]) for model in models],
            "Acc": [float(dual.loc[dual["model"] == model, "macro_accuracy"].iloc[0]) for model in models],
            "F1": [float(dual.loc[dual["model"] == model, "macro_f1"].iloc[0]) for model in models],
            "MCC": [float(dual.loc[dual["model"] == model, "macro_mcc"].iloc[0]) for model in models],
            "AUC": [float(dual.loc[dual["model"] == model, "macro_roc_auc"].iloc[0]) for model in models],
            "PR-AUC": [float(dual.loc[dual["model"] == model, "macro_pr_auc"].iloc[0]) for model in models],
        }
    )
    jpa.to_csv(TABLE_DIR / "table1_jpa_style_sota_duel.csv", index=False)
    jpa_rounded = jpa.copy()
    for col in ["SEN", "SPE", "Acc", "F1", "MCC", "AUC", "PR-AUC"]:
        jpa_rounded[col] = jpa_rounded[col].map(lambda x: f"{float(x):.4f}")
    (TABLE_DIR / "table1_jpa_style_sota_duel.md").write_text(
        "# Table 1\n\n"
        "Performance comparison between the proposed residual cross-attention model and top paired SOTA pipelines "
        "on the shortcut-controlled dual hard-test benchmark. Metrics are macro-averaged over AMP hard test and "
        "TOX hard test with equal task weight.\n\n"
        + jpa_rounded.to_markdown(index=False)
        + "\n\n"
        "SEN: sensitivity; SPE: specificity; Acc: accuracy; F1: F1-score; "
        "MCC: Matthews correlation coefficient; AUC: ROC-AUC.\n",
        encoding="utf-8",
    )


def write_audit() -> None:
    rows = [
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
            "reason": "Local standalone package was available; ML and Hybrid modes produced hard-test probabilities.",
        },
        {
            "candidate": "Macrel-AMP",
            "task_role": "AMP predictor",
            "status": "included",
            "reason": "The pip package includes a pretrained ONNX AMP model and produced AMP_probability for all hard-test sequences.",
        },
        {
            "candidate": "Macrel-Hemo",
            "task_role": "hemolysis-oriented toxicity proxy",
            "status": "evaluated but not shown in main figure",
            "reason": "The hemolysis endpoint is not identical to the TOX benchmark and performed poorly on the TOX hard test; full rows remain in the CSV table.",
        },
        {
            "candidate": "AMPSeek",
            "task_role": "AMP/TOX workflow",
            "status": "not used as an independent row",
            "reason": "Containerized Nextflow workflow wrapping AMPlify, LocalColabFold and tAMPer; Docker is unavailable locally and the AMP component overlaps AMPlify.",
        },
        {
            "candidate": "tAMPer",
            "task_role": "TOX predictor",
            "status": "not runnable in current environment",
            "reason": "The available local route requires AMPSeek/ColabFold structure features and a Docker container; no standalone hard-test probability run was available.",
        },
        {
            "candidate": "HyPepTox-Fuse",
            "task_role": "TOX predictor",
            "status": "not runnable from local checkout",
            "reason": "Prediction code exists, but pretrained checkpoints/features are absent locally; only source/config/raw-data files were available.",
        },
        {
            "candidate": "MLpeptide",
            "task_role": "AMP/hemolysis notebooks",
            "status": "not runnable from local checkout",
            "reason": "Repository provides notebooks/training code and datasets, but no ready pretrained predictor weights or command-line inference entry was found.",
        },
        {
            "candidate": "AMPScannerV2",
            "task_role": "AMP predictor",
            "status": "not included",
            "reason": "No pip-installable package was found in the current environment, and GitHub clone attempts failed due network reset; no local pretrained checkout was available.",
        },
        {
            "candidate": "AMPfun",
            "task_role": "AMP predictor",
            "status": "not included",
            "reason": "No local checkout or pip-installable package was found, so no reproducible local probability inference was available.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_CSV, index=False)
    AUDIT_MD.write_text(
        "# Extended SOTA candidate runnability audit\n\n"
        + audit.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    dual = pd.read_csv(DUAL_CSV)
    missing = sorted(set(MODEL_ORDER) - set(dual["model"]))
    if missing:
        raise ValueError(f"Missing required paired models in {DUAL_CSV}: {missing}")
    models = selected_models()
    source = build_source_data(dual, models)
    draw_figure(dual, models, source)
    write_top_table(dual, models)
    write_audit()
    print(f"[DONE] figure -> {FIG_STEM.with_suffix('.svg')}")
    print(f"[DONE] source -> {SOURCE_CSV}")
    print(f"[DONE] top table -> {TOP_TABLE_CSV}")
    print(f"[DONE] audit -> {AUDIT_CSV}")


if __name__ == "__main__":
    main()
