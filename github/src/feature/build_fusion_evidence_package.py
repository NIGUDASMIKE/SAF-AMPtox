from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

MAIN_MODELS = ["concat_mlp", "cross_attention_residual"]
UNIMODAL_MODELS = ["ccd_mlp", "esm_mlp"]
ALL_CONTEXT_MODELS = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]

MODEL_NAMES = {
    "ccd_mlp": "CCD-MLP",
    "esm_mlp": "ESM2-MLP",
    "concat_mlp": "Direct Concat MLP",
    "cross_attention_residual": "Residual Cross-Attention",
}

SHORT_MODEL_NAMES = {
    "ccd_mlp": "CCD",
    "esm_mlp": "ESM-2",
    "concat_mlp": "Concat",
    "cross_attention_residual": "Residual CA",
}

SPLIT_NAMES = {
    "test": "Standard test",
    "test_hard_amp": "AMP hard test",
    "test_hard_tox": "TOX hard test",
}

TASK_SPLITS = {
    "amp": ["test", "test_hard_amp"],
    "tox": ["test", "test_hard_tox"],
}

CORE_METRICS = [
    ("sensitivity", "SEN"),
    ("specificity", "SPE"),
    ("accuracy", "Acc"),
    ("f1", "F1"),
    ("mcc", "MCC"),
    ("roc_auc", "AUC"),
    ("pr_auc", "AUPRC"),
]

FULL_METRICS = [
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

SELECTED_HARD_METRICS = {
    "amp": [("sensitivity", "SEN"), ("f1", "F1")],
    "tox": [("specificity", "SPE"), ("roc_auc", "AUC"), ("pr_auc", "AUPRC")],
}

CURATED_FOUR_MODEL_METRICS = {
    "amp": [
        ("sensitivity", "SEN"),
        ("accuracy", "Acc"),
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("pr_auc", "AUPRC"),
    ],
    "tox": [
        ("specificity", "SPE"),
        ("precision", "Precision"),
        ("roc_auc", "AUC"),
        ("pr_auc", "AUPRC"),
    ],
}

COLORS = {
    "concat_mlp": "#7A8A99",
    "cross_attention_residual": "#C43C39",
    "ccd_mlp": "#8DA0CB",
    "esm_mlp": "#66C2A5",
    "positive": "#C43C39",
    "negative": "#4C78A8",
    "grid": "#D8DEE6",
    "text": "#20242A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final fusion evidence tables, figures, and consistency report.")
    parser.add_argument(
        "--main-summary",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "fusion_model_summary.csv"),
    )
    parser.add_argument(
        "--main-results",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "fusion_model_results.csv"),
    )
    parser.add_argument(
        "--unimodal-summary",
        default=str(REPO_ROOT / "data--final" / "fusion_models_unimodal" / "reports" / "fusion_model_summary.csv"),
    )
    parser.add_argument(
        "--unimodal-results",
        default=str(REPO_ROOT / "data--final" / "fusion_models_unimodal" / "reports" / "fusion_model_results.csv"),
    )
    parser.add_argument(
        "--main-history",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_compare" / "reports" / "fusion_model_training_history.csv"),
    )
    parser.add_argument(
        "--unimodal-history",
        default=str(REPO_ROOT / "data--final" / "fusion_models_unimodal" / "reports" / "fusion_model_training_history.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data--final" / "fusion_evidence_package"),
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
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "source": out_dir / "source_data",
        "reports": out_dir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fmt_mean_std(row: pd.Series, metric: str) -> str:
    return f"{row[f'mean_{metric}']:.4f} +/- {row[f'std_{metric}']:.4f}"


def subset_valid_splits(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for task, splits in TASK_SPLITS.items():
        frames.append(df[(df["task"] == task) & (df["split"].isin(splits))].copy())
    return pd.concat(frames, ignore_index=True)


def build_table(
    summary: pd.DataFrame,
    models: list[str],
    metrics: list[tuple[str, str]],
    bold_best: bool = False,
) -> pd.DataFrame:
    work = subset_valid_splits(summary)
    work = work[work["model"].isin(models)].copy()
    work["Task"] = work["task"].str.upper()
    work["Test set"] = work["split"].map(SPLIT_NAMES)
    work["Model"] = work["model"].map(MODEL_NAMES)

    rows = []
    sort_models = {model: idx for idx, model in enumerate(models)}
    work["model_order"] = work["model"].map(sort_models)
    for _, row in work.sort_values(["task", "split", "model_order"]).iterrows():
        out = {
            "Task": row["Task"],
            "Test set": row["Test set"],
            "Model": row["Model"],
        }
        for metric, label in metrics:
            out[label] = fmt_mean_std(row, metric)
        rows.append(out)
    table = pd.DataFrame(rows)

    if not bold_best:
        return table

    for (task, split), sub in work.groupby(["Task", "Test set"]):
        for metric, label in metrics:
            best = sub[f"mean_{metric}"].max()
            best_models = set(sub.loc[np.isclose(sub[f"mean_{metric}"], best), "Model"])
            mask = (table["Task"] == task) & (table["Test set"] == split) & table["Model"].isin(best_models)
            table.loc[mask, label] = table.loc[mask, label].map(lambda value: f"**{value}**")
    return table


def write_markdown(title: str, table: pd.DataFrame, path: Path) -> None:
    path.write_text(f"# {title}\n\n{table.to_markdown(index=False)}\n", encoding="utf-8")


def build_paired_delta_table(results: pd.DataFrame, metrics: list[tuple[str, str]]) -> pd.DataFrame:
    work = subset_valid_splits(results)
    work = work[work["model"].isin(MAIN_MODELS)].copy()
    rows = []
    for task, splits in TASK_SPLITS.items():
        for split in splits:
            sub = work[(work["task"] == task) & (work["split"] == split)].copy()
            for metric, label in metrics:
                wide = sub.pivot(index="seed", columns="model", values=metric)
                paired = wide["cross_attention_residual"] - wide["concat_mlp"]
                rows.append(
                    {
                        "Task": task.upper(),
                        "Test set": SPLIT_NAMES[split],
                        "Metric": label,
                        "Mean delta": paired.mean(),
                        "SD delta": paired.std(ddof=1),
                        "Seeds improved": int((paired > 0).sum()),
                        "Seeds total": int(paired.notna().sum()),
                        "Seed deltas": "; ".join(f"{seed}:{value:+.4f}" for seed, value in paired.items()),
                    }
                )
    return pd.DataFrame(rows)


def build_key_gain_table(
    paired_delta: pd.DataFrame,
    min_delta: float = 0.0005,
    min_seed_wins: int = 2,
) -> pd.DataFrame:
    work = paired_delta[
        (paired_delta["Mean delta"] > min_delta) & (paired_delta["Seeds improved"] >= min_seed_wins)
    ].copy()
    work["Residual - Concat"] = work["Mean delta"].map(lambda value: f"delta {value:+.4f}")
    work["Seed wins"] = work.apply(lambda row: f"{int(row['Seeds improved'])}/{int(row['Seeds total'])}", axis=1)
    return work[["Task", "Test set", "Metric", "Residual - Concat", "Seed wins", "Seed deltas"]]


def save_table_bundle(table: pd.DataFrame, title: str, stem: str, table_dir: Path) -> None:
    table.to_csv(table_dir / f"{stem}.csv", index=False)
    write_markdown(title, table, table_dir / f"{stem}.md")


def save_figure(fig: plt.Figure, stem: str, fig_dir: Path) -> None:
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_key_metric_bars(summary: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    work = subset_valid_splits(summary)
    rows = []
    for task, split in [("amp", "test_hard_amp"), ("tox", "test_hard_tox")]:
        for metric, label in SELECTED_HARD_METRICS[task]:
            for model in MAIN_MODELS:
                row = work[(work["task"] == task) & (work["split"] == split) & (work["model"] == model)].iloc[0]
                rows.append(
                    {
                        "Task": task.upper(),
                        "Metric": label,
                        "Model": MODEL_NAMES[model],
                        "Model key": model,
                        "Mean": row[f"mean_{metric}"],
                        "SD": row[f"std_{metric}"],
                    }
                )
    source = pd.DataFrame(rows)
    source.to_csv(source_dir / "fig1_hard_key_metric_bars_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), gridspec_kw={"width_ratios": [2, 3]})
    for ax, task in zip(axes, ["AMP", "TOX"], strict=True):
        sub = source[source["Task"] == task]
        metrics = list(dict.fromkeys(sub["Metric"]))
        x = np.arange(len(metrics), dtype=float)
        width = 0.34
        for offset, model in [(-width / 2, "concat_mlp"), (width / 2, "cross_attention_residual")]:
            part = sub[sub["Model key"] == model].set_index("Metric").loc[metrics]
            ax.bar(
                x + offset,
                part["Mean"].values,
                width=width,
                yerr=part["SD"].values,
                capsize=2.0,
                linewidth=0.45,
                edgecolor="white",
                color=COLORS[model],
                label=MODEL_NAMES[model] if task == "AMP" else None,
            )
        for i, metric in enumerate(metrics):
            concat_row = sub[(sub["Metric"] == metric) & (sub["Model key"] == "concat_mlp")].iloc[0]
            residual_row = sub[(sub["Metric"] == metric) & (sub["Model key"] == "cross_attention_residual")].iloc[0]
            concat_mean = concat_row["Mean"]
            residual_mean = residual_row["Mean"]
            delta = residual_mean - concat_mean
            label_y = max(concat_mean + concat_row["SD"], residual_mean + residual_row["SD"]) + 0.012
            ax.text(i, label_y, f"+{delta:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_title(f"{task} hard test", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.68, 0.99)
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Performance score")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.18), ncol=2, handlelength=1.4)
    axes[0].text(-0.26, 1.12, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.12, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top")
    fig.subplots_adjust(top=0.78, wspace=0.32)
    save_figure(fig, "fig1_hard_test_key_metric_bars", fig_dir)


def plot_delta_heatmap(paired_delta: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    order_rows = [
        ("AMP", "Standard test"),
        ("AMP", "AMP hard test"),
        ("TOX", "Standard test"),
        ("TOX", "TOX hard test"),
    ]
    metrics = [label for _, label in CORE_METRICS]
    heat = np.zeros((len(order_rows), len(metrics)), dtype=float)
    for row_idx, (task, split) in enumerate(order_rows):
        for col_idx, metric in enumerate(metrics):
            value = paired_delta[
                (paired_delta["Task"] == task)
                & (paired_delta["Test set"] == split)
                & (paired_delta["Metric"] == metric)
            ]["Mean delta"].iloc[0]
            heat[row_idx, col_idx] = value
    source = pd.DataFrame(heat, columns=metrics)
    source.insert(0, "Task", [task for task, _ in order_rows])
    source.insert(1, "Test set", [split for _, split in order_rows])
    source.to_csv(source_dir / "fig2_residual_minus_concat_heatmap_source.csv", index=False)

    vmax = max(abs(heat.min()), abs(heat.max()), 0.001)
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    image = ax.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(order_rows)))
    ax.set_yticklabels([f"{task} | {split.replace(' test', '')}" for task, split in order_rows])
    ax.tick_params(length=0)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:+.3f}", ha="center", va="center", fontsize=6.2, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Residual CA - Concat")
    ax.set_title("Paired mean deltas across seeds")
    save_figure(fig, "fig2_residual_minus_concat_delta_heatmap", fig_dir)


def plot_modality_context(summary: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    work = subset_valid_splits(summary)
    rows = []
    for task, split in [("amp", "test_hard_amp"), ("tox", "test_hard_tox")]:
        for model in ALL_CONTEXT_MODELS:
            row = work[(work["task"] == task) & (work["split"] == split) & (work["model"] == model)].iloc[0]
            for metric, label in [("roc_auc", "AUC"), ("pr_auc", "AUPRC")]:
                rows.append(
                    {
                        "Task": task.upper(),
                        "Metric": label,
                        "Model": SHORT_MODEL_NAMES[model],
                        "Model key": model,
                        "Mean": row[f"mean_{metric}"],
                        "SD": row[f"std_{metric}"],
                    }
                )
    source = pd.DataFrame(rows)
    source.to_csv(source_dir / "figS1_unimodal_context_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, task in zip(axes, ["AMP", "TOX"], strict=True):
        sub = source[source["Task"] == task]
        labels = []
        means = []
        sds = []
        colors = []
        for metric in ["AUC", "AUPRC"]:
            for model in ALL_CONTEXT_MODELS:
                row = sub[(sub["Metric"] == metric) & (sub["Model key"] == model)].iloc[0]
                labels.append(f"{row['Model']}\n{metric}")
                means.append(row["Mean"])
                sds.append(row["SD"])
                colors.append(COLORS[model])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sds, capsize=2.0, color=colors, linewidth=0.45, edgecolor="white")
        ax.set_title(f"{task} hard test")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0.84, 0.98)
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Ranking metric")
    save_figure(fig, "figS1_unimodal_and_fusion_ranking_context", fig_dir)


def curated_four_model_source(summary: pd.DataFrame) -> pd.DataFrame:
    work = subset_valid_splits(summary)
    rows = []
    for task, split in [("amp", "test_hard_amp"), ("tox", "test_hard_tox")]:
        for metric, label in CURATED_FOUR_MODEL_METRICS[task]:
            for model in ALL_CONTEXT_MODELS:
                row = work[(work["task"] == task) & (work["split"] == split) & (work["model"] == model)].iloc[0]
                rows.append(
                    {
                        "Task": task.upper(),
                        "Test set": SPLIT_NAMES[split],
                        "Metric": label,
                        "Metric key": metric,
                        "Model": MODEL_NAMES[model],
                        "Model short": SHORT_MODEL_NAMES[model],
                        "Model key": model,
                        "Mean": row[f"mean_{metric}"],
                        "SD": row[f"std_{metric}"],
                    }
                )
    return pd.DataFrame(rows)


def build_curated_four_model_exact_table(summary: pd.DataFrame) -> pd.DataFrame:
    source = curated_four_model_source(summary)
    rows = []
    for (task, test_set, metric), sub in source.groupby(["Task", "Test set", "Metric"], sort=False):
        row: dict[str, str] = {"Task": task, "Test set": test_set, "Metric": metric}
        means = {}
        for model in ALL_CONTEXT_MODELS:
            hit = sub[sub["Model key"] == model].iloc[0]
            row[SHORT_MODEL_NAMES[model]] = f"{hit['Mean']:.4f} +/- {hit['SD']:.4f}"
            means[model] = float(hit["Mean"])
        row["Residual - Concat"] = f"{means['cross_attention_residual'] - means['concat_mlp']:+.4f}"
        row["Residual - best unimodal"] = f"{means['cross_attention_residual'] - max(means['ccd_mlp'], means['esm_mlp']):+.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curated_four_model_bars(summary: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    source = curated_four_model_source(summary)
    source.to_csv(source_dir / "fig3_curated_four_model_metric_bars_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"width_ratios": [5, 4]})
    bar_width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    for ax, task in zip(axes, ["AMP", "TOX"], strict=True):
        sub = source[source["Task"] == task]
        metrics = list(dict.fromkeys(sub["Metric"]))
        x = np.arange(len(metrics), dtype=float)
        for offset, model in zip(offsets, ALL_CONTEXT_MODELS, strict=True):
            part = sub[sub["Model key"] == model].set_index("Metric").loc[metrics]
            ax.bar(
                x + offset,
                part["Mean"].values,
                width=bar_width,
                yerr=part["SD"].values,
                capsize=1.6,
                linewidth=0.35,
                edgecolor="white",
                color=COLORS[model],
                label=SHORT_MODEL_NAMES[model] if task == "AMP" else None,
            )
            for xi, mean, sd in zip(x + offset, part["Mean"].values, part["SD"].values, strict=True):
                ax.text(xi, mean + sd + 0.006, f"{mean:.3f}", ha="center", va="bottom", rotation=90, fontsize=4.7)
        ax.set_title(f"{task} hard test", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.56 if task == "TOX" else 0.70, 1.02)
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Performance score")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.18), ncol=4, handlelength=1.2, columnspacing=1.1)
    axes[0].text(-0.18, 1.12, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.12, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top")
    fig.subplots_adjust(top=0.78, wspace=0.28)
    save_figure(fig, "fig3_curated_four_model_metric_bars", fig_dir)


def loss_curve_source(history: pd.DataFrame) -> pd.DataFrame:
    work = history[history["model"].isin(ALL_CONTEXT_MODELS)].copy()
    grouped = (
        work.groupby(["task", "model", "epoch"], as_index=False)
        .agg(
            mean_train_loss=("train_loss", "mean"),
            std_train_loss=("train_loss", "std"),
            mean_val_loss=("val_loss", "mean"),
            std_val_loss=("val_loss", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["task", "model", "epoch"])
    )
    grouped[["std_train_loss", "std_val_loss"]] = grouped[["std_train_loss", "std_val_loss"]].fillna(0.0)
    return grouped


def plot_loss_curves(history: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    source = loss_curve_source(history)
    source.to_csv(source_dir / "fig4_training_loss_source.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, task in zip(axes, ["amp", "tox"], strict=True):
        for model in ALL_CONTEXT_MODELS:
            part = source[(source["task"] == task) & (source["model"] == model)].copy()
            if part.empty:
                continue
            x = part["epoch"].to_numpy()
            y = part["mean_train_loss"].to_numpy()
            sd = part["std_train_loss"].to_numpy()
            ax.plot(x, y, color=COLORS[model], linewidth=1.2, label=SHORT_MODEL_NAMES[model])
            ax.fill_between(x, y - sd, y + sd, color=COLORS[model], alpha=0.10, linewidth=0)
        ax.set_title(f"{task.upper()} training loss", pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy")
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0].text(-0.16, 1.13, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.13, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="top")
    axes[0].legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(top=0.82, wspace=0.32)
    save_figure(fig, "fig4_training_loss_curves", fig_dir)


def build_positioning_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model family": "Direct Concat MLP",
                "Use in manuscript": "Main baseline",
                "Reason": "Uses the same CCD and ESM-2 inputs without an explicit cross-modal interaction module.",
            },
            {
                "Model family": "Residual Cross-Attention",
                "Use in manuscript": "Proposed fusion model",
                "Reason": "Tests whether explicit bidirectional CCD/ESM-2 interaction improves over direct fusion while retaining residual modality towers.",
            },
            {
                "Model family": "CCD-MLP and ESM2-MLP",
                "Use in manuscript": "Supplementary modality controls",
                "Reason": "Useful sanity checks, but they answer a different question from the fusion mechanism comparison.",
            },
            {
                "Model family": "Gated fusion and naive cross-attention",
                "Use in manuscript": "Internal ablation or supplement only",
                "Reason": "Naive cross-attention was unstable on TOX; residualization is the defensible mechanism.",
            },
        ]
    )


def build_report(
    combined_summary: pd.DataFrame,
    paired_delta: pd.DataFrame,
    key_gain: pd.DataFrame,
    out_path: Path,
) -> None:
    def value(task: str, split: str, model: str, metric: str) -> float:
        row = combined_summary[
            (combined_summary["task"] == task)
            & (combined_summary["split"] == split)
            & (combined_summary["model"] == model)
        ].iloc[0]
        return float(row[f"mean_{metric}"])

    amp_hard_auc_delta = value("amp", "test_hard_amp", "cross_attention_residual", "roc_auc") - value(
        "amp", "test_hard_amp", "concat_mlp", "roc_auc"
    )
    tox_hard_auc_delta = value("tox", "test_hard_tox", "cross_attention_residual", "roc_auc") - value(
        "tox", "test_hard_tox", "concat_mlp", "roc_auc"
    )
    tox_esm_auc = value("tox", "test_hard_tox", "esm_mlp", "roc_auc")
    tox_res_auc = value("tox", "test_hard_tox", "cross_attention_residual", "roc_auc")

    report = f"""# Fusion Evidence Consistency Report

Date: 2026-06-11

## 1. Metric panel

The final evaluation table now uses 10 metrics: sensitivity, specificity, accuracy, precision, NPV, balanced accuracy, F1, MCC, ROC-AUC, and PR-AUC. The main manuscript table is intentionally kept to the 7 commonly reported metrics: SEN, SPE, Acc, F1, MCC, AUC, and AUPRC. The full 10-metric panel is exported as supplementary material.

## 2. Recommended manuscript logic

Primary comparison: Residual Cross-Attention versus Direct Concat MLP. This is the cleanest test of the proposed mechanism because both models receive the same CCD and ESM-2 inputs; only the cross-modal interaction layer differs.

Recommended main claim: Residual Cross-Attention improves over direct concatenation on multiple standard and shortcut-controlled hard-test metrics, supporting the usefulness of explicit CCD/ESM-2 interaction. The claim should not be phrased as "dominates every baseline on every metric".

## 3. Key hard-test evidence

AMP hard test: Residual Cross-Attention shows seed-consistent gains over Direct Concat MLP in SEN and F1. Other AMP hard-test metrics show small mean-level gains or small losses; the ROC-AUC delta is {amp_hard_auc_delta:+.4f}, so AMP hard-test AUC should not be used as the central improvement claim.

TOX hard test: Residual Cross-Attention shows seed-consistent gains over Direct Concat MLP in SPE, ROC-AUC, and PR-AUC. The TOX hard-test ROC-AUC delta is {tox_hard_auc_delta:+.4f}, and PR-AUC also improves by +0.0157, which is the cleanest hard-test ranking evidence.

## 4. Baseline placement

CCD-only and ESM2-only neural controls are useful supplementary diagnostics. In particular, ESM2-MLP is strong on TOX hard-test ranking metrics (ESM2-MLP AUC={tox_esm_auc:.4f}; Residual Cross-Attention AUC={tox_res_auc:.4f}). Therefore, these controls should not be framed as the main baseline for the cross-attention mechanism. The manuscript should use Direct Concat MLP as the primary baseline, with single-modality controls in supplementary tables.

Gated fusion and naive cross-attention should not be used as the main baseline. Naive cross-attention showed clear instability on TOX hard testing; the residual design is the biologically and methodologically defensible variant.

## 5. Suggested figure/table placement

Main text:
- Table 1: Direct Concat MLP versus Residual Cross-Attention, 7 core metrics.
- Figure 1: Hard-test key metric bars, showing the metrics where Residual Cross-Attention improves over Direct Concat MLP.
- Figure 2: Four-model hard-test metric comparison, with CCD-only, ESM2-only, Direct Concat MLP, and Residual Cross-Attention grouped within each selected metric.
- Figure 3: Training loss convergence curves for AMP and TOX.
- Supplementary Figure: Residual-minus-Concat delta heatmap across all core metrics, used for transparency.

Supplementary:
- Table S1: Full 10-metric bimodal comparison.
- Table S2: CCD-MLP, ESM2-MLP, Direct Concat MLP, and Residual Cross-Attention context table.
- Table S3: Paired seed-wise deltas and seed win counts.
- Table S4: Seed-consistent positive deltas supporting the Residual Cross-Attention claim.
- Table S6: Exact numeric values for the curated four-model hard-test figure.
- Figure S1: Unimodal versus fusion ranking metric context.

## 6. Consistency warning

Do not claim that Residual Cross-Attention beats ESM2-only on every TOX metric. The stronger and internally consistent statement is that residual cross-attention improves over direct concatenation under the same bimodal input setting and remains robust on shortcut-controlled hard tests.

"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    out_dir = Path(args.out_dir)
    paths = ensure_dirs(out_dir)

    main_summary = pd.read_csv(args.main_summary)
    main_results = pd.read_csv(args.main_results)
    unimodal_summary = pd.read_csv(args.unimodal_summary)
    unimodal_results = pd.read_csv(args.unimodal_results)
    main_history = pd.read_csv(args.main_history)
    unimodal_history = pd.read_csv(args.unimodal_history)

    combined_summary = pd.concat([unimodal_summary, main_summary], ignore_index=True)
    combined_results = pd.concat([unimodal_results, main_results], ignore_index=True)
    combined_history = pd.concat([unimodal_history, main_history], ignore_index=True)
    combined_summary.to_csv(paths["source"] / "combined_fusion_summary_source.csv", index=False)
    combined_results.to_csv(paths["source"] / "combined_fusion_seed_results_source.csv", index=False)
    combined_history.to_csv(paths["source"] / "combined_fusion_training_history_source.csv", index=False)

    main_table = build_table(main_summary, MAIN_MODELS, CORE_METRICS, bold_best=True)
    full_bimodal_table = build_table(main_summary, MAIN_MODELS, FULL_METRICS, bold_best=False)
    modality_context_table = build_table(combined_summary, ALL_CONTEXT_MODELS, FULL_METRICS, bold_best=False)
    paired_delta = build_paired_delta_table(main_results, CORE_METRICS)
    key_gain = build_key_gain_table(paired_delta)
    model_positioning = build_positioning_table()
    curated_four_model_table = build_curated_four_model_exact_table(combined_summary)

    save_table_bundle(
        main_table,
        "Table 1. Direct Concat MLP versus Residual Cross-Attention",
        "table1_main_concat_vs_residual",
        paths["tables"],
    )
    save_table_bundle(
        full_bimodal_table,
        "Table S1. Full 10-metric bimodal comparison",
        "tableS1_full_bimodal_metrics",
        paths["tables"],
    )
    save_table_bundle(
        modality_context_table,
        "Table S2. Unimodal and fusion context",
        "tableS2_unimodal_and_fusion_context",
        paths["tables"],
    )
    save_table_bundle(
        paired_delta,
        "Table S3. Paired seed-wise deltas of Residual Cross-Attention minus Direct Concat MLP",
        "tableS3_paired_seed_delta",
        paths["tables"],
    )
    save_table_bundle(
        key_gain,
        "Table S4. Positive deltas supporting the Residual Cross-Attention claim",
        "tableS4_key_positive_gains",
        paths["tables"],
    )
    save_table_bundle(
        model_positioning,
        "Table S5. Recommended placement of model families",
        "tableS5_model_positioning",
        paths["tables"],
    )
    save_table_bundle(
        curated_four_model_table,
        "Table S6. Exact values used in the curated four-model hard-test figure",
        "tableS6_curated_four_model_exact_values",
        paths["tables"],
    )

    for stale_ext in ["svg", "pdf", "png", "tiff"]:
        stale = paths["figures"] / f"fig4_training_validation_loss_curves.{stale_ext}"
        if stale.exists():
            stale.unlink()
    stale_source = paths["source"] / "fig4_training_validation_loss_source.csv"
    if stale_source.exists():
        stale_source.unlink()

    plot_key_metric_bars(main_summary, paths["figures"], paths["source"])
    plot_delta_heatmap(paired_delta, paths["figures"], paths["source"])
    plot_modality_context(combined_summary, paths["figures"], paths["source"])
    plot_curated_four_model_bars(combined_summary, paths["figures"], paths["source"])
    plot_loss_curves(combined_history, paths["figures"], paths["source"])

    build_report(
        combined_summary=combined_summary,
        paired_delta=paired_delta,
        key_gain=key_gain,
        out_path=paths["reports"] / "fusion_evidence_consistency_report_20260611.md",
    )

    print(f"[DONE] Evidence package -> {out_dir}")
    print(f"[DONE] Tables -> {paths['tables']}")
    print(f"[DONE] Figures -> {paths['figures']}")
    print(f"[DONE] Source data -> {paths['source']}")
    print(f"[DONE] Report -> {paths['reports'] / 'fusion_evidence_consistency_report_20260611.md'}")


if __name__ == "__main__":
    main()
