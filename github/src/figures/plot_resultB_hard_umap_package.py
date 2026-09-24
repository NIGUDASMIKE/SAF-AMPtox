from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data--final" / "manuscript_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUSION_DIR = ROOT / "data--final" / "fusion_evidence_package"
UMAP_DIR = ROOT / "data--final" / "fusion_umap_hard"

SUMMARY_CSV = FUSION_DIR / "source_data" / "combined_fusion_summary_source.csv"
LOSS_CSV = FUSION_DIR / "source_data" / "fig4_training_loss_source.csv"
UMAP_COORD_CSV = UMAP_DIR / "source_data" / "umap_embedding_2d_source.csv"
UMAP_SUMMARY_CSV = UMAP_DIR / "tables" / "umap_separation_metrics_summary.csv"

FIG1_BASE = OUT_DIR / "resultB_fig1_hard_test_ablation_convergence"
FIG2_BASE = OUT_DIR / "resultB_fig2_umap_latent_representations"
FIG3_BASE = OUT_DIR / "resultB_fig3_umap_geometry_metrics"

FIG1_SOURCE = OUT_DIR / "resultB_fig1_hard_test_ablation_convergence_source_data.csv"
FIG1_LOSS_SOURCE = OUT_DIR / "resultB_fig1_training_loss_source_data.csv"
FIG2_SOURCE = OUT_DIR / "resultB_fig2_umap_coordinates_source_data.csv"
FIG3_SOURCE = OUT_DIR / "resultB_fig3_umap_geometry_metrics_source_data.csv"
TABLE_CSV = OUT_DIR / "resultB_model_positioning_table.csv"
TABLE_MD = OUT_DIR / "resultB_model_positioning_table.md"
TABLE_TEX = OUT_DIR / "resultB_model_positioning_table.tex"
REPORT_MD = OUT_DIR / "resultB_hard_umap_review_report_cn.md"

MODEL_ORDER = ["ccd_mlp", "esm_mlp", "concat_mlp", "cross_attention_residual"]
MODEL_LABEL = {
    "ccd_mlp": "CCD-only",
    "esm_mlp": "ESM-2-only",
    "concat_mlp": "Direct Concat",
    "cross_attention_residual": "Residual CA",
}
MODEL_SHORT = {
    "ccd_mlp": "CCD",
    "esm_mlp": "ESM-2",
    "concat_mlp": "Concat",
    "cross_attention_residual": "Residual CA",
}
MODEL_COLORS = {
    "ccd_mlp": "#8FA3D1",
    "esm_mlp": "#63BFA5",
    "concat_mlp": "#82919E",
    "cross_attention_residual": "#C93F3A",
}
CLASS_COLORS = {
    "AMP": "#E95C4A",
    "Non-AMP": "#55BFD3",
    "Toxic": "#E95C4A",
    "Non-toxic": "#55BFD3",
}
METRIC_ORDER = ["SEN", "SPE", "Acc", "F1", "MCC", "ROC-AUC", "PR-AUC"]
METRIC_DISPLAY = {
    "SEN": "SEN",
    "SPE": "SPE",
    "Acc": "Acc",
    "F1": "F1",
    "MCC": "MCC",
    "ROC-AUC": "ROC\nAUC",
    "PR-AUC": "PR\nAUC",
}
MEAN_COL = {
    "SEN": "mean_sensitivity",
    "SPE": "mean_specificity",
    "Acc": "mean_accuracy",
    "F1": "mean_f1",
    "MCC": "mean_mcc",
    "ROC-AUC": "mean_roc_auc",
    "PR-AUC": "mean_pr_auc",
}
STD_COL = {
    "SEN": "std_sensitivity",
    "SPE": "std_specificity",
    "Acc": "std_accuracy",
    "F1": "std_f1",
    "MCC": "std_mcc",
    "ROC-AUC": "std_roc_auc",
    "PR-AUC": "std_pr_auc",
}
BEST_EPS = 5e-4


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
            "axes.linewidth": 0.75,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 160,
        }
    )


def save_all(fig: plt.Figure, base: Path) -> None:
    fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#DDE4EA", lw=0.5, ls="-", alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3, color="#20242A", labelcolor="#20242A")
    ax.spines["left"].set_color("#20242A")
    ax.spines["bottom"].set_color("#20242A")


def make_metric_source(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, split in [("AMP", "test_hard_amp"), ("TOX", "test_hard_tox")]:
        for model in MODEL_ORDER:
            hit = summary[(summary["task"] == task.lower()) & (summary["model"] == model) & (summary["split"] == split)]
            if hit.empty:
                raise ValueError(f"Missing summary row for {task} {split} {model}")
            rec = hit.iloc[0]
            for metric in METRIC_ORDER:
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "model": model,
                        "model_label": MODEL_LABEL[model],
                        "model_short": MODEL_SHORT[model],
                        "metric": metric,
                        "mean": float(rec[MEAN_COL[metric]]),
                        "sd": float(rec[STD_COL[metric]]),
                        "n_seeds": int(rec["n_seeds"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_panel(ax: plt.Axes, source: pd.DataFrame, task: str, title: str) -> None:
    panel = source[source["task"] == task].copy()
    x = np.arange(len(METRIC_ORDER))
    width = 0.17
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))
    best_by_metric = panel.groupby("metric")["mean"].max()
    concat = panel[panel["model"] == "concat_mlp"].set_index("metric")["mean"]
    residual = panel[panel["model"] == "cross_attention_residual"].set_index("metric")["mean"]

    for offset, model in zip(offsets, MODEL_ORDER, strict=True):
        rows = panel[panel["model"] == model].set_index("metric").loc[METRIC_ORDER].reset_index()
        values = rows["mean"].to_numpy()
        sds = rows["sd"].to_numpy()
        edge_colors = []
        line_widths = []
        for metric, value in zip(rows["metric"], values, strict=True):
            is_best = value >= best_by_metric.loc[metric] - BEST_EPS
            edge_colors.append("#111111" if is_best else "white")
            line_widths.append(0.9 if is_best else 0.35)
        ax.bar(
            x + offset,
            values,
            width,
            color=MODEL_COLORS[model],
            edgecolor=edge_colors,
            linewidth=line_widths,
            zorder=3,
            label=MODEL_SHORT[model],
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=sds,
            fmt="none",
            ecolor="#33383E",
            elinewidth=0.45,
            capsize=1.2,
            capthick=0.45,
            alpha=0.75,
            zorder=4,
        )
        for xi, value, sd in zip(x + offset, values, sds, strict=True):
            ax.text(
                xi,
                value + sd + 0.011,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=4.7,
                rotation=90,
                color="#20242A",
                clip_on=False,
            )

    for idx, metric in enumerate(METRIC_ORDER):
        delta = (float(residual.loc[metric]) - float(concat.loc[metric])) * 100.0
        color = "#B62F2F" if delta >= 0 else "#6E7880"
        ax.text(
            idx,
            1.035,
            f"{delta:+.2f} pp",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=color,
            clip_on=False,
        )

    ax.set_title(title, pad=8)
    ax.set_ylabel("Performance score")
    ax.set_ylim(0.55, 1.065)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_ORDER], rotation=0, linespacing=0.9)
    clean_axis(ax)


def plot_loss_panel(ax: plt.Axes, loss: pd.DataFrame, task: str, title: str, ylim: tuple[float, float]) -> None:
    panel = loss[loss["task"] == task].copy()
    for model in MODEL_ORDER:
        rows = panel[panel["model"] == model].sort_values("epoch")
        if rows.empty:
            continue
        x = rows["epoch"].to_numpy()
        y = rows["mean_train_loss"].to_numpy()
        sd = rows["std_train_loss"].fillna(0).to_numpy()
        ax.plot(x, y, color=MODEL_COLORS[model], lw=1.25, label=MODEL_SHORT[model])
        ax.fill_between(x, y - sd, y + sd, color=MODEL_COLORS[model], alpha=0.08, linewidth=0)
    ax.set_title(title, pad=5)
    ax.set_xlabel("Epoch", labelpad=1)
    ax.set_ylabel("BCE loss", labelpad=1)
    ax.set_ylim(*ylim)
    clean_axis(ax)


def make_fig1(metric_source: pd.DataFrame, loss: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9.3, 3.8))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.62, 1.62, 0.88],
        height_ratios=[1, 1],
        wspace=0.34,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])

    plot_metric_panel(ax_a, metric_source, "AMP", "AMP hard test")
    plot_metric_panel(ax_b, metric_source, "TOX", "TOX hard test")
    plot_loss_panel(ax_c, loss, "amp", "AMP training loss", (0.0, 0.31))
    plot_loss_panel(ax_d, loss, "tox", "TOX training loss", (0.0, 0.52))

    handles = [mpl.patches.Patch(color=MODEL_COLORS[m], label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="#111111", linewidth=0.9, label="best/tie-best"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.025),
        ncol=5,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.1,
    )

    for label, ax, dx, dy in [
        ("A", ax_a, -0.12, 1.055),
        ("B", ax_b, -0.12, 1.055),
        ("C", ax_c, -0.18, 1.12),
        ("D", ax_d, -0.18, 1.12),
    ]:
        ax.text(dx, dy, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")

    save_all(fig, FIG1_BASE)
    metric_source.to_csv(FIG1_SOURCE, index=False)
    loss.to_csv(FIG1_LOSS_SOURCE, index=False)


def make_fig2() -> None:
    coords = pd.read_csv(UMAP_COORD_CSV)
    label_map = {
        "CCD-MLP": "CCD-only",
        "ESM2-MLP": "ESM-2-only",
        "Direct Concat": "Direct Concat",
        "Residual CA": "Residual CA",
    }
    coords["plot_model_label"] = coords["model_label"].map(label_map).fillna(coords["model_label"])
    coords.to_csv(FIG2_SOURCE, index=False)

    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.75))
    panel_specs = [
        ("amp", "ccd_mlp", "AMP | CCD-only"),
        ("amp", "esm_mlp", "AMP | ESM-2-only"),
        ("amp", "concat_mlp", "AMP | Direct Concat"),
        ("amp", "cross_attention_residual", "AMP | Residual CA"),
        ("tox", "ccd_mlp", "TOX | CCD-only"),
        ("tox", "esm_mlp", "TOX | ESM-2-only"),
        ("tox", "concat_mlp", "TOX | Direct Concat"),
        ("tox", "cross_attention_residual", "TOX | Residual CA"),
    ]
    for idx, (ax, (task, model, title)) in enumerate(zip(axes.flat, panel_specs, strict=True)):
        sub = coords[(coords["task"] == task) & (coords["model"] == model)].copy()
        class_order = ["Non-AMP", "AMP"] if task == "amp" else ["Non-toxic", "Toxic"]
        for cls in class_order:
            cdf = sub[sub["label"] == cls]
            ax.scatter(
                cdf["umap_1"],
                cdf["umap_2"],
                s=8,
                c=CLASS_COLORS[cls],
                alpha=0.72,
                linewidths=0,
                label=cls,
                rasterized=True,
            )
        ax.set_title(title, pad=5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(False)
        clean_axis(ax, grid=False)
        if idx in [0, 4]:
            ax.legend(loc="upper left", frameon=False, fontsize=6, handletextpad=0.3, markerscale=1.0)
        ax.text(-0.22, 1.12, chr(ord("A") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.43, hspace=0.50)
    save_all(fig, FIG2_BASE)


def make_geometry_source(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_defs = [
        ("silhouette", "Silhouette", "higher"),
        ("davies_bouldin", "Davies-Bouldin", "lower"),
        ("fisher_ratio", "Fisher ratio", "higher"),
    ]
    for task in ["amp", "tox"]:
        for metric, label, direction in metric_defs:
            for model in MODEL_ORDER:
                rec = summary[(summary["task"] == task) & (summary["model"] == model)]
                if rec.empty:
                    raise ValueError(f"Missing UMAP metric for {task} {model}")
                hit = rec.iloc[0]
                rows.append(
                    {
                        "task": task.upper(),
                        "model": model,
                        "model_label": MODEL_LABEL[model],
                        "model_short": MODEL_SHORT[model],
                        "metric": metric,
                        "metric_label": label,
                        "direction": direction,
                        "mean": float(hit[f"mean_{metric}"]),
                        "sd": float(hit[f"std_{metric}"]),
                    }
                )
    return pd.DataFrame(rows)


def make_fig3(geom_source: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.25))
    metrics = [
        ("silhouette", "Silhouette", "higher better"),
        ("davies_bouldin", "Davies-Bouldin", "lower better"),
        ("fisher_ratio", "Fisher ratio", "higher better"),
    ]
    letters = list("ABCDEF")
    k = 0
    for row_idx, task in enumerate(["AMP", "TOX"]):
        for col_idx, (metric, title, subtitle) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            sub = geom_source[(geom_source["task"] == task) & (geom_source["metric"] == metric)]
            sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
            values = sub["mean"].to_numpy()
            sds = sub["sd"].to_numpy()
            if sub["direction"].iloc[0] == "lower":
                best_value = values.min()
                is_best = values <= best_value + BEST_EPS
            else:
                best_value = values.max()
                is_best = values >= best_value - BEST_EPS
            x = np.arange(len(MODEL_ORDER))
            ax.bar(
                x,
                values,
                yerr=sds,
                color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                edgecolor=["#111111" if b else "white" for b in is_best],
                linewidth=[0.9 if b else 0.35 for b in is_best],
                capsize=2.0,
                zorder=3,
            )
            upper = max(values + sds) * 1.18
            ax.set_ylim(0, upper)
            y_range = max(values + sds) - min(0, min(values - sds))
            for xi, value, sd in zip(x, values, sds, strict=True):
                ax.text(
                    xi,
                    value + sd + 0.035 * max(y_range, 1.0),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
            ax.set_title(f"{task} | {title}\n{subtitle}", pad=5)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=35, ha="right")
            ax.set_ylabel("Score" if col_idx == 0 else "")
            clean_axis(ax)
            ax.text(-0.20, 1.17, letters[k], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
            k += 1
    fig.subplots_adjust(wspace=0.36, hspace=0.62)
    save_all(fig, FIG3_BASE)
    geom_source.to_csv(FIG3_SOURCE, index=False)


def write_model_positioning_table() -> None:
    rows = [
        {
            "Context": "AMP hard test",
            "Model": "ESM-2-only",
            "Strength": "Strong semantic recall baseline.",
            "Limitation": "Lower Acc/F1/MCC/PR-AUC than Residual CA in the three-seed mean.",
            "Role": "Single-modality semantic reference.",
        },
        {
            "Context": "AMP hard test",
            "Model": "Residual CA",
            "Strength": "Best or near-best Acc, F1, MCC and PR-AUC; improves over Direct Concat on 5/7 metrics.",
            "Limitation": "SPE and ROC-AUC are slightly below Direct Concat; SEN is below ESM-2-only.",
            "Role": "Balanced AMP fusion discriminator.",
        },
        {
            "Context": "TOX hard test",
            "Model": "ESM-2-only",
            "Strength": "Strongest ROC-AUC and PR-AUC in the three-seed mean.",
            "Limitation": "Lower SPE and Precision than Residual CA.",
            "Role": "Strong toxic-ranking baseline.",
        },
        {
            "Context": "TOX hard test",
            "Model": "Residual CA",
            "Strength": "Best SPE and Precision; improves over Direct Concat on SPE, Precision, ROC-AUC and PR-AUC.",
            "Limitation": "Does not exceed ESM-2-only on TOX ROC-AUC/PR-AUC.",
            "Role": "Balanced bimodal toxicity discriminator.",
        },
        {
            "Context": "Fusion ablation",
            "Model": "Direct Concat",
            "Strength": "Clean bimodal baseline using the same CCD and ESM-2 inputs as Residual CA.",
            "Limitation": "Weaker UMAP geometry and lower key hard-test endpoints than Residual CA.",
            "Role": "Primary fusion-mechanism comparator.",
        },
        {
            "Context": "Descriptor baseline",
            "Model": "CCD-only",
            "Strength": "Interpretable physicochemical descriptor baseline.",
            "Limitation": "Generally weaker than ESM-2-only and fusion models.",
            "Role": "Descriptor audit and unimodal reference.",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_CSV, index=False)
    TABLE_MD.write_text(table.to_markdown(index=False), encoding="utf-8")
    TABLE_TEX.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")


def write_report(metric_source: pd.DataFrame, geom_source: pd.DataFrame) -> None:
    def rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    def delta(task: str, metric: str, model_a: str, model_b: str) -> float:
        sub = metric_source[(metric_source["task"] == task) & (metric_source["metric"] == metric)]
        a = float(sub[sub["model"] == model_a]["mean"].iloc[0])
        b = float(sub[sub["model"] == model_b]["mean"].iloc[0])
        return (a - b) * 100.0

    amp_wins = []
    tox_wins = []
    for metric in METRIC_ORDER:
        d_amp = delta("AMP", metric, "cross_attention_residual", "concat_mlp")
        d_tox = delta("TOX", metric, "cross_attention_residual", "concat_mlp")
        if d_amp >= -BEST_EPS * 100:
            amp_wins.append(metric)
        if d_tox >= -BEST_EPS * 100:
            tox_wins.append(metric)

    report = f"""# Result B Hard-Test Ablation and UMAP Package Report

## 审查结论

你的 Result B 文字主线基本成立，但需要保持克制：Residual CA 不能写成全面超过 ESM-2-only。最稳妥的结论是：ESM-2-only 是强语义排序基线；Residual CA 是更均衡的双模态融合模型，尤其相对使用相同输入的 Direct Concat 更有优势，并在 hard-test UMAP 几何上形成最清晰的类别组织。

## 数据来源

- Hard-test performance: `{rel(SUMMARY_CSV)}`
- Training loss: `{rel(LOSS_CSV)}`
- UMAP coordinates: `{rel(UMAP_COORD_CSV)}`
- UMAP geometry metrics: `{rel(UMAP_SUMMARY_CSV)}`

所有 hard-test performance 与 loss 曲线均为三 seed mean +/- SD。UMAP 坐标图使用既有 plot seed 的坐标，UMAP 几何指标使用三 seed mean +/- SD。Best/tie-best 判定阈值为 `{BEST_EPS}`。

## Residual CA vs Direct Concat

AMP hard test 中，Residual CA 相比 Direct Concat 非劣或提升的共享指标为：{", ".join(amp_wins)}。关键提升包括 SEN {delta("AMP", "SEN", "cross_attention_residual", "concat_mlp"):+.2f} pp、Acc {delta("AMP", "Acc", "cross_attention_residual", "concat_mlp"):+.2f} pp、F1 {delta("AMP", "F1", "cross_attention_residual", "concat_mlp"):+.2f} pp、MCC {delta("AMP", "MCC", "cross_attention_residual", "concat_mlp"):+.2f} pp、PR-AUC {delta("AMP", "PR-AUC", "cross_attention_residual", "concat_mlp"):+.2f} pp。

TOX hard test 中，Residual CA 相比 Direct Concat 非劣或提升的共享指标为：{", ".join(tox_wins)}。关键提升包括 SPE {delta("TOX", "SPE", "cross_attention_residual", "concat_mlp"):+.2f} pp、ROC-AUC {delta("TOX", "ROC-AUC", "cross_attention_residual", "concat_mlp"):+.2f} pp、PR-AUC {delta("TOX", "PR-AUC", "cross_attention_residual", "concat_mlp"):+.2f} pp。Precision 不在 Fig. 1 的七共享指标中，但在完整表中 Residual CA 相比 Direct Concat 提高约 +0.62 pp。

## ESM-2-only 的必要保留

TOX hard test 中，ESM-2-only 的 ROC-AUC 和 PR-AUC 高于 Residual CA，分别约为 0.9544 vs 0.9436、0.9578 vs 0.9447。因此正文应写作“ESM-2-only remains a strong semantic ranking baseline”，不要写 Residual CA 在 TOX ranking endpoint 上全面最好。

## UMAP 几何

Residual CA 在 AMP 与 TOX hard test 上均取得最高 silhouette、最低 Davies-Bouldin 和最高 Fisher ratio。该证据适合解释为 latent geometry audit，而不是分类性能的替代指标。

## 推荐 Result B 叙述

1. 先用 Fig. 1 展示七个共享 hard-test 指标和训练收敛，避免选择性指标嫌疑。
2. 承认 ESM-2-only 很强，尤其是 TOX ranking。
3. 把主要消融锚定在 Direct Concat vs Residual CA，因为二者输入完全一致。
4. 用 Fig. 2 和 Fig. 3 说明 Residual CA 的 latent space 更清晰，从而支持后续 SOTA duel 和生成筛选使用 Residual CA 作为主判别器。

## 输出文件

- `{rel(FIG1_BASE)}.svg/.png/.pdf`
- `{rel(FIG2_BASE)}.svg/.png/.pdf`
- `{rel(FIG3_BASE)}.svg/.png/.pdf`
- `{rel(FIG1_SOURCE)}`
- `{rel(FIG1_LOSS_SOURCE)}`
- `{rel(FIG2_SOURCE)}`
- `{rel(FIG3_SOURCE)}`
- `{rel(TABLE_CSV)}`
- `{rel(TABLE_MD)}`
- `{rel(TABLE_TEX)}`
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    setup_style()
    summary = pd.read_csv(SUMMARY_CSV)
    loss = pd.read_csv(LOSS_CSV)
    umap_summary = pd.read_csv(UMAP_SUMMARY_CSV)

    metric_source = make_metric_source(summary)
    geom_source = make_geometry_source(umap_summary)

    make_fig1(metric_source, loss)
    make_fig2()
    make_fig3(geom_source)
    write_model_positioning_table()
    write_report(metric_source, geom_source)

    print(f"Wrote {FIG1_BASE}.svg/.png/.pdf")
    print(f"Wrote {FIG2_BASE}.svg/.png/.pdf")
    print(f"Wrote {FIG3_BASE}.svg/.png/.pdf")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
