from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOTA_DIR = ROOT / "data--final" / "sota_duel"
OUT_DIR = ROOT / "data--final" / "manuscript_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DUAL_CSV = SOTA_DIR / "tables" / "sota_dual_screening_metrics.csv"
DELTA_CSV = SOTA_DIR / "tables" / "sota_duel_best_sota_delta.csv"
AMP_CSV = SOTA_DIR / "tables" / "sota_amp_hard_task_metrics.csv"
TOX_CSV = SOTA_DIR / "tables" / "sota_tox_hard_task_metrics.csv"
BIAS_CSV = SOTA_DIR / "tables" / "table_sota_test_mc_bias_summary.csv"
RUNNABILITY_CSV = SOTA_DIR / "tables" / "sota_candidate_runnability_audit.csv"

FIG_BASE = OUT_DIR / "resultC_fig_sota_paired_duel"
SOURCE_CSV = OUT_DIR / "resultC_fig_sota_paired_duel_source_data.csv"
TABLE_CSV = OUT_DIR / "resultC_table_paired_external_sota_duel.csv"
TABLE_MD = OUT_DIR / "resultC_table_paired_external_sota_duel.md"
REPORT_MD = OUT_DIR / "resultC_sota_duel_review_report_cn.md"

MODELS = [
    "AMPlify-balanced + ToxinPred3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid",
    "AMPlify-imbalanced + ToxinPred3-Hybrid",
    "Residual Cross-Attention",
]
MODEL_LABELS = {
    "AMPlify-balanced + ToxinPred3-ML": "AMPlify-B\n+ TP3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML": "AMPlify-I\n+ TP3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid": "AMPlify-B\n+ TP3-H",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "AMPlify-I\n+ TP3-H",
    "Residual Cross-Attention": "Residual\nCA",
}
COLORS = {
    "AMPlify-balanced + ToxinPred3-ML": "#E6B5B3",
    "AMPlify-imbalanced + ToxinPred3-ML": "#F0D3D2",
    "AMPlify-balanced + ToxinPred3-Hybrid": "#CFCFCF",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "#B7C5D6",
    "Residual Cross-Attention": "#0A0A8C",
}
METRICS = [
    ("macro_roc_auc", "Macro\nROC-AUC"),
    ("macro_pr_auc", "Macro\nPR-AUC"),
    ("macro_f1", "Macro\nF1"),
    ("macro_mcc", "Macro\nMCC"),
    ("worst_task_roc_auc", "Worst-task\nROC-AUC"),
    ("worst_task_pr_auc", "Worst-task\nPR-AUC"),
]
EPS = 5e-4


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
            "figure.dpi": 160,
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#DDE3EA", lw=0.5, ls="-", alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3, color="#20242A", labelcolor="#20242A")
    ax.spines["left"].set_color("#20242A")
    ax.spines["bottom"].set_color("#20242A")


def save_all(fig: plt.Figure, base: Path) -> None:
    fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_source(dual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, label in METRICS:
        for model in MODELS:
            rec = dual[dual["model"] == model]
            if rec.empty:
                raise ValueError(f"Missing model row: {model}")
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label.replace("\n", " "),
                    "model": model,
                    "plot_label": MODEL_LABELS[model].replace("\n", " "),
                    "value": float(rec.iloc[0][metric]),
                    "is_proposed": model == "Residual Cross-Attention",
                }
            )
    return pd.DataFrame(rows)


def make_figure(source: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.55), sharey=False)
    letters = list("ABCDEF")
    for idx, ((metric, label), ax) in enumerate(zip(METRICS, axes.flat, strict=True)):
        sub = source[source["metric"] == metric].set_index("model").loc[MODELS].reset_index()
        x = np.arange(len(MODELS))
        values = sub["value"].to_numpy()
        best = values.max()
        edges = ["#111111" if v >= best - EPS else "white" for v in values]
        lw = [1.0 if v >= best - EPS else 0.35 for v in values]
        ax.bar(
            x,
            values,
            color=[COLORS[m] for m in MODELS],
            edgecolor=edges,
            linewidth=lw,
            zorder=3,
        )
        for xi, val in zip(x, values, strict=True):
            ax.text(xi, val + 0.012, f"{val:.4f}", ha="center", va="bottom", fontsize=5.3, rotation=90)
        ax.set_ylim(0.50 if metric != "macro_mcc" else 0.35, 1.02)
        ax.set_title(label, pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=35, ha="right", linespacing=0.9)
        ax.set_ylabel("Score" if idx in [0, 3] else "")
        clean_axis(ax)
        ax.text(-0.18, 1.14, letters[idx], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    handles = [mpl.patches.Patch(color=COLORS[m], label=MODEL_LABELS[m].replace("\n", " ")) for m in MODELS]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="#111111", linewidth=1.0, label="best/tie-best"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=6,
        frameon=False,
        columnspacing=0.72,
        handlelength=1.0,
    )
    fig.subplots_adjust(wspace=0.30, hspace=0.64)
    save_all(fig, FIG_BASE)


def write_tables_and_report(dual: pd.DataFrame, source: pd.DataFrame) -> None:
    table_cols = [
        "model",
        "macro_roc_auc",
        "macro_pr_auc",
        "macro_f1",
        "macro_mcc",
        "worst_task_roc_auc",
        "worst_task_pr_auc",
    ]
    table = dual.set_index("model").loc[MODELS].reset_index()[table_cols]
    table.to_csv(TABLE_CSV, index=False)
    TABLE_MD.write_text(table.round(4).to_markdown(index=False), encoding="utf-8")

    delta = pd.read_csv(DELTA_CSV)
    amp = pd.read_csv(AMP_CSV)
    tox = pd.read_csv(TOX_CSV)
    bias = pd.read_csv(BIAS_CSV)
    runnable = pd.read_csv(RUNNABILITY_CSV)
    source.to_csv(SOURCE_CSV, index=False)

    def val(df: pd.DataFrame, row: str, col: str, key: str = "model") -> float:
        return float(df[df[key] == row][col].iloc[0])

    report = f"""# Result C Paired External SOTA Duel Audit

## 审查结论

GPT 初稿的总体方向可用，但需要三处修正：

1. 比较对象应写为“paired external screening pipelines”，而不是与外部“端到端双任务模型”直接同构比较。AMPlify 和 ToxinPred3 是两个单任务工具，我们把它们按候选筛选流程配对，是实践场景比较，不是架构同类对决。
2. 主指标建议使用 Macro ROC-AUC、Macro PR-AUC、Macro F1、Macro MCC、Worst-task ROC-AUC 和 Worst-task PR-AUC。现有 Acc/SEN/SPE/MCC 四图不够贴合双任务排序筛选故事。
3. “单任务也全面支持”这句话要小心。AMP 单任务 Residual CA 确实高于 AMPlify 两版本的 sensitivity/F1/MCC/ROC-AUC/PR-AUC；TOX 单任务中 ToxinPred3-ML 的 sensitivity 高于 Residual CA，因此 TOX 只能写 Residual CA 在 specificity、precision、accuracy、F1、MCC、ROC-AUC 和 PR-AUC 上更好，不能写全部指标都更好。

## 关键数值

- Residual CA macro ROC-AUC: {val(dual, "Residual Cross-Attention", "macro_roc_auc"):.4f}
- Residual CA macro PR-AUC: {val(dual, "Residual Cross-Attention", "macro_pr_auc"):.4f}
- Residual CA macro F1: {val(dual, "Residual Cross-Attention", "macro_f1"):.4f}
- Residual CA macro MCC: {val(dual, "Residual Cross-Attention", "macro_mcc"):.4f}
- Residual CA worst-task ROC-AUC: {val(dual, "Residual Cross-Attention", "worst_task_roc_auc"):.4f}
- Residual CA worst-task PR-AUC: {val(dual, "Residual Cross-Attention", "worst_task_pr_auc"):.4f}

Best external paired baseline deltas:

{delta.round(4).to_markdown(index=False)}

## 单任务边界

AMP hard test:

{amp.round(4).to_markdown(index=False)}

TOX hard test:

{tox.round(4).to_markdown(index=False)}

## 外部 benchmark shortcut bias

{bias.round(4).to_markdown(index=False)}

## Runnability boundary

{runnable.to_markdown(index=False)}

## 推荐正文写法

这一节应写成“统一 hard-test 上的可复现 paired SOTA duel”。核心论证不是外部工具差，而是外部工具原本被设计为单任务 predictor；在实际 AMP 设计中，AMP activity 与 toxicity risk 必须同时满足，因此把可运行 AMP predictor 与 TOX predictor 配对成 screening pipeline 是合理的工程比较。Residual CA 在同一 shortcut-controlled hard-test 条件下取得更高 macro ranking、macro F1/MCC 与 worst-task ranking，说明其双任务筛选稳定性优于简单串联外部单任务工具。

## 输出

- `{FIG_BASE.relative_to(ROOT).as_posix()}.svg/.png/.pdf`
- `{SOURCE_CSV.relative_to(ROOT).as_posix()}`
- `{TABLE_CSV.relative_to(ROOT).as_posix()}`
- `{TABLE_MD.relative_to(ROOT).as_posix()}`
- `{REPORT_MD.relative_to(ROOT).as_posix()}`
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    setup_style()
    dual = pd.read_csv(DUAL_CSV)
    source = make_source(dual)
    make_figure(source)
    write_tables_and_report(dual, source)
    print(f"Wrote {FIG_BASE}.svg/.png/.pdf")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
