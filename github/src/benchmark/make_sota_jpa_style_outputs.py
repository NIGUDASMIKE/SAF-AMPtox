from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data--final" / "sota_duel"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"


MODEL_ORDER = [
    "AMPlify-balanced + ToxinPred3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid",
    "AMPlify-imbalanced + ToxinPred3-Hybrid",
    "Residual Cross-Attention",
]

MODEL_SHORT = {
    "AMPlify-balanced + ToxinPred3-ML": "AMPlify-B\n+ TP3-ML",
    "AMPlify-imbalanced + ToxinPred3-ML": "AMPlify-I\n+ TP3-ML",
    "AMPlify-balanced + ToxinPred3-Hybrid": "AMPlify-B\n+ TP3-H",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "AMPlify-I\n+ TP3-H",
    "Residual Cross-Attention": "Residual\nCA",
}

TABLE_MODEL = {
    "AMPlify-balanced + ToxinPred3-ML": "AMPlify-B + ToxinPred3 (ML)",
    "AMPlify-imbalanced + ToxinPred3-ML": "AMPlify-I + ToxinPred3 (ML)",
    "AMPlify-balanced + ToxinPred3-Hybrid": "AMPlify-B + ToxinPred3 (Hybrid)",
    "AMPlify-imbalanced + ToxinPred3-Hybrid": "AMPlify-I + ToxinPred3 (Hybrid)",
    "Residual Cross-Attention": "Residual Cross-Attention",
}


def style() -> None:
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
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "legend.frameon": False,
        }
    )


def save_pub(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def build_jpa_table() -> pd.DataFrame:
    dual = pd.read_csv(TABLE_DIR / "sota_dual_screening_metrics.csv")
    amp = pd.read_csv(TABLE_DIR / "sota_amp_hard_task_metrics.csv")
    tox = pd.read_csv(TABLE_DIR / "sota_tox_hard_task_metrics.csv")
    amp = amp.set_index("model")
    tox = tox.set_index("model")

    pair_map = {
        "AMPlify-balanced + ToxinPred3-ML": ("AMPlify-balanced", "ToxinPred3-ML"),
        "AMPlify-imbalanced + ToxinPred3-ML": ("AMPlify-imbalanced", "ToxinPred3-ML"),
        "AMPlify-balanced + ToxinPred3-Hybrid": ("AMPlify-balanced", "ToxinPred3-Hybrid"),
        "AMPlify-imbalanced + ToxinPred3-Hybrid": ("AMPlify-imbalanced", "ToxinPred3-Hybrid"),
        "Residual Cross-Attention": ("Residual Cross-Attention", "Residual Cross-Attention"),
    }
    metric_map = {
        "SEN": "sensitivity",
        "SPE": "specificity",
        "Acc": "accuracy",
        "F1": "f1",
        "MCC": "mcc",
        "AUC": "roc_auc",
    }
    rows = []
    for model in MODEL_ORDER:
        amp_model, tox_model = pair_map[model]
        row = {"Model": TABLE_MODEL[model]}
        for out_col, metric in metric_map.items():
            row[out_col] = (float(amp.loc[amp_model, metric]) + float(tox.loc[tox_model, metric])) / 2
        dual_row = dual.loc[dual["model"] == model].iloc[0]
        row["PR-AUC"] = float(dual_row["macro_pr_auc"])
        row["P-value"] = "NA"
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["SEN", "SPE", "Acc", "F1", "MCC", "AUC", "PR-AUC"]:
        out[col] = out[col].astype(float)
    out.to_csv(TABLE_DIR / "table1_jpa_style_sota_duel.csv", index=False)

    rounded = out.copy()
    for col in ["SEN", "SPE", "Acc", "F1", "MCC", "AUC", "PR-AUC"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")
    (TABLE_DIR / "table1_jpa_style_sota_duel.md").write_text(
        "# Table 1\n\n"
        "Performance comparison between the proposed residual cross-attention model and paired SOTA predictors on the balanced hard-test dual-screening benchmark.\n\n"
        + rounded.to_markdown(index=False)
        + "\n\n"
        "NA: not available. P-values were not computed because per-sample probabilities for the proposed model were not retained for paired McNemar testing in this run. "
        "SEN: sensitivity; SPE: specificity; Acc: accuracy; F1: F1-score; MCC: Matthews correlation coefficient; AUC: ROC-AUC.\n",
        encoding="utf-8",
    )
    return out


def plot_table_image(table: pd.DataFrame) -> None:
    rounded = table.copy()
    for col in ["SEN", "SPE", "Acc", "F1", "MCC", "AUC", "PR-AUC"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 2.25))
    ax.axis("off")
    ax.text(0.0, 1.09, "Table 1", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(
        0.0,
        1.02,
        "Performance comparison on the balanced hard-test dual-screening benchmark.",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
    )

    table_artist = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        colWidths=[0.34, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.07],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(6.6)
    table_artist.scale(1, 1.28)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.4 if row == 0 else 0.25)
        if row == 0:
            cell.set_facecolor("#F1F3F4")
            cell.set_text_props(fontweight="bold")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")
        if row == len(rounded) and col in [1, 2, 3, 4, 5, 6, 7]:
            cell.set_text_props(fontweight="bold")

    ax.text(
        0.0,
        -0.12,
        "NA: not available for paired McNemar testing; values are macro-averaged over AMP hard and TOX hard tests.",
        transform=ax.transAxes,
        fontsize=6.3,
        va="top",
    )
    save_pub(fig, FIG_DIR / "table1_jpa_style_sota_duel")
    plt.close(fig)


def plot_metric_bars(table: pd.DataFrame) -> None:
    metrics = [("Acc", "Acc"), ("SPE", "Sp"), ("SEN", "Se"), ("MCC", "MCC")]
    letters = ["a", "b", "c", "d"]
    colors = {
        "AMPlify-balanced + ToxinPred3-ML": "#E7B6B5",
        "AMPlify-imbalanced + ToxinPred3-ML": "#F4D9D9",
        "AMPlify-balanced + ToxinPred3-Hybrid": "#D6D6D6",
        "AMPlify-imbalanced + ToxinPred3-Hybrid": "#B7C6D9",
        "Residual Cross-Attention": "#00008B",
    }
    reverse_label = {v: k for k, v in TABLE_MODEL.items()}
    table = table.copy()
    table["raw_model"] = table["Model"].map(reverse_label)

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.25), sharey=False)
    for ax, (metric, ylabel), letter in zip(axes, metrics, letters):
        vals = []
        labels = []
        bar_colors = []
        for model in MODEL_ORDER:
            row = table.loc[table["raw_model"] == model].iloc[0]
            vals.append(float(row[metric]))
            labels.append(MODEL_SHORT[model])
            bar_colors.append(colors[model])
        x = np.arange(len(vals))
        ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.6, width=0.78)
        ax.set_ylim(0, 1.06 if metric != "MCC" else 0.86)
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=6.2, fontweight="bold")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", color="#D9DEE2", linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.text(-0.22, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        for xpos, value in zip(x, vals):
            ax.text(xpos, value + 0.018, f"{value:.4f}", ha="center", va="bottom", fontsize=4.8, rotation=0)
    fig.subplots_adjust(wspace=0.38, bottom=0.34, left=0.07, right=0.995, top=0.88)
    save_pub(fig, FIG_DIR / "fig_jpa_style_sota_duel_metric_bars")
    plt.close(fig)


def build_mc_bias_table() -> pd.DataFrame:
    shortcut = pd.read_csv(TABLE_DIR / "sota_shortcut_bias_audit.csv")
    lookup = shortcut.set_index("dataset")

    def row(name: str, pos_name: str, neg_name: str, focus: str) -> dict[str, float | str]:
        pos = lookup.loc[pos_name]
        neg = lookup.loc[neg_name]
        return {
            "comparison": name,
            "positive_group": pos_name,
            "negative_group": neg_name,
            "pos_n": int(pos["n"]),
            "neg_n": int(neg["n"]),
            "pos_m_start": float(pos["m_start_rate"]),
            "neg_m_start": float(neg["m_start_rate"]),
            "delta_m_neg_minus_pos": float(neg["m_start_rate"] - pos["m_start_rate"]),
            "pos_contains_c": float(pos["contains_c_rate"]),
            "neg_contains_c": float(neg["contains_c_rate"]),
            "delta_contains_c_pos_minus_neg": float(pos["contains_c_rate"] - neg["contains_c_rate"]),
            "pos_mean_c_freq": float(pos["mean_c_frequency"]),
            "neg_mean_c_freq": float(neg["mean_c_frequency"]),
            "delta_mean_c_pos_minus_neg": float(pos["mean_c_frequency"] - neg["mean_c_frequency"]),
            "bias_call": focus,
        }

    rows = [
        row(
            "AMPlify original test (balanced negative)",
            "AMPlify test AMP",
            "AMPlify test non-AMP balanced",
            "Severe M-start bias; moderate cysteine imbalance.",
        ),
        row(
            "AMPlify original test (imbalanced negative)",
            "AMPlify test AMP",
            "AMPlify test non-AMP imbalanced",
            "Extreme M-start bias; cysteine presence and frequency also differ.",
        ),
        row(
            "ToxinPred3 original test",
            "ToxinPred3 test toxic",
            "ToxinPred3 test non-toxic",
            "Severe cysteine bias; non-toxic group also has higher M-start rate.",
        ),
        row(
            "Our AMP hard test",
            "Our AMP hard positive",
            "Our AMP hard negative",
            "M-start exactly matched by construction; cysteine was not the AMP matching target.",
        ),
        row(
            "Our TOX hard test",
            "Our TOX hard toxic",
            "Our TOX hard non-toxic",
            "Cysteine-containing rate exactly matched and mean C frequency nearly identical.",
        ),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_sota_test_mc_bias_summary.csv", index=False)

    rounded = out.copy()
    numeric_cols = [c for c in rounded.columns if c not in ["comparison", "positive_group", "negative_group", "bias_call"]]
    for col in numeric_cols:
        if col.endswith("_n") or col in ["pos_n", "neg_n"]:
            continue
        rounded[col] = rounded[col].map(lambda x: f"{float(x):.4f}")
    (TABLE_DIR / "table_sota_test_mc_bias_summary.md").write_text(
        "# M/C shortcut-bias summary for source test sets\n\n"
        + rounded.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    return out


def write_cn_report(table: pd.DataFrame, bias: pd.DataFrame) -> None:
    ours = table.loc[table["Model"] == "Residual Cross-Attention"].iloc[0]
    best_sota_auc = table.loc[table["Model"] != "Residual Cross-Attention", "AUC"].max()
    best_sota_mcc = table.loc[table["Model"] != "Residual Cross-Attention", "MCC"].max()
    lines = [
        "# JPA-style SOTA Table/Figure Handover",
        "",
        "## 已生成文件",
        "",
        "- `tables/table1_jpa_style_sota_duel.csv` / `.md`：Table 1 风格主表",
        "- `figures/table1_jpa_style_sota_duel.*`：可直接预览的表格图",
        "- `figures/fig_jpa_style_sota_duel_metric_bars.*`：四面板柱状图，包含 Acc、Sp、Se、MCC",
        "- `tables/table_sota_test_mc_bias_summary.csv` / `.md`：SOTA 原始 test 的 M/C 偏倚总结",
        "",
        "## 主结果",
        "",
        (
            f"Residual Cross-Attention 在双任务 hard-test 宏平均表中保持第一："
            f"SEN={ours['SEN']:.4f}, SPE={ours['SPE']:.4f}, Acc={ours['Acc']:.4f}, "
            f"F1={ours['F1']:.4f}, MCC={ours['MCC']:.4f}, AUC={ours['AUC']:.4f}, PR-AUC={ours['PR-AUC']:.4f}。"
        ),
        (
            f"相对于最强 paired SOTA，AUC 从 {best_sota_auc:.4f} 提升到 {ours['AUC']:.4f}，"
            f"MCC 从 {best_sota_mcc:.4f} 提升到 {ours['MCC']:.4f}。"
        ),
        "",
        "## M/C 偏倚结论",
        "",
        (
            "AMPlify 原始 test 存在明显 M-start 偏倚：AMP 阳性测试集 M-start 仅 0.0216，"
            "balanced non-AMP 为 0.4395，imbalanced non-AMP 为 0.9399。"
        ),
        (
            "ToxinPred3 原始 test 存在强 Cys 偏倚：toxic 测试集 contains-C 为 0.7228、mean C frequency 为 0.1546，"
            "non-toxic 测试集分别只有 0.2310 和 0.0186。"
        ),
        (
            "因此，它们的原始 test 确实存在可诱导模型利用 M/C 组成信息的 shortcut 风险。"
            "我们的 AMP hard test 已把 M-start 正负比例匹配到 0.0411/0.0411；"
            "TOX hard test 已把 contains-C 匹配到 0.5789/0.5789，mean C frequency 也非常接近 0.0627/0.0644。"
        ),
        "",
        "## Bias table",
        "",
        bias.round(4).to_markdown(index=False),
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "jpa_style_sota_and_mc_bias_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Legacy entry point kept for compatibility. The current main-line SOTA
    # figure/table generation lives in src/figures/plot_resultC_sota_top_pipelines.py.
    import runpy

    script = REPO_ROOT / "src" / "figures" / "plot_resultC_sota_top_pipelines.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
