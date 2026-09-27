from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data--final" / "manuscript_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORTCUT_PROFILE = ROOT / "data--final" / "reports" / "shortcut_bias_profile.csv"
HARD_SUMMARY = ROOT / "data--final" / "reports" / "hard_test_summary.json"
SOTA_BIAS = ROOT / "data--final" / "sota_duel" / "tables" / "table_sota_test_mc_bias_summary.csv"
AMP_IMPORTANCE = ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_amp_importance.csv"
TOX_IMPORTANCE = ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_tox_importance.csv"

FIG_BASENAME = OUT_DIR / "fig1_shortcut_bias_motivation"
SOURCE_DATA = OUT_DIR / "fig1_shortcut_bias_motivation_source_data.csv"
REPORT_PATH = OUT_DIR / "fig1_shortcut_bias_motivation_report.md"

COLORS = {
    "standard": "#C97979",
    "hard": "#5BA7A4",
    "external_m": "#8D8D8D",
    "external_c": "#D59A9A",
    "amp": "#D95F5F",
    "tox": "#4F93A3",
    "class_pos": "#D95F5F",
    "class_neg": "#4F93A3",
    "grid": "#DDE3E8",
    "text": "#20242A",
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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "figure.dpi": 140,
        }
    )


def pct(value: float) -> float:
    return float(value) * 100.0


def get_value(df: pd.DataFrame, task: str, split: str, label: str, metric: str) -> tuple[float, int, float]:
    row = df[
        (df["task"] == task)
        & (df["split"] == split)
        & (df["label"] == label)
        & (df["metric"] == metric)
    ]
    if row.empty:
        raise ValueError(f"Missing value for {task=} {split=} {label=} {metric=}")
    rec = row.iloc[0]
    return float(rec["value"]), int(rec["n"]), float(rec["count"])


def clean_axis(ax: plt.Axes, xgrid: bool = False, ygrid: bool = True) -> None:
    if ygrid:
        ax.grid(axis="y", color=COLORS["grid"], lw=0.5, ls="-", alpha=0.75)
    if xgrid:
        ax.grid(axis="x", color=COLORS["grid"], lw=0.5, ls="-", alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3, color=COLORS["text"], labelcolor=COLORS["text"])
    ax.spines["left"].set_color(COLORS["text"])
    ax.spines["bottom"].set_color(COLORS["text"])


def annotate_vertical(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * 0.018,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=5.9,
            color=COLORS["text"],
        )


def annotate_vertical_small(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = float(bar.get_height())
        if h < 0.35:
            label = "0.0" if h < 0.05 else fmt.format(h)
        else:
            label = fmt.format(h)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * 0.014,
            label,
            ha="center",
            va="bottom",
            fontsize=4.85,
            color=COLORS["text"],
            rotation=0,
        )


def annotate_grouped_vertical(ax: plt.Axes, left_bars, right_bars, fmt: str = "{:.1f}") -> None:
    ymax = ax.get_ylim()[1]
    for bars, ha, xoff in [(left_bars, "right", -1.2), (right_bars, "left", 1.2)]:
        for bar in bars:
            h = float(bar.get_height())
            if h < 0.35:
                label = "0.0" if h < 0.05 else fmt.format(h)
            else:
                label = fmt.format(h)
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(xoff, ymax * 0.014),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=4.85,
                color=COLORS["text"],
            )


def annotate_horizontal(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        w = float(bar.get_width())
        ax.text(
            w + xmax * 0.018,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left",
            va="center",
            fontsize=5.9,
            color=COLORS["text"],
        )


def is_residue_descriptor_feature(name: str, residue: str) -> bool:
    if name == f"AAC_{residue}" or name == f"DP_{residue}":
        return True
    if name == f"APAAC_Pc1.{residue}":
        return True
    if name.startswith("ASDC_") or name.startswith("DDE_"):
        token = name.split("_", 1)[1]
        return len(token) == 2 and residue in token
    if name.startswith("CKSAAP_"):
        token = name.split("_", 1)[1].split(".", 1)[0]
        return len(token) == 2 and residue in token
    return False


def shortcut_related(df: pd.DataFrame, residue: str, n: int) -> pd.DataFrame:
    out = df[df["feature_name"].map(lambda x: is_residue_descriptor_feature(str(x), residue))].copy()
    out["importance_gain"] = pd.to_numeric(out["importance_gain"])
    out["importance_rank"] = pd.to_numeric(out["importance_rank"])
    return out.sort_values("importance_rank").head(n)


def short_feature_name(name: str) -> str:
    return (
        name.replace("APAAC_Pc1.", "APAAC:")
        .replace("CKSAAP_", "CKSAAP:")
        .replace(".gap", " g")
        .replace("ASDC_", "ASDC:")
        .replace("AAC_", "AAC:")
    )


def main() -> None:
    setup_style()

    shortcut = pd.read_csv(SHORTCUT_PROFILE)
    hard = json.loads(HARD_SUMMARY.read_text(encoding="utf-8"))
    sota = pd.read_csv(SOTA_BIAS)
    amp_imp = pd.read_csv(AMP_IMPORTANCE)
    tox_imp = pd.read_csv(TOX_IMPORTANCE)

    source_rows: list[dict[str, object]] = []

    amp_pos_m, amp_pos_n, amp_pos_count = get_value(shortcut, "amp", "test", "positive", "starts_with_M_proportion")
    amp_neg_m, amp_neg_n, amp_neg_count = get_value(shortcut, "amp", "test", "negative", "starts_with_M_proportion")
    tox_pos_c, tox_pos_n, tox_pos_c_count = get_value(shortcut, "tox", "test", "positive", "contains_C_proportion")
    tox_neg_c, tox_neg_n, tox_neg_c_count = get_value(shortcut, "tox", "test", "negative", "contains_C_proportion")
    tox_pos_cfreq, _, tox_pos_cfreq_count = get_value(shortcut, "tox", "test", "positive", "mean_C_frequency")
    tox_neg_cfreq, _, tox_neg_cfreq_count = get_value(shortcut, "tox", "test", "negative", "mean_C_frequency")

    amp_hard_pos_m = hard["amp_hard"]["positive_starts_M_proportion"]
    amp_hard_neg_m = hard["amp_hard"]["negative_starts_M_proportion"]
    tox_hard_pos_c = hard["tox_hard"]["positive_contains_C_proportion"]
    tox_hard_neg_c = hard["tox_hard"]["negative_contains_C_proportion"]
    tox_hard_pos_cfreq = hard["tox_hard"]["positive_mean_C_frequency"]
    tox_hard_neg_cfreq = hard["tox_hard"]["negative_mean_C_frequency"]

    fig = plt.figure(figsize=(7.65, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.34, 1.16, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.065, right=0.99, top=0.835, bottom=0.25, wspace=0.50)

    # Panel A: internal shortcut ratios before/after hard-test matching.
    pair_specs = [
        ("Std", pct(amp_pos_m), pct(amp_neg_m), "AMP+", "AMP-", "AMP M-start"),
        ("Hard", pct(amp_hard_pos_m), pct(amp_hard_neg_m), "AMP+", "AMP-", "AMP M-start"),
        ("Std", pct(tox_pos_c), pct(tox_neg_c), "Toxic", "Non", "TOX Cys+"),
        ("Hard", pct(tox_hard_pos_c), pct(tox_hard_neg_c), "Toxic", "Non", "TOX Cys+"),
        ("Std", pct(tox_pos_cfreq), pct(tox_neg_cfreq), "Toxic", "Non", "TOX Cys freq."),
        ("Hard", pct(tox_hard_pos_cfreq), pct(tox_hard_neg_cfreq), "Toxic", "Non", "TOX Cys freq."),
    ]
    centers = np.arange(len(pair_specs))
    width = 0.28
    pos_vals = [spec[1] for spec in pair_specs]
    neg_vals = [spec[2] for spec in pair_specs]
    bars_pos = ax_a.bar(centers - width / 2, pos_vals, width, color=COLORS["class_pos"], label="Positive / toxic")
    bars_neg = ax_a.bar(centers + width / 2, neg_vals, width, color=COLORS["class_neg"], label="Negative / non-toxic")
    for idx in [1, 3, 5]:
        ax_a.axvspan(idx - 0.5, idx + 0.5, color="#EEF5F4", zorder=-2)
    ax_a.set_xticks(centers)
    ax_a.set_xticklabels([spec[0] for spec in pair_specs], fontsize=5.8)
    ax_a.tick_params(axis="x", pad=2)
    ax_a.set_ylabel("Ratio or residue frequency (%)")
    ax_a.set_ylim(0, 102)
    ax_a.set_title("Internal benchmark matching")
    annotate_grouped_vertical(ax_a, bars_pos, bars_neg)
    ax_a.legend(loc="upper right", handlelength=0.85, borderaxespad=0.2, labelspacing=0.25)
    clean_axis(ax_a)
    for center, label in [(0.5, "AMP\nM-start"), (2.5, "TOX\nCys+"), (4.5, "TOX\nCys freq.")]:
        ax_a.text(
            center,
            -0.18,
            label,
            transform=ax_a.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.65,
            color=COLORS["text"],
            linespacing=0.9,
        )
    for xpos in [1.5, 3.5]:
        ax_a.axvline(xpos, color="#EDF1F4", lw=0.8, zorder=-1)

    for label, pos_value, neg_value, pos_group, neg_group, metric_name in pair_specs:
        metric_name = f"{label} {metric_name}"
        source_rows.extend(
            [
                {
                    "panel": "A",
                    "dataset": "Internal benchmark",
                    "metric": metric_name,
                    "group": pos_group,
                    "value": pos_value / 100,
                    "value_percent": pos_value,
                    "n": "",
                    "count": "",
                    "note": "actual class-specific ratio/frequency; hard-test is matched, not depleted",
                },
                {
                    "panel": "A",
                    "dataset": "Internal benchmark",
                    "metric": metric_name,
                    "group": neg_group,
                    "value": neg_value / 100,
                    "value_percent": neg_value,
                    "n": "",
                    "count": "",
                    "note": "actual class-specific ratio/frequency; hard-test is matched, not depleted",
                },
            ]
        )
    source_rows.extend(
        [
            {
                "panel": "A_detail",
                "dataset": "AMP standard test",
                "metric": "M-start ratio",
                "group": "positive",
                "value": amp_pos_m,
                "value_percent": pct(amp_pos_m),
                "n": amp_pos_n,
                "count": amp_pos_count,
                "note": "29/1680 = 1.73%",
            },
            {
                "panel": "A_detail",
                "dataset": "AMP standard test",
                "metric": "M-start ratio",
                "group": "negative",
                "value": amp_neg_m,
                "value_percent": pct(amp_neg_m),
                "n": amp_neg_n,
                "count": amp_neg_count,
                "note": "973/1650 = 58.97%",
            },
            {
                "panel": "A_detail",
                "dataset": "TOX standard test",
                "metric": "Contains-C ratio",
                "group": "toxic",
                "value": tox_pos_c,
                "value_percent": pct(tox_pos_c),
                "n": tox_pos_n,
                "count": tox_pos_c_count,
                "note": "189/213 = 88.73%",
            },
            {
                "panel": "A_detail",
                "dataset": "TOX standard test",
                "metric": "Contains-C ratio",
                "group": "non-toxic",
                "value": tox_neg_c,
                "value_percent": pct(tox_neg_c),
                "n": tox_neg_n,
                "count": tox_neg_c_count,
                "note": "63/215 = 29.30%",
            },
            {
                "panel": "A_detail",
                "dataset": "TOX standard test",
                "metric": "Mean Cys frequency",
                "group": "toxic",
                "value": tox_pos_cfreq,
                "value_percent": pct(tox_pos_cfreq),
                "n": tox_pos_n,
                "count": tox_pos_cfreq_count,
                "note": "mean sequence-level Cys frequency",
            },
            {
                "panel": "A_detail",
                "dataset": "TOX standard test",
                "metric": "Mean Cys frequency",
                "group": "non-toxic",
                "value": tox_neg_cfreq,
                "value_percent": pct(tox_neg_cfreq),
                "n": tox_neg_n,
                "count": tox_neg_cfreq_count,
                "note": "mean sequence-level Cys frequency",
            },
        ]
    )

    # Panel B: external test-set class-specific shortcut ratios.
    amp_i = sota.loc[sota["comparison"].str.contains("imbalanced negative")].iloc[0]
    amp_b = sota.loc[sota["comparison"].str.contains("balanced negative")].iloc[0]
    toxinpred = sota.loc[sota["comparison"].eq("ToxinPred3 original test")].iloc[0]
    external_pairs = [
        ("AMPlify-I\nM-start", pct(amp_i["pos_m_start"]), pct(amp_i["neg_m_start"]), "AMP", "non-AMP"),
        ("AMPlify-B\nM-start", pct(amp_b["pos_m_start"]), pct(amp_b["neg_m_start"]), "AMP", "non-AMP"),
        ("ToxinPred3\nCys+", pct(toxinpred["pos_contains_c"]), pct(toxinpred["neg_contains_c"]), "Toxic", "Non-toxic"),
        (
            "ToxinPred3\nCys freq.",
            pct(toxinpred["pos_mean_c_freq"]),
            pct(toxinpred["neg_mean_c_freq"]),
            "Toxic",
            "Non-toxic",
        ),
    ]
    x = np.arange(len(external_pairs))
    b_width = 0.32
    bars_pos = ax_b.bar(
        x - b_width / 2,
        [row[1] for row in external_pairs],
        b_width,
        color=COLORS["class_pos"],
        label="AMP / toxic",
    )
    bars_neg = ax_b.bar(
        x + b_width / 2,
        [row[2] for row in external_pairs],
        b_width,
        color=COLORS["class_neg"],
        label="non-AMP / non-toxic",
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([row[0] for row in external_pairs], fontsize=5.65)
    ax_b.tick_params(axis="x", pad=2)
    ax_b.set_ylim(0, 102)
    ax_b.set_ylabel("Ratio or residue frequency (%)")
    ax_b.set_title("External test-set bias")
    annotate_grouped_vertical(ax_b, bars_pos, bars_neg)
    ax_b.legend(loc="upper right", handlelength=0.85, borderaxespad=0.2, labelspacing=0.25)
    clean_axis(ax_b)
    for label, pos_value, neg_value, pos_group, neg_group in external_pairs:
        source_rows.append(
            {
                "panel": "B",
                "dataset": label.replace("\n", " "),
                "metric": "External test-set shortcut ratio/frequency",
                "group": pos_group,
                "value": pos_value / 100,
                "value_percent": pos_value,
                "n": "",
                "count": "",
                "note": "positive-class value from final external benchmark bias audit",
            }
        )
        source_rows.append(
            {
                "panel": "B",
                "dataset": label.replace("\n", " "),
                "metric": "External test-set shortcut ratio/frequency",
                "group": neg_group,
                "value": neg_value / 100,
                "value_percent": neg_value,
                "n": "",
                "count": "",
                "note": "negative-class value from final external benchmark bias audit",
            }
        )

    # Panel C: shortcut-related CCD features in final-main-line LightGBM importance.
    amp_short = shortcut_related(amp_imp, "M", 4)
    tox_short = shortcut_related(tox_imp, "C", 4)
    imp = pd.concat([amp_short.assign(task_label="AMP / Met"), tox_short.assign(task_label="TOX / Cys")], ignore_index=True)
    imp["importance_gain"] = pd.to_numeric(imp["importance_gain"])
    imp["importance_rank"] = pd.to_numeric(imp["importance_rank"])
    imp["norm_gain"] = imp.groupby("task_label")["importance_gain"].transform(lambda s: 100 * s / s.max())
    imp["plot_label"] = imp["feature_name"].map(short_feature_name)
    imp = imp.sort_values(["task_label", "importance_rank"], ascending=[True, False])
    y = np.arange(len(imp))
    bars = ax_c.barh(
        y,
        imp["norm_gain"],
        color=[COLORS["amp"] if task.startswith("AMP") else COLORS["tox"] for task in imp["task_label"]],
        height=0.68,
    )
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([row.plot_label for row in imp.itertuples()], fontsize=5.8)
    ax_c.set_xlim(0, 112)
    ax_c.set_xlabel("Normalized LightGBM gain")
    ax_c.set_title("CCD shortcut-related features")
    clean_axis(ax_c, xgrid=True, ygrid=False)
    for yi, row in enumerate(imp.itertuples()):
        ax_c.text(
            float(row.norm_gain) + 2.2,
            yi,
            f"rank {int(row.importance_rank)}",
            va="center",
            ha="left",
            fontsize=5.6,
            color=COLORS["text"],
        )
        source_rows.append(
            {
                "panel": "C",
                "dataset": f"{row.task_label} LightGBM compact importance",
                "metric": "Shortcut-related feature importance",
                "group": row.feature_name,
                "value": row.importance_gain,
                "value_percent": row.norm_gain,
                "n": "",
                "count": "",
                "note": f"importance_rank={int(row.importance_rank)}; normalized within task",
            }
        )
    handles = [
        mpl.patches.Patch(color=COLORS["amp"], label="AMP / Met"),
        mpl.patches.Patch(color=COLORS["tox"], label="TOX / Cys"),
    ]
    ax_c.legend(handles=handles, loc="lower right", handlelength=0.9, borderaxespad=0.2, labelspacing=0.25)

    for label, ax in zip(["A", "B", "C"], [ax_a, ax_b, ax_c]):
        ax.text(
            -0.22,
            1.12,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            color=COLORS["text"],
        )

    fig.suptitle(
        "Composition-level shortcuts are present in peptide benchmarks and descriptor models",
        y=0.985,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["text"],
    )

    pd.DataFrame(source_rows).to_csv(SOURCE_DATA, index=False)
    fig.savefig(f"{FIG_BASENAME}.svg", bbox_inches="tight")
    fig.savefig(f"{FIG_BASENAME}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Figure 1 Shortcut Bias Motivation Report

## Figure logic

This compact three-panel version is designed for Result A.

- Panel A: internal standard benchmarks show large M-start/Cys class gaps, while hard tests reduce the controlled gap to zero or near-zero.
- Panel B: external AMPlify and ToxinPred3 source benchmarks also contain large composition-level class gaps.
- Panel C: final-main-line LightGBM CCD importance ranks Met- and Cys-coded descriptor dimensions among the top task-relevant features.

## Key values

- AMP standard M-start: positive {amp_pos_count:.0f}/{amp_pos_n} = {pct(amp_pos_m):.2f}%; negative {amp_neg_count:.0f}/{amp_neg_n} = {pct(amp_neg_m):.2f}%.
- TOX standard contains-C: toxic {tox_pos_c_count:.0f}/{tox_pos_n} = {pct(tox_pos_c):.2f}%; non-toxic {tox_neg_c_count:.0f}/{tox_neg_n} = {pct(tox_neg_c):.2f}%.
- TOX standard mean Cys frequency: toxic {tox_pos_cfreq:.4f}; non-toxic {tox_neg_cfreq:.4f}.
- AMP hard M-start: positive {pct(amp_hard_pos_m):.2f}%; negative {pct(amp_hard_neg_m):.2f}%.
- TOX hard contains-C: toxic {pct(tox_hard_pos_c):.2f}%; non-toxic {pct(tox_hard_neg_c):.2f}%.
- TOX hard mean Cys frequency: toxic {tox_hard_pos_cfreq:.4f}; non-toxic {tox_hard_neg_cfreq:.4f}.
- AMPlify imbalanced original test M-start: AMP {pct(amp_i["pos_m_start"]):.2f}%; non-AMP {pct(amp_i["neg_m_start"]):.2f}%.
- AMPlify balanced original test M-start: AMP {pct(amp_b["pos_m_start"]):.2f}%; non-AMP {pct(amp_b["neg_m_start"]):.2f}%.
- ToxinPred3 original test contains-C: toxic {pct(toxinpred["pos_contains_c"]):.2f}%; non-toxic {pct(toxinpred["neg_contains_c"]):.2f}%.
- ToxinPred3 original test mean Cys frequency: toxic {pct(toxinpred["pos_mean_c_freq"]):.2f}%; non-toxic {pct(toxinpred["neg_mean_c_freq"]):.2f}%.

## Output files

- SVG: `{FIG_BASENAME}.svg`
- PNG: `{FIG_BASENAME}.png`
- Source data: `{SOURCE_DATA}`

## Writing note

The safe claim is that peptide benchmarks contain exploitable composition-level confounders. This motivates shortcut-controlled hard tests and semantic ESM-2 representation learning; it should not be phrased as intentional cheating by prior models.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {FIG_BASENAME}.svg")
    print(f"Wrote {FIG_BASENAME}.png")
    print(f"Wrote {SOURCE_DATA}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
