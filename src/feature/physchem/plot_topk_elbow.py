from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from plotting_common import PALETTE, add_panel_label, apply_publication_style, load_csv, report_path, save_figure, write_source_table


def main() -> None:
    apply_publication_style(font_size=8, axes_linewidth=0.9)

    amp_df = load_csv(report_path("final_lightgbm_compact_amp_topk_sweep.csv"))
    tox_df = load_csv(report_path("final_lightgbm_compact_tox_topk_sweep.csv"))
    amp_df["task"] = "amp"
    tox_df["task"] = "tox"
    source_df = pd.concat([amp_df, tox_df], ignore_index=True)
    write_source_table(source_df, "figureB_topk_elbow_source_data.csv")

    compact_k = 512
    tolerance = 0.0005
    amp_best = float(amp_df["val_roc_auc"].max())
    tox_best = float(tox_df["val_roc_auc"].max())
    amp_lower = amp_best - tolerance
    tox_lower = tox_best - tolerance
    amp_compact = amp_df.loc[amp_df["topk"] == compact_k].iloc[0]
    tox_compact = tox_df.loc[tox_df["topk"] == compact_k].iloc[0]

    fig, ax = plt.subplots(figsize=(6.9, 4.1))

    ax.axhspan(amp_lower, amp_best, color=PALETTE["amp_color"], alpha=0.10, zorder=0)
    ax.axhspan(tox_lower, tox_best, color=PALETTE["tox_color"], alpha=0.10, zorder=0)
    ax.axhline(amp_best, color=PALETTE["amp_color"], linestyle="--", linewidth=1.0)
    ax.axhline(tox_best, color=PALETTE["tox_color"], linestyle="--", linewidth=1.0)

    ax.plot(
        amp_df["topk"],
        amp_df["val_roc_auc"],
        color=PALETTE["amp_color"],
        marker="o",
        markersize=4.2,
        linewidth=1.7,
    )
    ax.plot(
        tox_df["topk"],
        tox_df["val_roc_auc"],
        color=PALETTE["tox_color"],
        marker="o",
        markersize=4.2,
        linewidth=1.7,
    )

    ax.scatter(
        [compact_k, compact_k],
        [amp_compact["val_roc_auc"], tox_compact["val_roc_auc"]],
        s=54,
        color=PALETTE["gold"],
        edgecolor=PALETTE["neutral_black"],
        linewidth=0.7,
        zorder=4,
    )
    ax.axvline(compact_k, color=PALETTE["neutral_mid"], linestyle=(0, (2, 2)), linewidth=0.9)

    midpoint_y = (float(amp_compact["val_roc_auc"]) + float(tox_compact["val_roc_auc"])) / 2
    ax.annotate(
        "Compact choice\nTop-K = 512",
        xy=(compact_k, midpoint_y),
        xytext=(760, 0.9918),
        fontsize=7.3,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.8, color=PALETTE["neutral_black"]),
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PALETTE["neutral_light"], lw=0.6),
    )

    ax.text(3800, float(amp_df.iloc[-1]["val_roc_auc"]) + 0.00010, "AMP", color=PALETTE["amp_color"], fontsize=7.4, fontweight="bold", ha="left")
    ax.text(3800, float(tox_df.iloc[-1]["val_roc_auc"]) - 0.00010, "TOX", color=PALETTE["tox_color"], fontsize=7.4, fontweight="bold", ha="left")

    tick_values = [64, 128, 256, 512, 1024, 2048, 4033]
    ax.set_xscale("log", base=2)
    ax.set_xticks(tick_values)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Top-K selected dimensions")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Compact LightGBM top-K selection", fontsize=10)
    ax.set_xlim(55, 5000)
    ax.set_ylim(0.9840, 0.9986)
    ax.grid(True, axis="y", color=PALETTE["neutral_light"], linewidth=0.5, alpha=0.7, linestyle=(0, (2, 2)))
    add_panel_label(ax, "B")
    fig.tight_layout()

    save_figure(fig, "figureB_topk_elbow")
    plt.close(fig)


if __name__ == "__main__":
    main()
