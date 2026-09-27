from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_common import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    load_csv,
    load_json,
    pretty_feature_label,
    report_path,
    save_figure,
    write_source_table,
)


TOP_N = 12


def draw_overlap_panel(ax: plt.Axes, amp_total: int, tox_total: int, overlap: int) -> None:
    amp_only = amp_total - overlap
    tox_only = tox_total - overlap

    left = mpatches.Circle((0.42, 0.50), 0.27, facecolor=PALETTE["amp_color"], edgecolor="none", alpha=0.22)
    right = mpatches.Circle((0.58, 0.50), 0.27, facecolor=PALETTE["tox_color"], edgecolor="none", alpha=0.22)
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.28, 0.82, "AMP-specific\n512 features", ha="center", va="center", fontsize=8, fontweight="bold", color=PALETTE["amp_color"])
    ax.text(0.72, 0.82, "TOX-specific\n512 features", ha="center", va="center", fontsize=8, fontweight="bold", color=PALETTE["tox_color"])
    ax.text(0.33, 0.50, f"{amp_only}", ha="center", va="center", fontsize=14, fontweight="semibold", color=PALETTE["neutral_black"])
    ax.text(0.50, 0.50, f"{overlap}", ha="center", va="center", fontsize=14, fontweight="semibold", color=PALETTE["neutral_black"])
    ax.text(0.67, 0.50, f"{tox_only}", ha="center", va="center", fontsize=14, fontweight="semibold", color=PALETTE["neutral_black"])
    ax.text(0.50, 0.18, "Shared union = 832 features", ha="center", va="center", fontsize=8, color=PALETTE["neutral_dark"])

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.08, 0.92)
    ax.axis("off")


def draw_back_to_back_panel(ax: plt.Axes, amp_df: pd.DataFrame, tox_df: pd.DataFrame) -> None:
    amp_top = amp_df.nsmallest(TOP_N, "importance_rank").copy().sort_values("importance_rank", ascending=True)
    tox_top = tox_df.nsmallest(TOP_N, "importance_rank").copy().sort_values("importance_rank", ascending=True)

    amp_top["pretty_label"] = amp_top["feature_name"].map(lambda x: pretty_feature_label(x, max_len=38))
    tox_top["pretty_label"] = tox_top["feature_name"].map(lambda x: pretty_feature_label(x, max_len=38))
    amp_top["norm_gain"] = amp_top["importance_gain"] / amp_top["importance_gain"].max()
    tox_top["norm_gain"] = tox_top["importance_gain"] / tox_top["importance_gain"].max()
    y = np.arange(TOP_N)

    ax.barh(y, -amp_top["norm_gain"], color=PALETTE["amp_color"], alpha=0.92, height=0.72)
    ax.barh(y, tox_top["norm_gain"], color=PALETTE["tox_color"], alpha=0.92, height=0.72)
    ax.axvline(0.0, color=PALETTE["neutral_mid"], linewidth=0.8)

    for idx, (_, row) in enumerate(amp_top.iterrows()):
        ax.text(-1.10, idx, row["pretty_label"], ha="right", va="center", fontsize=6.7, color=PALETTE["neutral_black"])
    for idx, (_, row) in enumerate(tox_top.iterrows()):
        ax.text(1.10, idx, row["pretty_label"], ha="left", va="center", fontsize=6.7, color=PALETTE["neutral_black"])

    ax.text(-0.58, -0.85, f"AMP top {TOP_N}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["amp_color"])
    ax.text(0.58, -0.85, f"TOX top {TOP_N}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["tox_color"])

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-0.4, TOP_N - 0.4)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("Task-wise normalized LightGBM gain")
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
    ax.grid(True, axis="x", color=PALETTE["neutral_light"], linewidth=0.55, alpha=0.8, linestyle=(0, (2, 3)))
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def main() -> None:
    apply_publication_style(font_size=8, axes_linewidth=0.9)

    manifest = load_json(report_path("final_lightgbm_compact_manifest.json"))
    amp_importance = load_csv(report_path("final_lightgbm_compact_amp_importance.csv"))
    tox_importance = load_csv(report_path("final_lightgbm_compact_tox_importance.csv"))

    amp_features = set(manifest["tasks"]["amp"]["selected_features"])
    tox_features = set(manifest["tasks"]["tox"]["selected_features"])
    overlap = len(amp_features & tox_features)
    amp_total = len(amp_features)
    tox_total = len(tox_features)

    overlap_df = pd.DataFrame([
        {"set_name": "AMP_selected", "count": amp_total},
        {"set_name": "TOX_selected", "count": tox_total},
        {"set_name": "intersection", "count": overlap},
        {"set_name": "shared_union", "count": len(amp_features | tox_features)},
    ])
    write_source_table(overlap_df, "figureC_overlap_source_data.csv")

    display_df = pd.concat([
        amp_importance.nsmallest(TOP_N, "importance_rank").assign(task="amp"),
        tox_importance.nsmallest(TOP_N, "importance_rank").assign(task="tox"),
    ], ignore_index=True)
    display_df["pretty_label"] = display_df["feature_name"].map(lambda x: pretty_feature_label(x))
    write_source_table(display_df, "figureC_importance_top12_source_data.csv")

    fig = plt.figure(figsize=(9.2, 4.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 2.45], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    draw_overlap_panel(ax_left, amp_total=amp_total, tox_total=tox_total, overlap=overlap)
    draw_back_to_back_panel(ax_right, amp_importance, tox_importance)

    add_panel_label(ax_left, "C")
    fig.suptitle("Overlap and task-specific feature importance", y=0.98, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.15, wspace=0.16)

    save_figure(fig, "figureC_importance_overlap")
    plt.close(fig)


if __name__ == "__main__":
    main()
