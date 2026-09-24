from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plotting_common import (
    PALETTE,
    add_panel_label,
    apply_publication_style,
    load_csv,
    load_json,
    report_path,
    save_figure,
    write_source_table,
)


INSET_XLIM = (0.9682, 0.9852)
INSET_YLIM = (0.9840, 0.9974)


def main() -> None:
    apply_publication_style(font_size=8, axes_linewidth=0.9)

    quick_screen = load_csv(report_path("quick_screen_summary.csv"))
    selected = load_json(report_path("final_group_selection.json"))
    selected_union = set(selected["selected_union_groups"])

    pivot = quick_screen.pivot_table(
        index="feature_group",
        columns="task",
        values=["mean_roc_auc", "mean_pr_auc", "n_features"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{task}" for metric, task in pivot.columns]
    pivot = pivot.reset_index()
    pivot["selected_union"] = pivot["feature_group"].isin(selected_union)

    source_table = pivot.loc[:, [
        "feature_group",
        "mean_roc_auc_amp",
        "mean_roc_auc_tox",
        "mean_pr_auc_amp",
        "mean_pr_auc_tox",
        "n_features_amp",
        "selected_union",
    ]].rename(columns={"n_features_amp": "n_features"})
    write_source_table(source_table, "figureA_feature_landscape_source_data.csv")

    fig, ax = plt.subplots(figsize=(6.9, 4.9))

    background = pivot[~pivot["selected_union"]].copy()
    chosen = pivot[pivot["selected_union"]].copy()
    chosen = chosen.rename(columns={
        "mean_roc_auc_amp": "amp",
        "mean_roc_auc_tox": "tox",
    })

    ax.scatter(
        background["mean_roc_auc_amp"],
        background["mean_roc_auc_tox"],
        s=52,
        c=PALETTE["blue_secondary"],
        alpha=0.78,
        edgecolor="none",
        zorder=1,
    )
    ax.scatter(
        chosen["amp"],
        chosen["tox"],
        s=165,
        c=PALETTE["red_strong"],
        marker="*",
        edgecolor=PALETTE["neutral_black"],
        linewidth=0.7,
        zorder=3,
    )

    zoom_rect = Rectangle(
        (INSET_XLIM[0], INSET_YLIM[0]),
        INSET_XLIM[1] - INSET_XLIM[0],
        INSET_YLIM[1] - INSET_YLIM[0],
        fill=False,
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        edgecolor=PALETTE["neutral_dark"],
        zorder=2,
    )
    ax.add_patch(zoom_rect)

    axins = ax.inset_axes([0.60, 0.08, 0.31, 0.25])

    display_offsets = {
        "AAC": (-0.00045, -0.00010),
        "DistancePair": (0.00045, 0.00008),
        "DDE": (-0.00025, 0.00020),
        "APAAC": (-0.00025, -0.00018),
        "CTDD": (-0.00010, 0.00018),
        "KSCTriad": (-0.00010, 0.00005),
        "CKSAAP": (0.00012, 0.00010),
        "ASDC": (0.00018, 0.00018),
    }
    label_offsets = {
        "KSCTriad": (-0.0010, 0.00012),
        "CTDD": (-0.0010, 0.00030),
        "APAAC": (-0.0012, -0.00035),
        "AAC": (-0.0011, -0.00058),
        "DistancePair": (0.00030, -0.00052),
        "DDE": (-0.00045, 0.00042),
        "CKSAAP": (-0.00008, 0.00034),
        "ASDC": (-0.00022, 0.00048),
    }

    for _, row in chosen.iterrows():
        dx, dy = display_offsets.get(row["feature_group"], (0.0, 0.0))
        plot_x = float(row["amp"]) + dx
        plot_y = float(row["tox"]) + dy
        axins.scatter(
            [plot_x],
            [plot_y],
            s=74,
            c=PALETTE["red_strong"],
            marker="*",
            edgecolor=PALETTE["neutral_black"],
            linewidth=0.5,
            zorder=3,
        )
        lx, ly = label_offsets[row["feature_group"]]
        axins.text(
            plot_x + lx,
            plot_y + ly,
            row["feature_group"],
            fontsize=5.6,
            color=PALETTE["neutral_black"],
            ha="left",
            va="center",
        )

    axins.set_xlim(*INSET_XLIM)
    axins.set_ylim(*INSET_YLIM)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["neutral_mid"])
        spine.set_linewidth(0.75)

    ax.set_xlabel("AMP validation ROC-AUC")
    ax.set_ylabel("TOX validation ROC-AUC")
    ax.set_title("Feature-group landscape", fontsize=10)
    ax.set_xlim(0.78, 0.9905)
    ax.set_ylim(0.43, 1.006)
    ax.grid(
        True,
        which="major",
        axis="both",
        color=PALETTE["neutral_light"],
        linewidth=0.55,
        alpha=0.8,
        linestyle=(0, (2, 3)),
    )
    add_panel_label(ax, "A")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.12)

    save_figure(fig, "figureA_feature_group_landscape")
    plt.close(fig)


if __name__ == "__main__":
    main()
