from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_final_benchmark import REPO_ROOT, choose_latest, find_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile potential shortcut features in final AMP/TOX train/test splits.")
    parser.add_argument("--amp-dir", default=None, help="Final AMP benchmark directory. Auto-discovered when omitted.")
    parser.add_argument("--tox-dir", default=None, help="Final TOX benchmark directory. Auto-discovered when omitted.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Split names to profile.")
    parser.add_argument("--amp-splits", nargs="+", default=None, help="AMP-specific split names. Overrides --splits for AMP.")
    parser.add_argument("--tox-splits", nargs="+", default=None, help="TOX-specific split names. Overrides --splits for TOX.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data--final" / "reports"))
    parser.add_argument("--figure-name", default="shortcut_bias_proof.png")
    return parser.parse_args()


def discover_dirs(amp_dir: str | None, tox_dir: str | None) -> tuple[Path, Path]:
    selected = choose_latest(find_candidates([REPO_ROOT / "data--final", REPO_ROOT / "data" / "amp"]))
    amp_path = Path(amp_dir).resolve() if amp_dir else Path(selected["amp"].path).resolve()
    tox_path = Path(tox_dir).resolve() if tox_dir else Path(selected["tox"].path).resolve()
    return amp_path, tox_path


def label_name(label_id: int) -> str:
    return "positive" if int(label_id) == 1 else "negative"


def load_split(bench_dir: Path, split: str) -> pd.DataFrame:
    path = bench_dir / "splits" / f"{split}.csv"
    df = pd.read_csv(path)
    if "sequence" not in df.columns or "label_id" not in df.columns:
        raise ValueError(f"{path} must contain sequence and label_id columns")
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    df["label_id"] = df["label_id"].astype(int)
    return df


def profile_amp(df: pd.DataFrame, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    starts_m = df["sequence"].str.startswith("M")
    for label_id, sub_df in df.assign(starts_m=starts_m).groupby("label_id"):
        rows.append(
            {
                "task": "amp",
                "split": split,
                "label_id": int(label_id),
                "label": label_name(int(label_id)),
                "metric": "starts_with_M_proportion",
                "value": float(sub_df["starts_m"].mean()),
                "n": int(sub_df.shape[0]),
                "count": int(sub_df["starts_m"].sum()),
            }
        )
    return rows


def profile_tox(df: pd.DataFrame, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    c_count = df["sequence"].str.count("C")
    c_frequency = c_count / df["sequence"].str.len().clip(lower=1)
    contains_c = c_count > 0
    work = df.assign(c_count=c_count, c_frequency=c_frequency, contains_c=contains_c)
    for label_id, sub_df in work.groupby("label_id"):
        rows.append(
            {
                "task": "tox",
                "split": split,
                "label_id": int(label_id),
                "label": label_name(int(label_id)),
                "metric": "contains_C_proportion",
                "value": float(sub_df["contains_c"].mean()),
                "n": int(sub_df.shape[0]),
                "count": int(sub_df["contains_c"].sum()),
            }
        )
        rows.append(
            {
                "task": "tox",
                "split": split,
                "label_id": int(label_id),
                "label": label_name(int(label_id)),
                "metric": "mean_C_frequency",
                "value": float(sub_df["c_frequency"].mean()),
                "n": int(sub_df.shape[0]),
                "count": float(sub_df["c_count"].sum()),
            }
        )
        rows.append(
            {
                "task": "tox",
                "split": split,
                "label_id": int(label_id),
                "label": label_name(int(label_id)),
                "metric": "mean_C_count",
                "value": float(sub_df["c_count"].mean()),
                "n": int(sub_df.shape[0]),
                "count": float(sub_df["c_count"].sum()),
            }
        )
    return rows


def plot_bias(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary[summary["metric"].isin(["starts_with_M_proportion", "contains_C_proportion", "mean_C_frequency"])].copy()
    labels = {
        "starts_with_M_proportion": "AMP: starts with M",
        "contains_C_proportion": "TOX: contains C",
        "mean_C_frequency": "TOX: mean C frequency",
    }
    colors = {"negative": "#4DBBD5", "positive": "#E64B35"}

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=220, constrained_layout=True)
    preferred_order = ["train", "val", "test", "test_hard_amp", "test_hard_tox"]
    for axis, metric in zip(axes, labels):
        sub = plot_df[plot_df["metric"] == metric].copy()
        observed = set(sub["split"].unique())
        splits = [split for split in preferred_order if split in observed]
        splits.extend(sorted(observed.difference(splits)))
        x_positions = np.arange(len(splits))
        width = 0.34
        for offset, label in [(-width / 2, "negative"), (width / 2, "positive")]:
            values = []
            for split in splits:
                value = sub[(sub["split"] == split) & (sub["label"] == label)]["value"]
                values.append(float(value.iloc[0]) if not value.empty else np.nan)
            axis.bar(x_positions + offset, values, width=width, color=colors[label], label=label, edgecolor="white", linewidth=0.5)
        axis.set_title(labels[metric], fontsize=9)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(splits, fontsize=8)
        axis.set_ylim(0, max(0.05, min(1.0, np.nanmax(sub["value"].to_numpy()) * 1.18)))
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Proportion / frequency", fontsize=8)
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    amp_dir, tox_dir = discover_dirs(args.amp_dir, args.tox_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    amp_splits = args.amp_splits if args.amp_splits is not None else args.splits
    tox_splits = args.tox_splits if args.tox_splits is not None else args.splits
    for split in amp_splits:
        rows.extend(profile_amp(load_split(amp_dir, split), split))
    for split in tox_splits:
        rows.extend(profile_tox(load_split(tox_dir, split), split))

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "shortcut_bias_profile.csv"
    json_path = out_dir / "shortcut_bias_profile.json"
    fig_path = out_dir / args.figure_name
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_bias(summary, fig_path)

    display = summary.copy()
    display["value"] = display["value"].map(lambda value: f"{value:.6f}")
    print(display.to_string(index=False))
    print(f"[DONE] CSV -> {csv_path}")
    print(f"[DONE] JSON -> {json_path}")
    print(f"[DONE] Figure -> {fig_path}")


if __name__ == "__main__":
    main()
