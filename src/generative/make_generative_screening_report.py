from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data--final" / "generative_sft_evodiff"
DEFAULT_LEAD_ID = "MD_PRIOR_LEAD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tables, figures, and report for SFT-EvoDiff candidate screening.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--screening-subdir", default="screening")
    parser.add_argument("--output-prefix", default="generative_sft_screening_report")
    parser.add_argument("--lead-id", default=DEFAULT_LEAD_ID)
    parser.add_argument("--date", default="2026-06-12")
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "tables": root / "tables",
        "figures": root / "figures",
        "reports": root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def write_tables(df: pd.DataFrame, shortlist: pd.DataFrame, lead_id: str, table_dir: Path) -> dict[str, Path]:
    top_cols = [
        "rank",
        "candidate_id",
        "sequence",
        "length",
        "candidate_source",
        "amp_score",
        "tox_score",
        "amp_safe_joint_score",
        "decision_score",
        "qc_pass",
        "entropy_norm",
        "basic_fraction",
        "aromatic_fraction",
        "c_fraction",
    ]
    top_cols = [column for column in top_cols if column in df.columns]
    top50 = df.loc[:, top_cols].head(50)
    top50_path = table_dir / "top50_screened_candidates.csv"
    top50.to_csv(top50_path, index=False)

    lead = df[df["candidate_id"] == lead_id].copy()
    lead_path = table_dir / "md_prior_lead_rank.csv"
    lead.loc[:, top_cols].to_csv(lead_path, index=False)

    guardrail_cols = ["qc_pass", "low_complexity_flag", "homopolymer_flag", "gf_rich_flag"]
    summary_rows = []
    for name, frame in [("all_candidates", df), ("shortlist", shortlist)]:
        row = {
            "set": name,
            "n": int(frame.shape[0]),
            "mean_amp_score": float(frame["amp_score"].mean()),
            "mean_tox_score": float(frame["tox_score"].mean()),
            "mean_decision_score": float(frame["decision_score"].mean()),
        }
        for column in guardrail_cols:
            row[column + "_rate"] = float(frame[column].mean()) if column in frame.columns else np.nan
        summary_rows.append(row)
    if "candidate_source" in df.columns:
        for source, frame in df.groupby("candidate_source", sort=False):
            row = {
                "set": f"source:{source}",
                "n": int(frame.shape[0]),
                "mean_amp_score": float(frame["amp_score"].mean()),
                "mean_tox_score": float(frame["tox_score"].mean()),
                "mean_decision_score": float(frame["decision_score"].mean()),
            }
            for column in guardrail_cols:
                row[column + "_rate"] = float(frame[column].mean()) if column in frame.columns else np.nan
            summary_rows.append(row)
    qc_summary = pd.DataFrame(summary_rows)
    qc_path = table_dir / "screening_qc_summary.csv"
    qc_summary.to_csv(qc_path, index=False)

    quantile_rows = []
    quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for column in ["amp_score", "tox_score", "amp_safe_joint_score", "decision_score", "entropy_norm"]:
        for q, value in df[column].quantile(quantiles).items():
            quantile_rows.append({"metric": column, "quantile": q, "value": float(value)})
    quantile_path = table_dir / "screening_score_quantiles.csv"
    pd.DataFrame(quantile_rows).to_csv(quantile_path, index=False)

    return {
        "top50": top50_path,
        "lead": lead_path,
        "qc": qc_path,
        "quantiles": quantile_path,
    }


def plot_landscape(df: pd.DataFrame, shortlist: pd.DataFrame, lead_id: str, figure_dir: Path) -> None:
    lead = df[df["candidate_id"] == lead_id]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    background = df[~df["candidate_id"].isin(shortlist["candidate_id"])]
    ax.scatter(
        background["amp_score"],
        background["tox_score"],
        s=8,
        c="#B0BEC5",
        alpha=0.35,
        linewidths=0,
        label="Candidate pool",
    )
    ax.scatter(
        shortlist["amp_score"],
        shortlist["tox_score"],
        s=14,
        c="#4DBBD5",
        alpha=0.75,
        linewidths=0,
        label="Computational shortlist",
    )
    if not lead.empty:
        row = lead.iloc[0]
        ax.scatter(
            [row["amp_score"]],
            [row["tox_score"]],
            marker="*",
            s=170,
            c="#E64B35",
            edgecolors="black",
            linewidths=0.5,
            label="MD-prior lead",
            zorder=5,
        )
        ax.annotate(
            f"rank {int(row['rank'])}/{len(df)}",
            xy=(row["amp_score"], row["tox_score"]),
            xytext=(-72, 24),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.7},
        )
    ax.axvline(0.70, color="#777777", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.30, color="#777777", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted AMP probability")
    ax.set_ylabel("Predicted toxicity probability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, color="#E0E0E0")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    save_figure(fig, figure_dir, "fig1_sft_evodiff_screening_landscape")


def plot_distributions(df: pd.DataFrame, shortlist: pd.DataFrame, lead_id: str, figure_dir: Path) -> None:
    lead = df[df["candidate_id"] == lead_id]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=False)
    metrics = [
        ("amp_score", "AMP probability", "#E64B35"),
        ("tox_score", "Toxicity probability", "#4DBBD5"),
        ("decision_score", "Decision score", "#00A087"),
    ]
    for ax, (column, label, color) in zip(axes, metrics):
        ax.hist(df[column], bins=40, color="#B0BEC5", alpha=0.55, density=True, label="All")
        ax.hist(shortlist[column], bins=25, color=color, alpha=0.60, density=True, label="Shortlist")
        if not lead.empty:
            ax.axvline(float(lead.iloc[0][column]), color="#C00000", lw=1.4)
        ax.set_xlabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", ls="--", lw=0.4, color="#E0E0E0")
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=7)
    save_figure(fig, figure_dir, "fig2_sft_evodiff_score_distributions")


def report_text(
    root: Path,
    df: pd.DataFrame,
    shortlist: pd.DataFrame,
    lead_id: str,
    generation_manifest: dict,
    screening_manifest: dict,
    table_paths: dict[str, Path],
    date: str,
    screening_subdir: str,
) -> str:
    lead = df[df["candidate_id"] == lead_id].iloc[0]
    lead_in_shortlist = bool(lead_id in set(shortlist["candidate_id"])) if "candidate_id" in shortlist.columns else False
    if shortlist.empty:
        shortlist_scope = "empty"
        shortlist_qc = float("nan")
    else:
        shortlist_scope = shortlist["shortlist_scope"].iloc[0] if "shortlist_scope" in shortlist.columns else "threshold_rule"
        shortlist_qc = shortlist["qc_pass"].mean()
    if lead_in_shortlist:
        lead_interpretation = (
            "The MD-prior peptide was retained by the same model/QC screening rule used for all generated candidates."
        )
    else:
        lead_interpretation = (
            "The MD-prior peptide was not retained by the low-toxicity computational shortlist. "
            "It should be described only as an externally motivated MD-prior reference candidate that was evaluated by the same predictor, "
            "not as a model-selected low-toxicity hit."
        )
    lines = [
        "# SFT-EvoDiff Generative Screening Report",
        "",
        f"Date: {date}",
        "",
        "## Pipeline",
        "",
        (
            "A constraint-aware SFT-EvoDiff sampler was used to generate a 4,999-sequence de novo peptide pool. "
            "Sequence lengths were sampled from the empirical AMP-positive training length distribution, and masked decoding used a repetition penalty of "
            f"{generation_manifest.get('penalty', 'NA')}."
        ),
        "",
        (
            f"The pre-specified MD-prior peptide `{lead['sequence']}` was appended as `{lead_id}`, producing a 5,000-member evaluation pool. "
            "All candidates were scored by the same main-line Residual CA CCD/ESM-2 fusion predictor ensemble and the same QC guardrails."
        ),
        "",
        "## Main Results",
        "",
        f"- Generated accepted sequences: {generation_manifest['n_generated_accepted']}; total pool size: {generation_manifest['n_pool']}.",
        f"- Overall QC pass rate: {screening_manifest['qc_pass_rate']:.4f}.",
        f"- Low-complexity rate: {screening_manifest['low_complexity_rate']:.4f}; homopolymer rate: {screening_manifest['homopolymer_rate']:.4f}; G/F-rich rate: {screening_manifest['gf_rich_rate']:.4f}.",
        (
            f"- MD-prior lead rank: {int(lead['rank'])}/{len(df)} "
            f"(top {100.0 * int(lead['rank']) / len(df):.2f}%). "
            f"AMP score={lead['amp_score']:.6f}, TOX score={lead['tox_score']:.6f}, decision score={lead['decision_score']:.6f}."
        ),
        f"- MD lead retained by shortlist: {lead_in_shortlist}.",
        f"- Candidate shortlist: {shortlist_scope}, n={len(shortlist)}, QC pass rate={shortlist_qc:.4f}.",
        "",
        "## Interpretation",
        "",
        lead_interpretation,
        "",
        (
            "The computational contribution can be framed as a constraint-aware SFT-EvoDiff generation and dual-task residual-fusion screening workflow: "
            "length-prior sampling, repetition-penalized decoding, AMP/TOX ensemble scoring, nonlinear geometric decision scoring, and explicit anti-reward-hacking guardrails."
        ),
        "",
        "## Output Files",
        "",
        f"- All screened candidates: `{root / screening_subdir / 'screened_candidates_all.csv'}`",
        f"- Candidate shortlist: `{root / screening_subdir / 'screened_candidates_shortlist.csv'}`",
        f"- Top-50 table: `{table_paths['top50']}`",
        f"- MD lead rank table: `{table_paths['lead']}`",
        f"- QC summary: `{table_paths['qc']}`",
        f"- Score quantiles: `{table_paths['quantiles']}`",
        f"- Figure 1: `{root / 'figures' / 'fig1_sft_evodiff_screening_landscape.svg'}`",
        f"- Figure 2: `{root / 'figures' / 'fig2_sft_evodiff_score_distributions.svg'}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    root = Path(args.root)
    dirs = ensure_dirs(root)
    screening_dir = root / args.screening_subdir
    df = pd.read_csv(screening_dir / "screened_candidates_all.csv")
    shortlist = pd.read_csv(screening_dir / "screened_candidates_shortlist.csv")
    generation_manifest = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))
    screening_manifest = json.loads((screening_dir / "screening_manifest.json").read_text(encoding="utf-8"))

    table_paths = write_tables(df, shortlist, args.lead_id, dirs["tables"])
    plot_landscape(df, shortlist, args.lead_id, dirs["figures"])
    plot_distributions(df, shortlist, args.lead_id, dirs["figures"])

    report = report_text(
        root,
        df,
        shortlist,
        args.lead_id,
        generation_manifest,
        screening_manifest,
        table_paths,
        args.date,
        args.screening_subdir,
    )
    report_path = dirs["reports"] / f"{args.output_prefix}_{args.date.replace('-', '')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[DONE] report -> {report_path}")
    print(f"[DONE] tables -> {dirs['tables']}")
    print(f"[DONE] figures -> {dirs['figures']}")


if __name__ == "__main__":
    main()
