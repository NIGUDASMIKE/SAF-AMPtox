from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data--final" / "generative_main_residual_evodiff"
DEFAULT_LEAD_ID = "MD_PRIOR_LEAD"


TIERS = {
    "broad_0p70_0p30": {"amp": 0.70, "tox": 0.30},
    "stringent_0p90_0p10": {"amp": 0.90, "tox": 0.10},
    "ultra_0p99_0p05": {"amp": 0.99, "tox": 0.05},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize main-line Residual CA screening of base and SFT EvoDiff pools.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--lead-id", default=DEFAULT_LEAD_ID)
    parser.add_argument("--date", default="2026-06-12")
    parser.add_argument(
        "--checkpoint-root",
        default=str(REPO_ROOT / "data--final" / "fusion_models_main_checkpoint_export" / "checkpoints"),
        help="Residual CA checkpoint root used by the upstream candidate scoring run.",
    )
    return parser.parse_args()


def load_screened(root: Path, subdir: str) -> pd.DataFrame:
    path = root / subdir / "screening" / "screened_candidates_all.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def tier_mask(df: pd.DataFrame, amp_threshold: float, tox_threshold: float) -> pd.Series:
    return (
        df["qc_pass"].astype(bool)
        & (df["amp_score"].astype(float) >= amp_threshold)
        & (df["tox_score"].astype(float) <= tox_threshold)
    )


def summarize_pool(name: str, df: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "pool": name,
        "n": int(len(df)),
        "qc_pass_n": int(df["qc_pass"].sum()),
        "qc_pass_rate": float(df["qc_pass"].mean()),
        "amp_ge_0p70_n": int((df["amp_score"] >= 0.70).sum()),
        "amp_ge_0p70_rate": float((df["amp_score"] >= 0.70).mean()),
        "tox_le_0p30_n": int((df["tox_score"] <= 0.30).sum()),
        "tox_le_0p30_rate": float((df["tox_score"] <= 0.30).mean()),
        "mean_amp_score": float(df["amp_score"].mean()),
        "median_amp_score": float(df["amp_score"].median()),
        "mean_tox_score": float(df["tox_score"].mean()),
        "median_tox_score": float(df["tox_score"].median()),
        "mean_decision_score": float(df["decision_score"].mean()),
        "median_decision_score": float(df["decision_score"].median()),
    }
    for tier_name, thresholds in TIERS.items():
        mask = tier_mask(df, thresholds["amp"], thresholds["tox"])
        row[f"{tier_name}_n"] = int(mask.sum())
        row[f"{tier_name}_rate"] = float(mask.mean())
    return row


def make_tier_table(base: pd.DataFrame, sft: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier_name, thresholds in TIERS.items():
        base_mask = tier_mask(base, thresholds["amp"], thresholds["tox"])
        sft_mask = tier_mask(sft, thresholds["amp"], thresholds["tox"])
        base_rate = float(base_mask.mean())
        sft_rate = float(sft_mask.mean())
        rows.append(
            {
                "tier": tier_name,
                "amp_threshold": thresholds["amp"],
                "tox_threshold": thresholds["tox"],
                "base_n": int(base_mask.sum()),
                "base_total": int(len(base)),
                "base_rate": base_rate,
                "sft_n": int(sft_mask.sum()),
                "sft_total": int(len(sft)),
                "sft_rate": sft_rate,
                "rate_delta": sft_rate - base_rate,
                "fold_enrichment": float(sft_rate / base_rate) if base_rate > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def plot_pass_rates(summary: pd.DataFrame, figure_dir: Path) -> None:
    metrics = [
        ("qc_pass_rate", "QC pass"),
        ("amp_ge_0p70_rate", "AMP >= 0.70"),
        ("tox_le_0p30_rate", "TOX <= 0.30"),
        ("broad_0p70_0p30_rate", "Broad dual"),
        ("stringent_0p90_0p10_rate", "Stringent dual"),
        ("ultra_0p99_0p05_rate", "Ultra dual"),
    ]
    colors = {"base_evodiff": "#9E9E9E", "sft_evodiff": "#E64B35"}
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    x = np.arange(len(metrics))
    width = 0.35
    for idx, pool in enumerate(["base_evodiff", "sft_evodiff"]):
        row = summary[summary["pool"] == pool].iloc[0]
        values = [float(row[col]) * 100.0 for col, _ in metrics]
        positions = x + (idx - 0.5) * width
        ax.bar(positions, values, width=width, color=colors[pool], label=pool.replace("_", " ").title())
        for px, value in zip(positions, values):
            ax.text(px, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_ylabel("Candidates (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
    ax.set_ylim(0, max(100.0, summary["qc_pass_rate"].max() * 110.0))
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", ls="--", lw=0.4, color="#DDDDDD")
    save_figure(fig, figure_dir, "fig_base_vs_sft_screening_pass_rates")


def plot_sft_landscape(sft: pd.DataFrame, lead_id: str, figure_dir: Path) -> None:
    broad = tier_mask(sft, 0.70, 0.30)
    stringent = tier_mask(sft, 0.90, 0.10)
    lead = sft[sft["candidate_id"] == lead_id]
    fig, ax = plt.subplots(figsize=(4.3, 3.55))
    ax.scatter(sft.loc[~broad, "amp_score"], sft.loc[~broad, "tox_score"], s=7, color="#B0BEC5", alpha=0.28, linewidths=0)
    ax.scatter(sft.loc[broad & ~stringent, "amp_score"], sft.loc[broad & ~stringent, "tox_score"], s=9, color="#4DBBD5", alpha=0.42, linewidths=0)
    ax.scatter(sft.loc[stringent, "amp_score"], sft.loc[stringent, "tox_score"], s=11, color="#00A087", alpha=0.72, linewidths=0)
    if not lead.empty:
        row = lead.iloc[0]
        ax.scatter([row["amp_score"]], [row["tox_score"]], marker="*", s=165, color="#E64B35", edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(
            f"MD lead\nTOX={row['tox_score']:.3f}",
            xy=(row["amp_score"], row["tox_score"]),
            xytext=(-76, -24),
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.7},
        )
    ax.axvline(0.70, color="#888888", lw=0.7, ls="--")
    ax.axhline(0.30, color="#888888", lw=0.7, ls="--")
    ax.axvline(0.90, color="#333333", lw=0.8, ls=":")
    ax.axhline(0.10, color="#333333", lw=0.8, ls=":")
    label_box = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72}
    ax.text(0.735, 0.255, f"Broad pass\nn={int(broad.sum())}", color="#168CA3", fontsize=7, bbox=label_box)
    ax.text(0.73, 0.075, f"Stringent pass\nn={int(stringent.sum())}", color="#007A5E", fontsize=7, ha="left", bbox=label_box)
    ax.set_xlabel("Predicted AMP probability")
    ax.set_ylabel("Predicted toxicity probability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.4, color="#E0E0E0")
    save_figure(fig, figure_dir, "fig_sft_evodiff_amp_tox_landscape")


def build_validation_set(sft: pd.DataFrame, lead_id: str) -> pd.DataFrame:
    strict = sft[tier_mask(sft, 0.90, 0.10)].copy()
    strict["selection_route"] = "model_stringent_threshold_pass"
    lead = sft[sft["candidate_id"] == lead_id].copy()
    if not lead.empty and lead_id not in set(strict["candidate_id"]):
        lead["selection_route"] = "md_prior_reference_predictor_flagged"
        strict = pd.concat([strict, lead], ignore_index=True)
    elif not lead.empty:
        strict.loc[strict["candidate_id"] == lead_id, "selection_route"] = "model_pass_and_md_prior"
    return strict.sort_values(["selection_route", "rank"], ascending=[True, True]).reset_index(drop=True)


def build_md_ready_top50(sft: pd.DataFrame) -> pd.DataFrame:
    strict = sft[tier_mask(sft, 0.90, 0.10)].copy()
    practical = strict[
        strict["length"].between(12, 35)
        & ~strict["contains_c"].astype(bool)
        & ~strict["m_start"].astype(bool)
    ].copy()
    if practical.shape[0] < 50:
        practical = strict[strict["length"].between(10, 40) & ~strict["contains_c"].astype(bool)].copy()
    if practical.shape[0] < 50:
        practical = strict.copy()
    practical = practical.sort_values(
        ["decision_score", "amp_score", "tox_score", "length"],
        ascending=[False, False, True, True],
    ).head(50)
    practical["selection_route"] = "md_ready_top50_from_stringent_sft"
    return practical.reset_index(drop=True)


def write_report(
    out_root: Path,
    summary: pd.DataFrame,
    tier_table: pd.DataFrame,
    lead_row: pd.Series,
    validation_set: pd.DataFrame,
    md_top50: pd.DataFrame,
    date: str,
    checkpoint_root: Path,
) -> None:
    base = summary[summary["pool"] == "base_evodiff"].iloc[0]
    sft = summary[summary["pool"] == "sft_evodiff"].iloc[0]
    stringent = tier_table[tier_table["tier"] == "stringent_0p90_0p10"].iloc[0]
    broad = tier_table[tier_table["tier"] == "broad_0p70_0p30"].iloc[0]
    lines = [
        "# Main-Line Residual CA Generative Screening Report",
        "",
        f"Date: {date}",
        "",
        "## Model Identity",
        "",
        f"Generated candidates were scored only with the validation-selected task-specific Residual Cross-Attention CCD/ESM-2 ensemble exported from `{checkpoint_root}`.",
        "Legacy `src/oracle/*`, `best_oracle*.pth`, and no-PHYS8 checkpoints were not used.",
        "",
        "## Base vs SFT EvoDiff",
        "",
        f"- Base EvoDiff control: n={int(base['n'])}; broad dual-pass={int(broad['base_n'])}/{int(broad['base_total'])} ({100*broad['base_rate']:.2f}%).",
        f"- SFT EvoDiff pool: n={int(sft['n'])}; broad dual-pass={int(broad['sft_n'])}/{int(broad['sft_total'])} ({100*broad['sft_rate']:.2f}%), fold-enrichment={broad['fold_enrichment']:.2f}x.",
        f"- Stringent final tier (QC pass, AMP>=0.90, TOX<=0.10): base={int(stringent['base_n'])}/{int(stringent['base_total'])} ({100*stringent['base_rate']:.2f}%), SFT={int(stringent['sft_n'])}/{int(stringent['sft_total'])} ({100*stringent['sft_rate']:.2f}%), fold-enrichment={stringent['fold_enrichment']:.2f}x.",
        "",
        "## MD-Prior Lead",
        "",
        f"- Sequence: `{lead_row['sequence']}`.",
        f"- QC pass: {bool(lead_row['qc_pass'])}; AMP score={lead_row['amp_score']:.6f}; TOX score={lead_row['tox_score']:.6f}; decision-score rank={int(lead_row['rank'])}/{int(sft['n'])}.",
        "- Interpretation: the peptide is strongly AMP-like under the discriminator but is flagged as high toxicity risk. It is therefore retained only as an externally motivated MD-prior reference, not as a model-selected low-toxicity hit.",
        f"- MD-prior validation set size: {len(validation_set)} candidates, including the stringent model-selected tier plus the MD-prior reference row with an explicit `selection_route` flag.",
        f"- MD-ready Top50: n={len(md_top50)}; median length={md_top50['length'].median():.1f}; max TOX={md_top50['tox_score'].max():.6f}; min AMP={md_top50['amp_score'].min():.6f}.",
        "",
        "## Output Files",
        "",
        f"- Summary table: `{out_root / 'tables' / 'base_vs_sft_generation_screening_summary.csv'}`",
        f"- Tier enrichment table: `{out_root / 'tables' / 'base_vs_sft_tier_enrichment.csv'}`",
        f"- Strict SFT shortlist: `{out_root / 'tables' / 'sft_stringent_shortlist.csv'}`",
        f"- MD-ready Top50: `{out_root / 'tables' / 'top50_md_ready_candidates.csv'}`",
        f"- MD-prior validation set: `{out_root / 'tables' / 'md_priority_validation_set.csv'}`",
        f"- Pass-rate figure: `{out_root / 'figures' / 'fig_base_vs_sft_screening_pass_rates.svg'}`",
        f"- SFT AMP/TOX landscape: `{out_root / 'figures' / 'fig_sft_evodiff_amp_tox_landscape.svg'}`",
        "",
    ]
    report_path = out_root / "reports" / f"main_residual_generative_screening_report_{date.replace('-', '')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_root = root / "summary"
    table_dir = out_root / "tables"
    figure_dir = out_root / "figures"
    report_dir = out_root / "reports"
    for path in (table_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
        }
    )

    base_subdir = "base_control_5000" if (root / "base_control_5000" / "screening" / "screened_candidates_all.csv").exists() else "base_control"
    base = load_screened(root, base_subdir)
    sft = load_screened(root, "sft_final")
    summary = pd.DataFrame([summarize_pool("base_evodiff", base), summarize_pool("sft_evodiff", sft)])
    tier_table = make_tier_table(base, sft)
    lead = sft[sft["candidate_id"] == args.lead_id]
    if lead.empty:
        raise RuntimeError(f"Lead id {args.lead_id} not found in SFT pool")
    lead_row = lead.iloc[0]
    validation_set = build_validation_set(sft, args.lead_id)
    strict = sft[tier_mask(sft, 0.90, 0.10)].copy()
    md_top50 = build_md_ready_top50(sft)

    summary.to_csv(table_dir / "base_vs_sft_generation_screening_summary.csv", index=False)
    tier_table.to_csv(table_dir / "base_vs_sft_tier_enrichment.csv", index=False)
    lead.to_csv(table_dir / "md_prior_lead_assessment.csv", index=False)
    strict.to_csv(table_dir / "sft_stringent_shortlist.csv", index=False)
    md_top50.to_csv(table_dir / "top50_md_ready_candidates.csv", index=False)
    validation_set.to_csv(table_dir / "md_priority_validation_set.csv", index=False)

    legacy_manifest = {
        "used_checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "excluded_legacy_patterns": [
            "src/oracle/*",
            "models/best_oracle*.pth",
            "checkpoints/best_oracle_model.pth",
            "data--final/fusion_models_no_phys8_residual_export/checkpoints/*",
        ],
        "note": "Legacy oracle/no-PHYS8 artifacts were left on disk to avoid destructive cleanup, but all scoring manifests in this generative run point to the supplied Residual CA checkpoint root.",
    }
    (out_root / "legacy_oracle_exclusion_manifest.json").write_text(json.dumps(legacy_manifest, indent=2), encoding="utf-8")

    plot_pass_rates(summary, figure_dir)
    plot_sft_landscape(sft, args.lead_id, figure_dir)
    write_report(out_root, summary, tier_table, lead_row, validation_set, md_top50, args.date, Path(args.checkpoint_root))
    print(f"[DONE] summary tables -> {table_dir}")
    print(f"[DONE] figures -> {figure_dir}")
    print(f"[DONE] report -> {report_dir}")


if __name__ == "__main__":
    main()
