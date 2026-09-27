from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data--final"
    / "generative_selected_residual_evodiff"
    / "sft_final"
    / "screening"
    / "screened_candidates_all.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data--final"
    / "generative_selected_residual_evodiff"
    / "simple_priority_screening"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Top-N candidates by QC pass and sqrt(AMP score * (1 - TOX score))."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--amp-col", default="amp_score")
    parser.add_argument("--tox-col", default="tox_score")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    require_columns(
        df,
        [
            "candidate_id",
            "sequence",
            "candidate_source",
            "length",
            "qc_pass",
            args.amp_col,
            args.tox_col,
        ],
    )

    out = df.copy()
    amp = out[args.amp_col].astype(float).clip(lower=0.0, upper=1.0)
    tox = out[args.tox_col].astype(float).clip(lower=0.0, upper=1.0)
    out["simple_priority_score"] = np.sqrt(amp * (1.0 - tox))
    out["qc_pass"] = out["qc_pass"].astype(bool)

    ranked_all = out.sort_values(
        ["qc_pass", "simple_priority_score", args.amp_col, args.tox_col, "length"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    ranked_all["simple_priority_rank_all"] = np.arange(1, len(ranked_all) + 1)

    qc_ranked = ranked_all[ranked_all["qc_pass"]].copy().reset_index(drop=True)
    qc_ranked["simple_priority_rank_qc"] = np.arange(1, len(qc_ranked) + 1)
    top = qc_ranked.head(args.top_n).copy()

    preferred_columns = [
        "simple_priority_rank_qc",
        "candidate_id",
        "sequence",
        "candidate_source",
        "length",
        "qc_pass",
        args.amp_col,
        args.tox_col,
        "simple_priority_score",
        "amp_score_std",
        "tox_score_std",
        "standard_aa",
        "valid_length",
        "m_start",
        "contains_c",
        "c_fraction",
        "top1_aa_fraction",
        "top2_aa_fraction",
        "entropy_norm",
        "longest_homopolymer_run",
        "gly_phe_fraction",
        "basic_fraction",
        "aromatic_fraction",
        "low_complexity_flag",
        "homopolymer_flag",
        "gf_rich_flag",
        "fasta_id",
    ]
    top_columns = [column for column in preferred_columns if column in top.columns]

    ranked_all.to_csv(output_root / "all_candidates_simple_priority.csv", index=False)
    qc_ranked.to_csv(output_root / "qc_pass_candidates_ranked.csv", index=False)
    top[top_columns].to_csv(output_root / "top50_simple_priority_candidates.csv", index=False)
    md_columns = [
        "simple_priority_rank_qc",
        "candidate_id",
        "sequence",
        "length",
        args.amp_col,
        args.tox_col,
        "simple_priority_score",
    ]
    top[[column for column in md_columns if column in top.columns]].to_markdown(
        output_root / "top50_simple_priority_candidates.md",
        index=False,
        floatfmt=".6f",
    )

    top_md_mask = top.astype(str).apply(
        lambda row: row.str.contains("MD_PRIOR|INDIISWHSKLLPRLLRKIKDLYRKLNNG", regex=True).any(),
        axis=1,
    )
    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_root": str(output_root.resolve()),
        "n_total": int(len(out)),
        "n_qc_pass": int(out["qc_pass"].sum()),
        "qc_pass_rate": float(out["qc_pass"].mean()),
        "top_n": int(args.top_n),
        "priority_score": "sqrt(clipped_amp_score * (1 - clipped_tox_score))",
        "sort_order": [
            "qc_pass descending",
            "simple_priority_score descending",
            "amp_score descending",
            "tox_score ascending",
            "length ascending",
        ],
        "top50_min_amp_score": float(top[args.amp_col].min()),
        "top50_max_tox_score": float(top[args.tox_col].max()),
        "top50_min_priority_score": float(top["simple_priority_score"].min()),
        "top50_median_length": float(top["length"].median()),
        "top50_contains_md_prior": bool(top_md_mask.any()) if len(top) else False,
        "candidate_source_counts": {
            str(key): int(value) for key, value in out["candidate_source"].value_counts(dropna=False).items()
        },
    }
    (output_root / "simple_priority_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Simple Priority Top-50 Candidate Selection",
        "",
        "Selection rule: retain QC-passed candidates and rank by `sqrt(AMP score * (1 - TOX score))`.",
        "",
        f"- Input candidates: {summary['n_total']}",
        f"- QC-passed candidates: {summary['n_qc_pass']} ({100 * summary['qc_pass_rate']:.2f}%)",
        f"- Top-{args.top_n} minimum AMP score: {summary['top50_min_amp_score']:.6f}",
        f"- Top-{args.top_n} maximum TOX score: {summary['top50_max_tox_score']:.6f}",
        f"- Top-{args.top_n} minimum priority score: {summary['top50_min_priority_score']:.6f}",
        f"- Top-{args.top_n} median length: {summary['top50_median_length']:.1f}",
        f"- Top-{args.top_n} contains MD-prior reference: {summary['top50_contains_md_prior']}",
        "",
        "Outputs:",
        f"- `{output_root / 'top50_simple_priority_candidates.csv'}`",
        f"- `{output_root / 'top50_simple_priority_candidates.md'}`",
        f"- `{output_root / 'qc_pass_candidates_ranked.csv'}`",
        f"- `{output_root / 'all_candidates_simple_priority.csv'}`",
    ]
    (output_root / "simple_priority_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
