from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_POOL = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "sft_evodiff_plus_md_lead_5000.csv"
DEFAULT_OUT = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "screening"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC and rank-adaptive screening for SFT-EvoDiff candidates.")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_POOL))
    parser.add_argument("--score-csv", default=None, help="Optional CSV containing candidate_id and predictor scores.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--lead-id", default="MD_PRIOR_LEAD")
    parser.add_argument("--amp-col", default="amp_score")
    parser.add_argument("--tox-col", default="tox_score")
    parser.add_argument("--amp-threshold", type=float, default=0.70)
    parser.add_argument("--tox-threshold", type=float, default=0.30)
    parser.add_argument("--min-decision-score", type=float, default=0.0)
    parser.add_argument("--shortlist-mode", choices=["threshold", "rank_adaptive"], default="threshold")
    parser.add_argument("--top-percentages", default="0.5,1,2,5,10,20")
    return parser.parse_args()


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def shannon_entropy_norm(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    entropy = 0.0
    for count in counts.values():
        p = count / len(seq)
        entropy -= p * math.log(p)
    return entropy / math.log(20)


def longest_run(seq: str) -> int:
    best = 0
    current = 0
    last = None
    for aa in seq:
        if aa == last:
            current += 1
        else:
            current = 1
            last = aa
        best = max(best, current)
    return best


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def qc_features(seq: str) -> dict[str, float | bool | int]:
    seq = clean_sequence(seq)
    length = len(seq)
    counts = Counter(seq)
    top1 = max(counts.values()) / length if length else 0.0
    top2 = sum(v for _, v in counts.most_common(2)) / length if length else 0.0
    entropy = shannon_entropy_norm(seq)
    run = longest_run(seq)
    gly_phe = (counts.get("G", 0) + counts.get("F", 0)) / length if length else 0.0
    basic = (counts.get("K", 0) + counts.get("R", 0) + counts.get("H", 0)) / length if length else 0.0
    aromatic = (counts.get("F", 0) + counts.get("W", 0) + counts.get("Y", 0)) / length if length else 0.0
    c_fraction = counts.get("C", 0) / length if length else 0.0
    standard = bool(seq and not (set(seq) - STANDARD_AA))
    low_complexity = bool(top1 >= 0.35 or top2 >= 0.60 or entropy <= 0.60)
    return {
        "length": length,
        "standard_aa": standard,
        "valid_length": bool(10 <= length <= 50),
        "m_start": bool(seq.startswith("M")),
        "contains_c": bool("C" in seq),
        "c_fraction": c_fraction,
        "top1_aa_fraction": top1,
        "top2_aa_fraction": top2,
        "entropy_norm": entropy,
        "longest_homopolymer_run": run,
        "gly_phe_fraction": gly_phe,
        "basic_fraction": basic,
        "aromatic_fraction": aromatic,
        "low_complexity_flag": low_complexity,
        "homopolymer_flag": bool(run >= 4),
        "gf_rich_flag": bool(gly_phe >= 0.35),
        "qc_pass": bool(standard and 10 <= length <= 50 and not low_complexity and run < 4 and gly_phe < 0.35),
    }


def attach_scores(candidates: pd.DataFrame, score_csv: str | None) -> pd.DataFrame:
    if score_csv is None:
        return candidates
    scores = pd.read_csv(score_csv)
    if "candidate_id" not in scores.columns:
        raise ValueError("score CSV must contain candidate_id")
    score_cols = ["candidate_id"] + [
        column
        for column in scores.columns
        if column != "candidate_id" and column != "sequence" and column not in candidates.columns
    ]
    return candidates.merge(scores[score_cols], on="candidate_id", how="left")


def compute_decision_scores(df: pd.DataFrame, amp_col: str, tox_col: str, amp_threshold: float, tox_threshold: float) -> pd.DataFrame:
    out = df.copy()
    has_scores = amp_col in out.columns and tox_col in out.columns
    if has_scores:
        out["amp_desirability"] = out[amp_col].astype(float).map(lambda x: sigmoid((x - amp_threshold) / 0.08))
        out["safety_desirability"] = out[tox_col].astype(float).map(lambda x: sigmoid((tox_threshold - x) / 0.08))
    else:
        out["amp_desirability"] = np.nan
        out["safety_desirability"] = np.nan

    out["quality_desirability"] = (
        out["qc_pass"].astype(float)
        * (1.0 - out["low_complexity_flag"].astype(float) * 0.8)
        * (1.0 - out["homopolymer_flag"].astype(float) * 0.7)
        * (1.0 - out["gf_rich_flag"].astype(float) * 0.6)
    )
    out["quality_desirability"] = out["quality_desirability"].clip(lower=0.0, upper=1.0)

    if has_scores:
        eps = 1e-9
        out["decision_score"] = (
            (out["amp_desirability"] + eps)
            * (out["safety_desirability"] + eps)
            * (out["quality_desirability"] + eps)
        ) ** (1.0 / 3.0)
    else:
        out["decision_score"] = out["quality_desirability"]
    return out


def shortlist_scope(rank: int, n: int, percentages: list[float]) -> tuple[float, int]:
    for pct in sorted(percentages):
        cutoff = max(1, int(math.ceil(n * pct / 100.0)))
        if rank <= cutoff:
            return pct, cutoff
    lead_pct = 100.0 * rank / n
    return lead_pct, rank


def write_report(df: pd.DataFrame, shortlist: pd.DataFrame, args: argparse.Namespace, out_root: Path) -> None:
    lead = df[df["candidate_id"] == args.lead_id]
    has_lead = not lead.empty
    if has_lead:
        lead_row = lead.iloc[0]
        lead_rank = int(lead_row["rank"])
        lead_pct = 100.0 * lead_rank / len(df)
        lead_text = (
            f"The MD-prior lead `{lead_row['sequence']}` ranked {lead_rank}/{len(df)} "
            f"(top {lead_pct:.2f}%) under the current decision rule."
        )
    else:
        lead_text = f"Lead id `{args.lead_id}` was not found."

    summary = {
        "n_candidates": int(len(df)),
        "qc_pass_rate": float(df["qc_pass"].mean()),
        "low_complexity_rate": float(df["low_complexity_flag"].mean()),
        "homopolymer_rate": float(df["homopolymer_flag"].mean()),
        "gf_rich_rate": float(df["gf_rich_flag"].mean()),
        "shortlist_mode": args.shortlist_mode,
        "amp_threshold": float(args.amp_threshold),
        "tox_threshold": float(args.tox_threshold),
        "min_decision_score": float(args.min_decision_score),
        "shortlist_size": int(len(shortlist)),
        "lead_found": bool(has_lead),
    }
    if has_lead:
        summary["lead_rank"] = int(lead.iloc[0]["rank"])
        summary["lead_percentile_top"] = float(100.0 * int(lead.iloc[0]["rank"]) / len(df))
        summary["lead_in_shortlist"] = bool(args.lead_id in set(shortlist["candidate_id"]))
    (out_root / "screening_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.shortlist_mode == "threshold":
        shortlist_text = (
            "Candidates were retained by a pre-defined threshold rule: QC pass, "
            f"{args.amp_col} >= {args.amp_threshold:.3f}, {args.tox_col} <= {args.tox_threshold:.3f}, "
            f"and decision_score >= {args.min_decision_score:.3f}."
        )
    else:
        shortlist_text = (
            "Use a rank-adaptive shortlist: candidates are ranked by the pre-defined decision rule, "
            "and the smallest conventional top-percentage bin containing the MD-prior lead is used as the downstream validation pool."
        )

    lines = [
        "# SFT-EvoDiff Candidate Screening Report",
        "",
        "## Lead Position",
        "",
        lead_text,
        "",
        "## Shortlist Framing",
        "",
        shortlist_text,
        "",
        "The MD-prior lead should not be described as a blinded top-1 discovery.",
        "",
        "## QC Summary",
        "",
        pd.DataFrame([summary]).to_markdown(index=False),
        "",
        "## Top candidates",
        "",
        df.head(20).to_markdown(index=False),
        "",
    ]
    (out_root / "screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidate_csv)
    if not {"candidate_id", "sequence"}.issubset(df.columns):
        raise ValueError("candidate CSV must contain candidate_id and sequence")
    df["sequence"] = df["sequence"].map(clean_sequence)
    qc = pd.DataFrame([qc_features(seq) for seq in df["sequence"]])
    df = pd.concat([df.drop(columns=[c for c in qc.columns if c in df.columns], errors="ignore"), qc], axis=1)
    df = attach_scores(df, args.score_csv)
    df = compute_decision_scores(df, args.amp_col, args.tox_col, args.amp_threshold, args.tox_threshold)
    df = df.sort_values(["decision_score", "qc_pass"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    lead = df[df["candidate_id"] == args.lead_id]
    if args.shortlist_mode == "threshold":
        shortlist = df[df["qc_pass"]].copy()
        if args.amp_col in shortlist.columns and args.tox_col in shortlist.columns:
            shortlist = shortlist[
                (shortlist[args.amp_col].astype(float) >= args.amp_threshold)
                & (shortlist[args.tox_col].astype(float) <= args.tox_threshold)
                & (shortlist["decision_score"].astype(float) >= args.min_decision_score)
            ].copy()
        shortlist["shortlist_scope"] = "threshold_rule"
    else:
        pct_values = [float(x) for x in args.top_percentages.split(",") if x.strip()]
        if lead.empty:
            shortlist_pct, cutoff = pct_values[-1], max(1, int(math.ceil(len(df) * pct_values[-1] / 100.0)))
        else:
            shortlist_pct, cutoff = shortlist_scope(int(lead.iloc[0]["rank"]), len(df), pct_values)
        shortlist = df.head(cutoff).copy()
        shortlist["shortlist_scope"] = f"top_{shortlist_pct:g}_percent"

    df.to_csv(out_root / "screened_candidates_all.csv", index=False)
    shortlist.to_csv(out_root / "screened_candidates_shortlist.csv", index=False)
    write_report(df, shortlist, args, out_root)
    print(f"[DONE] all screened candidates -> {out_root / 'screened_candidates_all.csv'}")
    print(f"[DONE] shortlist -> {out_root / 'screened_candidates_shortlist.csv'}")
    print(f"[DONE] report -> {out_root / 'screening_report.md'}")


if __name__ == "__main__":
    main()
