from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "data--final" / "generative_evodiff_audit"
TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


GROUPS = {
    "Base EvoDiff": REPO_ROOT / "results" / "group1_zeroshot" / "group1_metrics.csv",
    "SFT EvoDiff": REPO_ROOT / "results" / "group2_sft" / "group2_metrics.csv",
    "DPO EvoDiff": REPO_ROOT / "results" / "group3_dpo" / "group3_metrics.csv",
    "HITL-DPO EvoDiff": REPO_ROOT / "results" / "group4_hitl" / "group4_metrics.csv",
}


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
    if not seq:
        return 0
    best = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def composition_features(seq: str) -> dict[str, float | int | bool]:
    length = len(seq)
    counts = Counter(seq)
    top_freq = max(counts.values()) / length if length else 0.0
    top2_freq = sum(v for _, v in counts.most_common(2)) / length if length else 0.0
    aromatic_fraction = sum(counts.get(aa, 0) for aa in "FWY") / length if length else 0.0
    gly_phe_fraction = (counts.get("G", 0) + counts.get("F", 0)) / length if length else 0.0
    basic_fraction = (counts.get("K", 0) + counts.get("R", 0) + counts.get("H", 0)) / length if length else 0.0
    acidic_fraction = (counts.get("D", 0) + counts.get("E", 0)) / length if length else 0.0
    c_fraction = counts.get("C", 0) / length if length else 0.0
    return {
        "length": length,
        "valid_standard_aa": bool(seq and not (set(seq) - STANDARD_AA)),
        "m_start": bool(seq.startswith("M")),
        "contains_c": bool("C" in seq),
        "c_fraction": c_fraction,
        "net_charge_simple": counts.get("K", 0) + counts.get("R", 0) + 0.1 * counts.get("H", 0) - counts.get("D", 0) - counts.get("E", 0),
        "top1_aa_fraction": top_freq,
        "top2_aa_fraction": top2_freq,
        "aromatic_fraction": aromatic_fraction,
        "gly_phe_fraction": gly_phe_fraction,
        "basic_fraction": basic_fraction,
        "acidic_fraction": acidic_fraction,
        "entropy_norm": shannon_entropy_norm(seq),
        "longest_homopolymer_run": longest_run(seq),
        "low_complexity_flag": bool(top_freq >= 0.35 or top2_freq >= 0.60 or shannon_entropy_norm(seq) <= 0.55),
        "homopolymer_flag": bool(longest_run(seq) >= 4),
        "fg_glycine_rich_flag": bool(gly_phe_fraction >= 0.35),
    }


def load_group(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "Sequence" not in df.columns:
        raise ValueError(f"{path} must contain a Sequence column")
    records = []
    for _, row in df.iterrows():
        seq = clean_sequence(row["Sequence"])
        item = {
            "group": name,
            "id": row.get("ID", ""),
            "sequence": seq,
            "oracle_score": float(row.get("Oracle_Score", float("nan"))),
        }
        item.update(composition_features(seq))
        records.append(item)
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby("group", sort=False):
        rows.append(
            {
                "group": group,
                "n": len(sub),
                "valid_rate": sub["valid_standard_aa"].mean(),
                "unique_rate": sub["sequence"].nunique() / len(sub),
                "mean_oracle": sub["oracle_score"].mean(),
                "median_oracle": sub["oracle_score"].median(),
                "oracle_ge_0_9_rate": (sub["oracle_score"] >= 0.9).mean(),
                "mean_length": sub["length"].mean(),
                "m_start_rate": sub["m_start"].mean(),
                "contains_c_rate": sub["contains_c"].mean(),
                "mean_c_fraction": sub["c_fraction"].mean(),
                "mean_simple_charge": sub["net_charge_simple"].mean(),
                "mean_entropy_norm": sub["entropy_norm"].mean(),
                "low_complexity_rate": sub["low_complexity_flag"].mean(),
                "homopolymer_rate": sub["homopolymer_flag"].mean(),
                "fg_glycine_rich_rate": sub["fg_glycine_rich_flag"].mean(),
                "mean_gly_phe_fraction": sub["gly_phe_fraction"].mean(),
                "mean_basic_fraction": sub["basic_fraction"].mean(),
                "mean_aromatic_fraction": sub["aromatic_fraction"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rounded = summary.copy()
    for col in rounded.columns:
        if col not in {"group", "n"}:
            rounded[col] = rounded[col].map(lambda x: f"{float(x):.4f}")
    lines = [
        "# EvoDiff Generative Audit",
        "",
        "## Purpose",
        "",
        "This audit checks whether the legacy EvoDiff generation runs show reward-hacking signatures before we formalize the SFT/DPO experiment.",
        "",
        "## Summary",
        "",
        rounded.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- DPO should be evaluated against SFT and rejection-sampling baselines, not only against the raw base generator.",
        "- Any future preference reward must include hard validity, novelty, diversity, low-complexity, and composition-distribution constraints.",
        "- HITL-DPO-style outputs require special caution if Oracle score increases together with low-complexity or glycine/phenylalanine-rich signatures.",
        "- Final claims must use external judges and non-oracle evidence; the internal discriminator can rank candidates but cannot be the sole proof of success.",
        "",
    ]
    (REPORT_DIR / "existing_evodiff_group_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = [load_group(name, path) for name, path in GROUPS.items()]
    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    detail.to_csv(TABLE_DIR / "existing_group_sequence_qc_detail.csv", index=False)
    summary.to_csv(TABLE_DIR / "existing_group_sequence_qc_summary.csv", index=False)
    write_report(summary)
    print(f"[DONE] detail -> {TABLE_DIR / 'existing_group_sequence_qc_detail.csv'}")
    print(f"[DONE] summary -> {TABLE_DIR / 'existing_group_sequence_qc_summary.csv'}")
    print(f"[DONE] report -> {REPORT_DIR / 'existing_evodiff_group_audit.md'}")


if __name__ == "__main__":
    main()
