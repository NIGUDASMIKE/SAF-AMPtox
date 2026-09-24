from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from discover_final_benchmark import REPO_ROOT, choose_latest, find_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced hard test subsets for AMP M-start and TOX cysteine shortcuts.")
    parser.add_argument("--amp-dir", default=None)
    parser.add_argument("--tox-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--tox-bin-candidates", nargs="+", type=int, default=[20, 16, 12, 10, 8, 5])
    parser.add_argument("--out-report", default=str(REPO_ROOT / "data--final" / "reports" / "hard_test_summary.json"))
    return parser.parse_args()


def discover_dirs(amp_dir: str | None, tox_dir: str | None) -> tuple[Path, Path]:
    selected = choose_latest(find_candidates([REPO_ROOT / "data--final", REPO_ROOT / "data" / "amp"]))
    amp_path = Path(amp_dir).resolve() if amp_dir else Path(selected["amp"].path).resolve()
    tox_path = Path(tox_dir).resolve() if tox_dir else Path(selected["tox"].path).resolve()
    return amp_path, tox_path


def load_test(bench_dir: Path) -> pd.DataFrame:
    path = bench_dir / "splits" / "test.csv"
    df = pd.read_csv(path)
    if "sequence" not in df.columns or "label_id" not in df.columns:
        raise ValueError(f"{path} must contain sequence and label_id columns")
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    df["label_id"] = df["label_id"].astype(int)
    return df


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if df.shape[0] <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def build_amp_hard(test_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    work = test_df.copy()
    work["_starts_m"] = work["sequence"].str.startswith("M")

    selected_parts: list[pd.DataFrame] = []
    stratum_report: list[dict[str, object]] = []
    for starts_m, stratum in work.groupby("_starts_m"):
        pos = stratum[stratum["label_id"] == 1]
        neg = stratum[stratum["label_id"] == 0]
        n = min(pos.shape[0], neg.shape[0])
        selected_parts.append(sample_rows(pos, n, seed + int(starts_m) + 11))
        selected_parts.append(sample_rows(neg, n, seed + int(starts_m) + 29))
        stratum_report.append(
            {
                "starts_with_M": bool(starts_m),
                "positive_available": int(pos.shape[0]),
                "negative_available": int(neg.shape[0]),
                "selected_per_class": int(n),
            }
        )

    hard = pd.concat(selected_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if hard.empty:
        raise RuntimeError("AMP hard test construction failed; no balanced M-start strata were available.")

    original_columns = [column for column in test_df.columns if column in hard.columns]
    hard = hard.loc[:, original_columns].copy()
    pos_hard = hard[hard["label_id"] == 1]
    neg_hard = hard[hard["label_id"] == 0]
    report = {
        "n_total": int(hard.shape[0]),
        "n_positive": int(pos_hard.shape[0]),
        "n_negative": int(neg_hard.shape[0]),
        "positive_starts_M_proportion": float(pos_hard["sequence"].str.startswith("M").mean()),
        "negative_starts_M_proportion": float(neg_hard["sequence"].str.startswith("M").mean()),
        "strata": stratum_report,
    }
    return hard, report


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float)
    q = q.astype(float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def tox_match_with_bins(work: pd.DataFrame, bins: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    work = work.copy()
    # Keep true zero-cysteine peptides in a dedicated stratum. Otherwise a coarse
    # first bin would mix C-free negatives with low-C positives and preserve the
    # shortcut we are trying to neutralize.
    nonzero_mask = work["_c_frequency"] > 0
    work["_c_bin"] = -1
    work.loc[nonzero_mask, "_c_bin"] = pd.cut(
        work.loc[nonzero_mask, "_c_frequency"],
        bins=edges,
        include_lowest=False,
        labels=False,
    ).astype("Int64")
    selected_parts: list[pd.DataFrame] = []
    bin_rows: list[dict[str, object]] = []
    for bin_id, stratum in work.groupby("_c_bin", dropna=False):
        pos = stratum[stratum["label_id"] == 1]
        neg = stratum[stratum["label_id"] == 0]
        n = min(pos.shape[0], neg.shape[0])
        if n > 0:
            selected_parts.append(sample_rows(pos, n, seed + int(bin_id) + 101))
            selected_parts.append(sample_rows(neg, n, seed + int(bin_id) + 409))
        bin_rows.append(
            {
                "bin": int(bin_id) if pd.notna(bin_id) else -1,
                "positive_available": int(pos.shape[0]),
                "negative_available": int(neg.shape[0]),
                "selected_per_class": int(n),
            }
        )
    if selected_parts:
        hard = pd.concat(selected_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        hard = work.iloc[0:0].copy()

    histogram_edges = np.concatenate(([-1e-12], edges[1:]))
    pos_hist, _ = np.histogram(hard.loc[hard["label_id"] == 1, "_c_frequency"], bins=histogram_edges)
    neg_hist, _ = np.histogram(hard.loc[hard["label_id"] == 0, "_c_frequency"], bins=histogram_edges)
    tv_distance = float(0.5 * np.abs((pos_hist / max(pos_hist.sum(), 1)) - (neg_hist / max(neg_hist.sum(), 1))).sum())
    report = {
        "bins": bins,
        "n_total": int(hard.shape[0]),
        "n_positive": int((hard["label_id"] == 1).sum()) if not hard.empty else 0,
        "n_negative": int((hard["label_id"] == 0).sum()) if not hard.empty else 0,
        "jensen_shannon_distance": jensen_shannon_distance(pos_hist, neg_hist) if hard.shape[0] > 0 else 1.0,
        "total_variation_distance": tv_distance,
        "bin_report": bin_rows,
    }
    return hard, report


def build_tox_hard(test_df: pd.DataFrame, bin_candidates: list[int], seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    work = test_df.copy()
    c_count = work["sequence"].str.count("C")
    work["_c_count"] = c_count
    work["_c_frequency"] = c_count / work["sequence"].str.len().clip(lower=1)

    attempts: list[dict[str, object]] = []
    best_hard: pd.DataFrame | None = None
    best_report: dict[str, object] | None = None
    for bins in bin_candidates:
        hard, report = tox_match_with_bins(work, bins=bins, seed=seed)
        attempts.append(report)
        if best_report is None:
            best_hard, best_report = hard, report
            continue
        current_key = (
            -report["jensen_shannon_distance"],
            -report["total_variation_distance"],
            report["n_total"],
        )
        best_key = (
            -best_report["jensen_shannon_distance"],
            -best_report["total_variation_distance"],
            best_report["n_total"],
        )
        if current_key > best_key:
            best_hard, best_report = hard, report

    if best_hard is None or best_report is None or best_hard.empty:
        raise RuntimeError("TOX hard test construction failed; no overlapping C-frequency bins were available.")

    original_columns = [column for column in test_df.columns if column in best_hard.columns]
    hard = best_hard.loc[:, original_columns].copy()
    pos = best_hard[best_hard["label_id"] == 1]
    neg = best_hard[best_hard["label_id"] == 0]
    best_report = dict(best_report)
    best_report.update(
        {
            "positive_contains_C_proportion": float((pos["_c_count"] > 0).mean()),
            "negative_contains_C_proportion": float((neg["_c_count"] > 0).mean()),
            "positive_mean_C_frequency": float(pos["_c_frequency"].mean()),
            "negative_mean_C_frequency": float(neg["_c_frequency"].mean()),
            "attempts": attempts,
        }
    )
    return hard, best_report


def main() -> None:
    args = parse_args()
    amp_dir, tox_dir = discover_dirs(args.amp_dir, args.tox_dir)

    amp_test = load_test(amp_dir)
    tox_test = load_test(tox_dir)
    amp_hard, amp_report = build_amp_hard(amp_test, args.seed)
    tox_hard, tox_report = build_tox_hard(tox_test, args.tox_bin_candidates, args.seed)

    amp_out = amp_dir / "splits" / "test_hard_amp.csv"
    tox_out = tox_dir / "splits" / "test_hard_tox.csv"
    amp_hard.to_csv(amp_out, index=False)
    tox_hard.to_csv(tox_out, index=False)

    report = {
        "amp_dir": str(amp_dir),
        "tox_dir": str(tox_dir),
        "seed": args.seed,
        "amp_hard_path": str(amp_out),
        "tox_hard_path": str(tox_out),
        "amp_hard": amp_report,
        "tox_hard": tox_report,
    }
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[AMP HARD]", json.dumps(amp_report, indent=2))
    print("[TOX HARD]", json.dumps(tox_report, indent=2))
    print(f"[DONE] AMP hard -> {amp_out}")
    print(f"[DONE] TOX hard -> {tox_out}")
    print(f"[DONE] report -> {report_path}")


if __name__ == "__main__":
    main()
