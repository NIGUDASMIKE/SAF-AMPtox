from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_FEATURE_JSON = REPO_ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_feature_union.json"
DEFAULT_POOL = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "sft_evodiff_plus_md_lead_5000.csv"
DEFAULT_OUT = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "features" / "candidate_ccd_features.parquet"


def feature_code_dir() -> Path:
    feature_root = REPO_ROOT / "src" / "feature"
    for child in feature_root.iterdir():
        if child.is_dir() and (child / "feature_registry.py").exists() and (child / "extractors.py").exists():
            return child
    raise FileNotFoundError("Could not locate physicochemical feature code directory under src/feature")


FEATURE_DIR = feature_code_dir()
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from extractors import extract_feature_group  # noqa: E402
from feature_registry import FEATURE_SPEC_BY_NAME  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected main-line CCD features for generated candidates.")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_POOL))
    parser.add_argument("--feature-json", default=str(DEFAULT_FEATURE_JSON))
    parser.add_argument("--out-parquet", default=str(DEFAULT_OUT))
    parser.add_argument("--id-col", default="candidate_id")
    parser.add_argument("--sequence-col", default="sequence")
    return parser.parse_args()


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def load_candidates(path: Path, id_col: str, sequence_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {id_col, sequence_col}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["fasta_id"] = out[id_col].astype(str)
    out["sequence"] = out[sequence_col].map(clean_sequence)
    out["length"] = out["sequence"].str.len()
    bad = ~out["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
    bad |= ~out["length"].between(10, 50)
    if bad.any():
        raise ValueError(f"{path} contains {int(bad.sum())} invalid candidate sequences")
    if out["fasta_id"].duplicated().any():
        raise ValueError("candidate_id values must be unique")
    return out


def group_from_feature(feature: str) -> str:
    prefix = feature.split("_", 1)[0]
    if prefix == "DP":
        return "DistancePair"
    return prefix


def required_groups(features: list[str]) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for feature in features:
        group = group_from_feature(feature)
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def main() -> None:
    args = parse_args()
    candidate_csv = Path(args.candidate_csv)
    feature_json = Path(args.feature_json)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = json.loads(feature_json.read_text(encoding="utf-8"))
    candidates = load_candidates(candidate_csv, args.id_col, args.sequence_col)
    records = list(candidates[["fasta_id", "sequence"]].itertuples(index=False, name=None))

    metadata_cols = [args.id_col, "fasta_id", "sequence", "length"]
    if "candidate_source" in candidates.columns:
        metadata_cols.append("candidate_source")
    combined = candidates.loc[:, metadata_cols].copy()

    extracted_groups = {}
    for group in required_groups(features):
        if group not in FEATURE_SPEC_BY_NAME:
            raise ValueError(f"No feature spec found for required group {group}")
        spec = FEATURE_SPEC_BY_NAME[group]
        frame = extract_feature_group(
            group,
            candidates["sequence"].tolist(),
            backend=spec.backend,
            extractor_key=spec.extractor_key or group,
            fasta_ids=[record[0] for record in records],
        ).reset_index(drop=True)
        extracted_groups[group] = int(frame.shape[1])
        combined = pd.concat([combined, frame], axis=1)
        print(f"[CCD] {group}: {frame.shape[1]} features")

    missing_features = sorted(set(features).difference(combined.columns))
    if missing_features:
        raise RuntimeError(f"Missing selected CCD columns: {missing_features[:20]}")

    output = combined.loc[:, metadata_cols + features].copy()
    output.to_parquet(out_path, index=False, compression="zstd")

    manifest = {
        "candidate_csv": str(candidate_csv.resolve()),
        "feature_json": str(feature_json.resolve()),
        "out_parquet": str(out_path.resolve()),
        "n_candidates": int(output.shape[0]),
        "n_selected_features": int(len(features)),
        "extracted_groups": extracted_groups,
        "feature_code_dir": str(FEATURE_DIR),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] candidate CCD features -> {out_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
