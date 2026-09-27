from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = REPO_ROOT / "src" / "feature"
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from extract_esm2 import extract_embeddings, load_esm_model  # noqa: E402


DEFAULT_POOL = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "sft_evodiff_plus_md_lead_5000.csv"
DEFAULT_OUT = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "features" / "candidate_esm2_features.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean-pooled ESM-2 embeddings for generated candidates.")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_POOL))
    parser.add_argument("--out-parquet", default=str(DEFAULT_OUT))
    parser.add_argument("--id-col", default="candidate_id")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--model", default="esm2_t12_35M_UR50D", choices=["esm2_t12_35M_UR50D"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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


def main() -> None:
    args = parse_args()
    candidate_csv = Path(args.candidate_csv)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(candidate_csv, args.id_col, args.sequence_col)
    records = list(candidates[["fasta_id", "sequence"]].itertuples(index=False, name=None))

    model, alphabet, repr_layer, device, torch = load_esm_model(args.model, args.device)
    print(f"[ESM2] model={args.model} repr_layer={repr_layer} device={device} n={len(records)}")
    features = extract_embeddings(records, model, alphabet, repr_layer, device, torch, batch_size=args.batch_size)
    features = features.rename(columns={"fasta_id": args.id_col})
    features.insert(1, "fasta_id", features[args.id_col])
    if "candidate_source" in candidates.columns:
        features = features.merge(candidates[[args.id_col, "candidate_source"]], on=args.id_col, how="left")
    features.to_parquet(out_path, index=False, compression="zstd")

    manifest = {
        "candidate_csv": str(candidate_csv.resolve()),
        "out_parquet": str(out_path.resolve()),
        "model": args.model,
        "device": str(device),
        "n_candidates": int(features.shape[0]),
        "n_esm_features": int(len([column for column in features.columns if column.startswith("esm2_")])),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] candidate ESM-2 features -> {out_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
