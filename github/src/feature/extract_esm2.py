from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "src" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from discover_final_benchmark import choose_latest, find_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean-pooled ESM-2 embeddings for final AMP/TOX benchmark splits.")
    parser.add_argument("--amp-dir", default=None)
    parser.add_argument("--tox-dir", default=None)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data--final" / "feature_esm2"))
    parser.add_argument("--model", default="esm2_t12_35M_UR50D", choices=["esm2_t12_35M_UR50D"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--include-hard", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def discover_dirs(amp_dir: str | None, tox_dir: str | None) -> tuple[Path, Path]:
    selected = choose_latest(find_candidates([REPO_ROOT / "data--final", REPO_ROOT / "data" / "amp"]))
    amp_path = Path(amp_dir).resolve() if amp_dir else Path(selected["amp"].path).resolve()
    tox_path = Path(tox_dir).resolve() if tox_dir else Path(selected["tox"].path).resolve()
    return amp_path, tox_path


def task_splits(include_hard: bool) -> dict[str, list[str]]:
    splits = {
        "amp": ["train", "val", "test"],
        "tox": ["train", "val", "test"],
    }
    if include_hard:
        splits["amp"].append("test_hard_amp")
        splits["tox"].append("test_hard_tox")
    return splits


def load_split(bench_dir: Path, split: str) -> pd.DataFrame:
    path = bench_dir / "splits" / f"{split}.csv"
    df = pd.read_csv(path)
    required = {"fasta_id", "sequence", "label_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    return df


def load_membership(amp_dir: Path, tox_dir: Path, include_hard: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dirs = {"amp": amp_dir, "tox": tox_dir}
    split_frames: dict[str, pd.DataFrame] = {}
    membership_rows: list[pd.DataFrame] = []
    for task, splits in task_splits(include_hard).items():
        for split in splits:
            frame = load_split(dirs[task], split)
            frame = frame.copy()
            frame["task"] = task
            frame["split"] = split
            split_frames[f"{task}:{split}"] = frame
            membership_rows.append(frame[["fasta_id", "sequence", "task", "split", "label_id"]])
    membership = pd.concat(membership_rows, axis=0, ignore_index=True)
    return membership, split_frames


def unique_records(membership: pd.DataFrame) -> list[tuple[str, str]]:
    unique = membership.drop_duplicates("fasta_id", keep="first")
    duplicated = unique["fasta_id"].duplicated().sum()
    if duplicated:
        raise RuntimeError("Unexpected duplicated fasta_id after drop_duplicates")
    sequence_conflicts = membership.groupby("fasta_id")["sequence"].nunique()
    conflicts = sequence_conflicts[sequence_conflicts > 1]
    if not conflicts.empty:
        raise RuntimeError(f"fasta_id sequence conflicts detected: {conflicts.index[:5].tolist()}")
    return list(unique[["fasta_id", "sequence"]].itertuples(index=False, name=None))


def dependency_status() -> dict[str, bool]:
    import importlib.util

    return {
        "torch": bool(importlib.util.find_spec("torch")),
        "esm": bool(importlib.util.find_spec("esm")),
    }


def load_esm_model(model_name: str, device_arg: str):
    status = dependency_status()
    if not all(status.values()):
        missing = [name for name, available in status.items() if not available]
        raise ImportError(
            "Missing ESM-2 dependencies: "
            + ", ".join(missing)
            + ". Install PyTorch and fair-esm in the execution environment before running extraction."
        )

    import torch
    import esm

    if model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        repr_layer = 12
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    model.eval()
    model.to(device)
    return model, alphabet, repr_layer, device, torch


def extract_embeddings(records: list[tuple[str, str]], model, alphabet, repr_layer: int, device: str, torch, batch_size: int) -> pd.DataFrame:
    batch_converter = alphabet.get_batch_converter()
    rows: list[pd.DataFrame] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        _, sequences, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            result = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        representations = result["representations"][repr_layer].detach().cpu().numpy()
        vectors = []
        for index, sequence in enumerate(sequences):
            vectors.append(representations[index, 1:len(sequence) + 1, :].mean(axis=0))
        vector_array = np.asarray(vectors, dtype=np.float32)
        feature_columns = [f"esm2_{i:04d}" for i in range(vector_array.shape[1])]
        batch_df = pd.DataFrame(vector_array, columns=feature_columns)
        batch_df.insert(0, "sequence", [sequence for _, sequence in batch])
        batch_df.insert(0, "fasta_id", [fasta_id for fasta_id, _ in batch])
        rows.append(batch_df)
        print(f"[ESM2] {min(start + batch_size, len(records))}/{len(records)} sequences")
    return pd.concat(rows, axis=0, ignore_index=True)


def write_split_aligned_features(features: pd.DataFrame, split_frames: dict[str, pd.DataFrame], output_root: Path) -> dict[str, str]:
    feature_cols = [column for column in features.columns if column.startswith("esm2_")]
    written: dict[str, str] = {}
    for key, split_df in split_frames.items():
        task, split = key.split(":", 1)
        out_dir = output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = split_df[["fasta_id", "sequence", "label_id", "task", "split"]].merge(
            features[["fasta_id", *feature_cols]],
            on="fasta_id",
            how="left",
            sort=False,
        )
        if merged[feature_cols].isnull().any().any():
            raise RuntimeError(f"Missing ESM features after alignment for {key}")
        out_path = out_dir / f"esm2_{task}_{split}.parquet"
        merged.to_parquet(out_path, index=False, compression="zstd")
        written[key] = str(out_path)
    return written


def main() -> None:
    args = parse_args()
    amp_dir, tox_dir = discover_dirs(args.amp_dir, args.tox_dir)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    status = dependency_status()
    status_path = reports_dir / "esm2_dependency_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[ESM2] dependency status -> {status_path}: {status}")
    if args.check_only:
        return

    membership, split_frames = load_membership(amp_dir, tox_dir, include_hard=args.include_hard)
    membership_path = reports_dir / "esm2_feature_membership.csv"
    membership.to_csv(membership_path, index=False)
    records = unique_records(membership)

    output_path = output_root / "esm2_features.parquet"
    if output_path.exists() and not args.overwrite:
        features = pd.read_parquet(output_path)
        print(f"[ESM2] Reusing existing features -> {output_path}")
    else:
        model, alphabet, repr_layer, device, torch = load_esm_model(args.model, args.device)
        print(f"[ESM2] model={args.model} repr_layer={repr_layer} device={device} unique_sequences={len(records)}")
        features = extract_embeddings(records, model, alphabet, repr_layer, device, torch, batch_size=args.batch_size)
        features.to_parquet(output_path, index=False, compression="zstd")

    split_paths = write_split_aligned_features(features, split_frames, output_root)
    manifest = {
        "model": args.model,
        "amp_dir": str(amp_dir),
        "tox_dir": str(tox_dir),
        "include_hard": args.include_hard,
        "n_unique_sequences": len(records),
        "feature_table": str(output_path),
        "membership_csv": str(membership_path),
        "split_feature_tables": split_paths,
        "dependency_status": status,
    }
    manifest_path = reports_dir / "esm2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] ESM-2 features -> {output_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
