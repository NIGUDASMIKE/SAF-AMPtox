from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = REPO_ROOT / "src" / "feature"
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from train_fusion_models import build_model  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "data--final" / "fusion_models_main_checkpoint_export" / "checkpoints"
DEFAULT_CCD = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "features" / "candidate_ccd_features.parquet"
DEFAULT_ESM = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "features" / "candidate_esm2_features.parquet"
DEFAULT_OUT = REPO_ROOT / "data--final" / "generative_sft_evodiff" / "screening" / "candidate_fusion_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated candidates with the residual CCD/ESM fusion predictor ensemble.")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--ccd-parquet", default=str(DEFAULT_CCD))
    parser.add_argument("--esm-parquet", default=str(DEFAULT_ESM))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT))
    parser.add_argument("--model-name", default="cross_attention_residual")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(value)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_paths(root: Path, task: str, model_name: str) -> list[Path]:
    paths = sorted(root.glob(f"{task}__{model_name}__seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for task={task} model={model_name} under {root}")
    return paths


def load_feature_tables(ccd_path: Path, esm_path: Path) -> pd.DataFrame:
    ccd = pd.read_parquet(ccd_path)
    esm = pd.read_parquet(esm_path)
    if "candidate_id" not in ccd.columns and "fasta_id" in ccd.columns:
        ccd = ccd.rename(columns={"fasta_id": "candidate_id"})
    if "candidate_id" not in esm.columns and "fasta_id" in esm.columns:
        esm = esm.rename(columns={"fasta_id": "candidate_id"})
    if "candidate_id" not in ccd.columns or "candidate_id" not in esm.columns:
        raise ValueError("Both CCD and ESM feature tables must contain candidate_id or fasta_id")
    esm_cols = [column for column in esm.columns if column.startswith("esm2_")]
    meta_cols = [column for column in ["candidate_id", "fasta_id", "sequence", "length", "candidate_source"] if column in ccd.columns]
    merged = ccd.merge(
        esm[["candidate_id", *esm_cols]],
        on="candidate_id",
        how="left",
        sort=False,
        suffixes=("", "_esm"),
    )
    if merged[esm_cols].isnull().any().any():
        raise RuntimeError("Missing ESM-2 features after candidate merge")
    return merged.loc[:, meta_cols + [column for column in merged.columns if column not in set(meta_cols)]].copy()


def scaler_transform(x: np.ndarray, scaler_state: dict) -> np.ndarray:
    mean = np.asarray(scaler_state["mean"], dtype=np.float32)
    std = np.asarray(scaler_state["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def predict_checkpoint(frame: pd.DataFrame, checkpoint: dict, device: torch.device, batch_size: int) -> np.ndarray:
    ccd_features = checkpoint["ccd_features"]
    esm_features = checkpoint["esm_features"]
    missing_ccd = sorted(set(ccd_features).difference(frame.columns))
    missing_esm = sorted(set(esm_features).difference(frame.columns))
    if missing_ccd:
        raise RuntimeError(f"Candidate CCD table missing features: {missing_ccd[:20]}")
    if missing_esm:
        raise RuntimeError(f"Candidate ESM table missing features: {missing_esm[:20]}")

    ccd = frame.loc[:, ccd_features].to_numpy(dtype=np.float32)
    esm = frame.loc[:, esm_features].to_numpy(dtype=np.float32)
    ccd = scaler_transform(ccd, checkpoint["ccd_scaler"])
    esm = scaler_transform(esm, checkpoint["esm_scaler"])

    hyper = checkpoint["model_hyperparameters"]
    args = SimpleNamespace(
        hidden_dim=int(hyper["hidden_dim"]),
        d_model=int(hyper["d_model"]),
        heads=int(hyper["heads"]),
        esm_tokens=int(hyper["esm_tokens"]),
        dropout=float(hyper["dropout"]),
    )
    model = build_model(
        checkpoint["model_name"],
        ccd_dim=ccd.shape[1],
        esm_dim=esm.shape[1],
        group_dims=list(checkpoint["group_dims"]),
        args=args,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ccd), torch.from_numpy(esm)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for ccd_batch, esm_batch in loader:
            logits = model(ccd_batch.to(device, non_blocking=True), esm_batch.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_feature_tables(Path(args.ccd_parquet), Path(args.esm_parquet))
    score_frame = frame[[column for column in ["candidate_id", "fasta_id", "sequence", "length", "candidate_source"] if column in frame.columns]].copy()

    manifest: dict[str, object] = {
        "checkpoint_root": str(checkpoint_root.resolve()),
        "ccd_parquet": str(Path(args.ccd_parquet).resolve()),
        "esm_parquet": str(Path(args.esm_parquet).resolve()),
        "device": str(device),
        "n_candidates": int(frame.shape[0]),
        "tasks": {},
    }
    for task, output_col in [("amp", "amp_score"), ("tox", "tox_score")]:
        task_paths = checkpoint_paths(checkpoint_root, task, args.model_name)
        task_probs = []
        seeds = []
        for path in task_paths:
            checkpoint = load_checkpoint(path, device)
            seed = checkpoint.get("seed", path.stem.rsplit("seed", 1)[-1])
            probs = predict_checkpoint(frame, checkpoint, device, args.batch_size)
            seed_col = f"{output_col}_seed{seed}"
            score_frame[seed_col] = probs
            task_probs.append(probs)
            seeds.append(int(seed))
            print(f"[SCORE] {task} seed={seed} n={len(probs)}")
        stacked = np.vstack(task_probs)
        score_frame[output_col] = stacked.mean(axis=0)
        score_frame[f"{output_col}_std"] = stacked.std(axis=0)
        manifest["tasks"][task] = {
            "checkpoints": [str(path.resolve()) for path in task_paths],
            "seeds": seeds,
        }

    score_frame["amp_safe_joint_score"] = np.sqrt(
        np.clip(score_frame["amp_score"], 0.0, 1.0) * np.clip(1.0 - score_frame["tox_score"], 0.0, 1.0)
    )
    score_frame = score_frame.sort_values("amp_safe_joint_score", ascending=False).reset_index(drop=True)
    score_frame["amp_safe_rank"] = np.arange(1, len(score_frame) + 1)
    score_frame.to_csv(out_path, index=False)

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] candidate fusion scores -> {out_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
