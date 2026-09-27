from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    AMP_SPLITS,
    FEATURE_OUTPUT_ROOT,
    FEATURE_TABLE_COMPRESSION,
    FEATURE_TABLE_FORMAT,
    REPORT_DIR,
    TOX_SPLITS,
    VALID_AA,
)


METADATA_COLUMNS = (
    "fasta_id",
    "sequence",
    "length",
    "label",
    "label_id",
    "task_label",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_splits(task: str) -> dict[str, Path]:
    task_lower = task.lower()
    if task_lower == "amp":
        return AMP_SPLITS
    if task_lower == "tox":
        return TOX_SPLITS
    raise ValueError(f"Unsupported task: {task}")


def load_split(task: str, split: str) -> pd.DataFrame:
    path = get_task_splits(task)[split]
    df = pd.read_csv(path)
    if "label_id" not in df.columns:
        raise ValueError(f"{path} is missing label_id")
    if "length" not in df.columns:
        df["length"] = df["sequence"].astype(str).str.len()
    if "task_label" not in df.columns:
        df["task_label"] = task.lower()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    bad_mask = ~df["sequence"].str.fullmatch(f"[{VALID_AA}]+")
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        raise ValueError(f"{path} contains {bad_count} sequences with non-standard residues")
    return df


def output_dir_for(task: str, group: str) -> Path:
    return ensure_dir(FEATURE_OUTPUT_ROOT / task.lower() / group)


def feature_path(task: str, group: str, split: str) -> Path:
    suffix = "parquet" if FEATURE_TABLE_FORMAT == "parquet" else "csv"
    return output_dir_for(task, group) / f"feat_{group}_{split}.{suffix}"


def report_path(name: str) -> Path:
    return ensure_dir(REPORT_DIR) / name


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False, compression=FEATURE_TABLE_COMPRESSION)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table suffix: {path.suffix}")


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in df.columns]
    return df.loc[:, columns].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def combine_feature_frames(metadata: pd.DataFrame, feature_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = metadata.copy()
    for frame in feature_frames:
        drop_columns = [column for column in METADATA_COLUMNS if column in frame.columns]
        combined = pd.concat([combined, frame.drop(columns=drop_columns)], axis=1)
    return combined
