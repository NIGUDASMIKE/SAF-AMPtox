from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

AMP_SPLIT_DIR = REPO_ROOT / "data--final" / "splits"
TOX_SPLIT_DIR = REPO_ROOT / "data--final" / "tox" / "splits"

FEATURE_OUTPUT_ROOT = Path(os.environ.get("CCD_FEATURE_OUTPUT_ROOT", REPO_ROOT / "data--final" / "feature_physchem"))
REPORT_DIR = FEATURE_OUTPUT_ROOT / "reports"

AMP_SPLITS = {
    "train": AMP_SPLIT_DIR / "train.csv",
    "val": AMP_SPLIT_DIR / "val.csv",
    "test": AMP_SPLIT_DIR / "test.csv",
    "test_hard_amp": AMP_SPLIT_DIR / "test_hard_amp.csv",
}

TOX_SPLITS = {
    "train": TOX_SPLIT_DIR / "train.csv",
    "val": TOX_SPLIT_DIR / "val.csv",
    "test": TOX_SPLIT_DIR / "test.csv",
    "test_hard_tox": TOX_SPLIT_DIR / "test_hard_tox.csv",
}

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
RANDOM_SEEDS = (7, 13, 29, 47, 101)

DEFAULT_FEATURE_GROUPS = (
    "PHYS8",
    "AAC",
    "DPC",
    "CTDC",
    "CTDT",
    "CTDD",
)

FEATURE_TABLE_FORMAT = "parquet"
FEATURE_TABLE_COMPRESSION = "zstd"

QUICK_SCREEN_SEEDS = (13,)
REFINED_SEEDS = (7, 13, 29, 47, 101)
TOPK_CANDIDATES = (64, 128, 256, 512, 768, 1024, 1536)

DEFAULT_TASKS = ("amp", "tox")
DEFAULT_SPLITS = ("train", "val", "test")

DEFAULT_MODEL = "lightgbm"

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "objective": "binary",
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}
