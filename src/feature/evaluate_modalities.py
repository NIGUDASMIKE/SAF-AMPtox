from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = {"fasta_id", "sequence", "length", "label", "label_id", "task_label", "task", "split"}


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
    "verbosity": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CCD-only, ESM2-only, and CCD+ESM2 concatenation baselines.")
    parser.add_argument("--ccd-root", default=str(REPO_ROOT / "data--final" / "feature_physchem"))
    parser.add_argument("--ccd-union-subdir", default="union_final_groups")
    parser.add_argument("--ccd-feature-json", default=str(REPO_ROOT / "data--final" / "feature_physchem" / "reports" / "final_lightgbm_compact_feature_union.json"))
    parser.add_argument("--esm-root", default=str(REPO_ROOT / "data--final" / "feature_esm2"))
    parser.add_argument("--report-dir", default=str(REPO_ROOT / "data--final" / "model_baselines" / "reports"))
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--modalities", nargs="+", default=["ccd", "esm2", "concat"], choices=["ccd", "esm2", "concat"])
    return parser.parse_args()


def hard_split(task: str) -> str:
    return "test_hard_amp" if task == "amp" else "test_hard_tox"


def ccd_path(root: Path, task: str, union_subdir: str, split: str) -> Path:
    return root / task / union_subdir / f"{union_subdir}_{task}_{split}.parquet"


def esm_path(root: Path, task: str, split: str) -> Path:
    return root / task / f"esm2_{task}_{split}.parquet"


def load_ccd(root: Path, task: str, union_subdir: str, split: str, features: list[str]) -> pd.DataFrame:
    path = ccd_path(root, task, union_subdir, split)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path} missing CCD features: {missing[:10]}")
    keep_meta = [column for column in frame.columns if column in METADATA_COLUMNS]
    return frame.loc[:, keep_meta + features].copy()


def load_esm(root: Path, task: str, split: str) -> pd.DataFrame:
    path = esm_path(root, task, split)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    feature_cols = [column for column in frame.columns if column.startswith("esm2_")]
    keep_meta = [column for column in frame.columns if column in METADATA_COLUMNS]
    return frame.loc[:, keep_meta + feature_cols].copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def merge_modalities(ccd: pd.DataFrame, esm: pd.DataFrame) -> pd.DataFrame:
    ccd_features = feature_columns(ccd)
    esm_features = feature_columns(esm)
    left = ccd[["fasta_id", "sequence", "label_id", *ccd_features]].copy()
    right = esm[["fasta_id", *esm_features]].copy()
    merged = left.merge(right, on="fasta_id", how="left", sort=False)
    if merged[esm_features].isnull().any().any():
        raise RuntimeError("Missing ESM features after CCD/ESM merge")
    return merged


def build_model(random_state: int) -> LGBMClassifier:
    params = dict(LGBM_PARAMS)
    params["random_state"] = random_state
    return LGBMClassifier(**params)


def evaluate(model: LGBMClassifier, frame: pd.DataFrame) -> dict[str, float]:
    features = feature_columns(frame)
    y = frame["label_id"].to_numpy(dtype=np.int64)
    x = frame.loc[:, features].astype(np.float32)
    y_prob = model.predict_proba(x)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }


def load_modality_split(
    modality: str,
    ccd_root: Path,
    ccd_union_subdir: str,
    ccd_features: list[str],
    esm_root: Path,
    task: str,
    split: str,
) -> pd.DataFrame:
    if modality == "ccd":
        return load_ccd(ccd_root, task, ccd_union_subdir, split, ccd_features)
    if modality == "esm2":
        return load_esm(esm_root, task, split)
    if modality == "concat":
        ccd = load_ccd(ccd_root, task, ccd_union_subdir, split, ccd_features)
        esm = load_esm(esm_root, task, split)
        return merge_modalities(ccd, esm)
    raise ValueError(f"Unsupported modality: {modality}")


def main() -> None:
    args = parse_args()
    ccd_root = Path(args.ccd_root).resolve()
    esm_root = Path(args.esm_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    ccd_features = json.loads(Path(args.ccd_feature_json).read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "ccd_root": str(ccd_root),
        "ccd_union_subdir": args.ccd_union_subdir,
        "ccd_feature_json": str(Path(args.ccd_feature_json).resolve()),
        "esm_root": str(esm_root),
        "random_state": args.random_state,
        "modalities": args.modalities,
    }
    for task in ("amp", "tox"):
        eval_splits = ["val", "test", hard_split(task)]
        for modality in args.modalities:
            train = load_modality_split(modality, ccd_root, args.ccd_union_subdir, ccd_features, esm_root, task, "train")
            features = feature_columns(train)
            model = build_model(args.random_state)
            model.fit(train.loc[:, features].astype(np.float32), train["label_id"].to_numpy(dtype=np.int64))
            for split in eval_splits:
                frame = load_modality_split(modality, ccd_root, args.ccd_union_subdir, ccd_features, esm_root, task, split)
                metrics = evaluate(model, frame)
                row = {
                    "task": task,
                    "modality": modality,
                    "split": split,
                    "n": int(frame.shape[0]),
                    "n_features": len(features),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[OK] {task}/{modality}/{split} n={frame.shape[0]} p={len(features)} "
                    f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
                    f"BA={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f}"
                )

    result = pd.DataFrame(rows)
    csv_path = report_dir / "modality_baseline_results.csv"
    json_path = report_dir / "modality_baseline_results.json"
    manifest_path = report_dir / "modality_baseline_manifest.json"
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest["results_csv"] = str(csv_path)
    manifest["results_json"] = str(json_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] results -> {csv_path}")
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
