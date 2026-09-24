from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data--final" / "feature_physchem"
DEFAULT_FEATURE_JSON = DEFAULT_FEATURE_ROOT / "reports" / "final_lightgbm_compact_feature_union.json"


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
    parser = argparse.ArgumentParser(description="Evaluate selected CCD features on standard and hard test splits.")
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--feature-union-json", default=str(DEFAULT_FEATURE_JSON))
    parser.add_argument("--union-subdir", default="union")
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--report-prefix", default="ccd_hard_test_eval")
    return parser.parse_args()


def union_path(feature_root: Path, task: str, union_subdir: str, split: str) -> Path:
    return feature_root / task / union_subdir / f"{union_subdir}_{task}_{split}.parquet"


def load_union(feature_root: Path, task: str, union_subdir: str, split: str) -> pd.DataFrame:
    path = union_path(feature_root, task, union_subdir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing union matrix: {path}")
    return pd.read_parquet(path)


def build_model(random_state: int) -> LGBMClassifier:
    params = dict(LGBM_PARAMS)
    params["random_state"] = random_state
    return LGBMClassifier(**params)


def evaluate(model: LGBMClassifier, frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).resolve()
    feature_union_json = Path(args.feature_union_json).resolve()
    features = json.loads(feature_union_json.read_text(encoding="utf-8"))

    split_map = {
        "amp": ["val", "test", "test_hard_amp"],
        "tox": ["val", "test", "test_hard_tox"],
    }
    rows: list[dict[str, object]] = []
    for task, eval_splits in split_map.items():
        train = load_union(feature_root, task, args.union_subdir, "train")
        missing = sorted(set(features).difference(train.columns))
        if missing:
            raise RuntimeError(f"{task} train matrix is missing selected features: {missing[:10]}")
        model = build_model(args.random_state)
        model.fit(train.loc[:, features].astype(np.float32), train["label_id"].to_numpy(dtype=np.int64))
        for split in eval_splits:
            frame = load_union(feature_root, task, args.union_subdir, split)
            metrics = evaluate(model, frame, features)
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "n": int(frame.shape[0]),
                    "n_features": len(features),
                    **metrics,
                }
            )
            print(
                f"[OK] {task}/{split} n={frame.shape[0]} "
                f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
                f"BA={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f}"
            )

    report_dir = feature_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{args.report_prefix}.csv"
    json_path = report_dir / f"{args.report_prefix}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[DONE] CSV -> {csv_path}")
    print(f"[DONE] JSON -> {json_path}")


if __name__ == "__main__":
    main()
