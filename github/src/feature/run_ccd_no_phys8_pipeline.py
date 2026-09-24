from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICOCHEMICAL_FEATURE_DIR = REPO_ROOT / "src" / "feature" / "physchem"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data--final" / "feature_ccd_no_phys8"
DEFAULT_SEEDS = ("7", "13", "29", "47", "101")
DEFAULT_TOPK = ("64", "128", "256", "512", "768", "1024", "1536", "2048", "4033")
METADATA_COLUMNS = {"fasta_id", "sequence", "length", "label", "label_id", "task_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PHYS8-free CCD extraction and LightGBM selection pipeline using iFeature-only descriptors."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--stages", nargs="+", default=["extract", "evaluate", "select", "union", "topk", "bundle"])
    parser.add_argument("--quick", action="store_true", help="Use a single seed for a faster smoke run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-hard", action="store_true", help="Also extract/build union matrices for hard test splits.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def load_no_phys8_groups() -> list[str]:
    sys.path.insert(0, str(PHYSICOCHEMICAL_FEATURE_DIR))
    from feature_registry import FEATURE_SPECS  # type: ignore

    return [
        spec.name
        for spec in FEATURE_SPECS
        if spec.backend == "ifeature" and spec.enabled and spec.status == "ready"
    ]


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[CMD] " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def python_script(script_name: str) -> str:
    return str(PHYSICOCHEMICAL_FEATURE_DIR / script_name)


def write_shared_bundle(output_root: Path, report_prefix: str, union_subdir: str, include_hard: bool) -> dict[str, object]:
    feature_json = output_root / "reports" / f"{report_prefix}_feature_union.json"
    selected_features = json.loads(feature_json.read_text(encoding="utf-8"))
    task_splits = {
        "amp": ["train", "val", "test"],
        "tox": ["train", "val", "test"],
    }
    if include_hard:
        task_splits["amp"].append("test_hard_amp")
        task_splits["tox"].append("test_hard_tox")

    frames: list[pd.DataFrame] = []
    written: dict[str, str] = {}
    for task, splits in task_splits.items():
        out_dir = output_root / task / "shared_union"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            in_path = output_root / task / union_subdir / f"{union_subdir}_{task}_{split}.parquet"
            if not in_path.exists():
                raise FileNotFoundError(in_path)
            frame = pd.read_parquet(in_path)
            keep = [column for column in frame.columns if column in METADATA_COLUMNS] + selected_features
            compact = frame.loc[:, keep].copy()
            compact["task"] = task
            compact["split"] = split
            out_path = out_dir / f"ccd_shared_union_{task}_{split}.parquet"
            compact.to_parquet(out_path, index=False, compression="zstd")
            written[f"{task}_{split}"] = str(out_path)
            frames.append(compact)

    all_frame = pd.concat(frames, axis=0, ignore_index=True)
    stable_path = output_root / "ccd_shared_union_features.parquet"
    count_path = output_root / f"ccd_{len(selected_features)}_features.parquet"
    all_frame.to_parquet(stable_path, index=False, compression="zstd")
    all_frame.to_parquet(count_path, index=False, compression="zstd")
    manifest = {
        "n_features": len(selected_features),
        "feature_union_json": str(feature_json),
        "stable_bundle": str(stable_path),
        "count_named_bundle": str(count_path),
        "split_tables": written,
    }
    manifest_path = output_root / "reports" / "ccd_no_phys8_bundle_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] shared CCD bundle -> {stable_path}")
    print(f"[DONE] count-named CCD bundle -> {count_path}")
    return manifest


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    groups = load_no_phys8_groups()
    seeds = ("13",) if args.quick else DEFAULT_SEEDS
    env = os.environ.copy()
    env["CCD_FEATURE_OUTPUT_ROOT"] = str(output_root)

    print(f"[INFO] PHYS8-free iFeature groups: {len(groups)}")
    print("[INFO] " + ", ".join(groups))
    print(f"[INFO] output root: {output_root}")

    if "extract" in args.stages:
        amp_splits = ["train", "val", "test"] + (["test_hard_amp"] if args.include_hard else [])
        tox_splits = ["train", "val", "test"] + (["test_hard_tox"] if args.include_hard else [])
        overwrite = ["--overwrite"] if args.overwrite else []
        run_command(
            [sys.executable, python_script("extract_features.py"), "--tasks", "amp", "--splits", *amp_splits, "--groups", *groups, *overwrite],
            env,
            args.dry_run,
        )
        run_command(
            [sys.executable, python_script("extract_features.py"), "--tasks", "tox", "--splits", *tox_splits, "--groups", *groups, *overwrite],
            env,
            args.dry_run,
        )

    if "evaluate" in args.stages:
        run_command(
            [
                sys.executable,
                python_script("evaluate_single_features.py"),
                "--tasks",
                "amp",
                "tox",
                "--groups",
                *groups,
                "--seeds",
                *seeds,
                "--report-prefix",
                "no_phys8_single_feature_results",
            ],
            env,
            args.dry_run,
        )

    selection_json = output_root / "reports" / "no_phys8_selected_groups_union.json"
    if "select" in args.stages:
        run_command(
            [
                sys.executable,
                python_script("select_union_groups_by_threshold.py"),
                "--summary-csv",
                str(output_root / "reports" / "no_phys8_single_feature_results_summary.csv"),
                "--auto-quantile",
                "0.75",
                "--min-roc",
                "0.80",
                "--min-pr",
                "0.80",
                "--report-prefix",
                "no_phys8_selected_groups_union",
            ],
            env,
            args.dry_run,
        )

    union_subdir = "no_phys8_union"
    if "union" in args.stages:
        amp_splits = ["train", "val", "test"] + (["test_hard_amp"] if args.include_hard else [])
        tox_splits = ["train", "val", "test"] + (["test_hard_tox"] if args.include_hard else [])
        run_command(
            [
                sys.executable,
                python_script("build_union_matrix.py"),
                "--selection-json",
                str(selection_json),
                "--tasks",
                "amp",
                "--splits",
                *amp_splits,
                "--output-subdir",
                union_subdir,
            ],
            env,
            args.dry_run,
        )
        run_command(
            [
                sys.executable,
                python_script("build_union_matrix.py"),
                "--selection-json",
                str(selection_json),
                "--tasks",
                "tox",
                "--splits",
                *tox_splits,
                "--output-subdir",
                union_subdir,
            ],
            env,
            args.dry_run,
        )

    report_prefix = "no_phys8_lightgbm_compact"
    if "topk" in args.stages:
        run_command(
            [
                sys.executable,
                python_script("select_union_features_lgbm.py"),
                "--tasks",
                "amp",
                "tox",
                "--union-subdir",
                union_subdir,
                "--topk-candidates",
                *DEFAULT_TOPK,
                "--random-state",
                "13",
                "--report-prefix",
                report_prefix,
                "--compact-roc-tolerance",
                "0.0005",
                "--compact-pr-tolerance",
                "0.0005",
            ],
            env,
            args.dry_run,
        )

    if "bundle" in args.stages and not args.dry_run:
        write_shared_bundle(output_root, report_prefix=report_prefix, union_subdir=union_subdir, include_hard=args.include_hard)


if __name__ == "__main__":
    main()
