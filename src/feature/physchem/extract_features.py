from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import feature_path, load_split, metadata_frame, output_dir_for
from config import DEFAULT_FEATURE_GROUPS, DEFAULT_SPLITS, DEFAULT_TASKS
from extractors import extract_feature_group
from feature_registry import FEATURE_SPEC_BY_NAME, ready_feature_groups, resolve_feature_groups
from common import write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export physicochemical feature matrices.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=list(DEFAULT_TASKS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS))
    parser.add_argument("--all-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_group(task: str, split: str, group: str, overwrite: bool) -> Path:
    output_dir_for(task, group)
    out_path = feature_path(task, group, split)
    if out_path.exists() and not overwrite:
        return out_path

    split_df = load_split(task, split)
    spec = FEATURE_SPEC_BY_NAME[group]
    features = extract_feature_group(
        group,
        split_df["sequence"].tolist(),
        backend=spec.backend,
        extractor_key=spec.extractor_key or group,
        fasta_ids=split_df["fasta_id"].tolist(),
    )
    features = features.reset_index(drop=True)
    exported = metadata_frame(split_df)
    exported = exported.join(features)
    write_table(exported, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    groups = ready_feature_groups() if args.all_ready else resolve_feature_groups(args.groups)
    for task in args.tasks:
        for split in args.splits:
            for group in groups:
                out_path = export_group(task, split, group, overwrite=args.overwrite)
                print(f"[OK] {task}/{split}/{group} -> {out_path}")


if __name__ == "__main__":
    main()
