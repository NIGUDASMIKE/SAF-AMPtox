from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import combine_feature_frames, feature_path, metadata_frame, read_table, report_path, write_table
from config import DEFAULT_SPLITS, DEFAULT_TASKS, FEATURE_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate selected feature groups into union matrices.")
    parser.add_argument("--selection-json", default=str(report_path("selected_groups_union.json")))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=list(DEFAULT_TASKS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output-subdir", default="union")
    return parser.parse_args()


def load_groups(selection_json: str) -> list[str]:
    payload = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    groups = payload.get("selected_union_groups", [])
    if not groups:
        raise ValueError("No selected_union_groups found in selection manifest")
    return groups


def build_union(task: str, split: str, groups: list[str], args: argparse.Namespace) -> Path:
    frames = []
    metadata = None
    for group in groups:
        frame = read_table(feature_path(task, group, split))
        if metadata is None:
            metadata = metadata_frame(frame)
        frames.append(frame)
    assert metadata is not None
    combined = combine_feature_frames(metadata, frames)
    out_dir = FEATURE_OUTPUT_ROOT / task / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_subdir}_{task}_{split}.parquet"
    write_table(combined, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    groups = load_groups(args.selection_json)
    for task in args.tasks:
        for split in args.splits:
            out_path = build_union(task, split, groups, args)
            print(f"[OK] {task}/{split} union -> {out_path}")


if __name__ == "__main__":
    main()
