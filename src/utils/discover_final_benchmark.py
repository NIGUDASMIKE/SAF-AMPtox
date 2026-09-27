from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (REPO_ROOT / "data--final", REPO_ROOT / "data" / "amp")
SPLIT_FILES = ("train.csv", "val.csv", "test.csv")
RAW_AMP_DIR_NAMES = {
    "dbaasp",
    "dramp3",
    "positive",
    "negative",
    "new negative",
    "uniprot",
}


@dataclass(frozen=True)
class BenchmarkCandidate:
    task: str
    path: str
    split_dir: str
    version: int
    mtime: float
    score: tuple[int, float, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-discover final AMP/TOX benchmark folders and optionally clean stale intermediate folders."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_ROOTS],
        help="Roots to scan for benchmark folders.",
    )
    parser.add_argument(
        "--json-output",
        default=str(REPO_ROOT / "data--final" / "reports" / "benchmark_discovery.json"),
        help="Where to write the discovery manifest.",
    )
    parser.add_argument(
        "--execute-cleanup",
        action="store_true",
        help="Actually delete stale candidates/reports. Default is dry-run only.",
    )
    parser.add_argument(
        "--cleanup-amp-reports",
        action="store_true",
        help="Include data/amp/reports in the cleanup proposal.",
    )
    return parser.parse_args()


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def infer_task(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "tox" in lowered:
        return "tox"
    return "amp"


def version_from_path(path: Path) -> int:
    versions: list[int] = []
    for part in path.parts:
        versions.extend(int(match) for match in re.findall(r"(?:^|[_-])v(\d+)(?:$|[_-])", part.lower()))
    return max(versions) if versions else 0


def split_dir_is_complete(split_dir: Path) -> bool:
    return split_dir.is_dir() and all((split_dir / name).is_file() for name in SPLIT_FILES)


def candidate_mtime(split_dir: Path) -> float:
    mtimes = [(split_dir / name).stat().st_mtime for name in SPLIT_FILES if (split_dir / name).exists()]
    return max(mtimes) if mtimes else 0.0


def find_candidates(roots: list[Path]) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        possible_split_dirs = []
        root_split = root / "splits"
        if split_dir_is_complete(root_split):
            possible_split_dirs.append(root_split)
        possible_split_dirs.extend(path for path in root.rglob("splits") if split_dir_is_complete(path))

        for split_dir in possible_split_dirs:
            bench_dir = split_dir.parent.resolve()
            if bench_dir in seen:
                continue
            seen.add(bench_dir)
            task = infer_task(bench_dir)
            version = version_from_path(bench_dir)
            mtime = candidate_mtime(split_dir)
            preferred_final_root = 1 if is_inside(bench_dir, REPO_ROOT / "data--final") else 0
            score = (preferred_final_root, mtime, version)
            candidates.append(
                BenchmarkCandidate(
                    task=task,
                    path=str(bench_dir),
                    split_dir=str(split_dir.resolve()),
                    version=version,
                    mtime=mtime,
                    score=score,
                )
            )
    return candidates


def choose_latest(candidates: list[BenchmarkCandidate]) -> dict[str, BenchmarkCandidate]:
    selected: dict[str, BenchmarkCandidate] = {}
    for task in ("amp", "tox"):
        task_candidates = [candidate for candidate in candidates if candidate.task == task]
        if task_candidates:
            selected[task] = sorted(task_candidates, key=lambda item: item.score, reverse=True)[0]
    return selected


def proposed_cleanup(candidates: list[BenchmarkCandidate], selected: dict[str, BenchmarkCandidate], cleanup_amp_reports: bool) -> list[Path]:
    keep_paths = {Path(candidate.path).resolve() for candidate in selected.values()}
    cleanup: list[Path] = []

    for candidate in candidates:
        candidate_path = Path(candidate.path).resolve()
        if candidate_path not in keep_paths and is_inside(candidate_path, REPO_ROOT):
            cleanup.append(candidate_path)

    amp_reports = REPO_ROOT / "data" / "amp" / "reports"
    if cleanup_amp_reports and amp_reports.exists():
        cleanup.append(amp_reports.resolve())

    # Only propose removing obvious intermediate benchmark/report directories.
    filtered: list[Path] = []
    for path in cleanup:
        if not is_inside(path, REPO_ROOT):
            continue
        if path == REPO_ROOT or path.parent == REPO_ROOT:
            continue
        if path.name.lower() in RAW_AMP_DIR_NAMES:
            continue
        if is_inside(path, REPO_ROOT / "data" / "amp") or is_inside(path, REPO_ROOT / "data--final"):
            filtered.append(path)
    return sorted(set(filtered), key=lambda item: str(item).lower())


def write_manifest(path: Path, candidates: list[BenchmarkCandidate], selected: dict[str, BenchmarkCandidate], cleanup: list[Path], executed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": str(REPO_ROOT),
        "candidates": [asdict(candidate) for candidate in candidates],
        "selected": {task: asdict(candidate) for task, candidate in selected.items()},
        "cleanup_proposal": [str(item) for item in cleanup],
        "cleanup_executed": executed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [Path(root).resolve() for root in args.roots]
    candidates = find_candidates(roots)
    selected = choose_latest(candidates)
    cleanup = proposed_cleanup(candidates, selected, cleanup_amp_reports=args.cleanup_amp_reports)

    print("[DISCOVERY] Benchmark candidates:")
    for candidate in sorted(candidates, key=lambda item: (item.task, item.path)):
        selected_mark = "SELECTED" if selected.get(candidate.task) == candidate else "candidate"
        print(f"  - {candidate.task.upper():3s} {selected_mark:9s} v{candidate.version:<2d} {candidate.path}")

    print("[DISCOVERY] Final selected benchmark folders:")
    for task, candidate in selected.items():
        print(f"  - {task.upper()}: {candidate.path}")

    print("[CLEANUP] Proposed stale paths:")
    if cleanup:
        for path in cleanup:
            print(f"  - {path}")
    else:
        print("  - none")

    if args.execute_cleanup:
        print("[CLEANUP] execute-cleanup enabled; deleting proposed stale paths.")
        for path in cleanup:
            if not is_inside(path, REPO_ROOT):
                raise RuntimeError(f"Refusing to delete outside repo: {path}")
            shutil.rmtree(path)
            print(f"  deleted: {path}")
    else:
        print("[CLEANUP] dry-run only; nothing was deleted.")

    write_manifest(Path(args.json_output), candidates, selected, cleanup, executed=args.execute_cleanup)
    print(f"[DONE] manifest -> {args.json_output}")


if __name__ == "__main__":
    main()
