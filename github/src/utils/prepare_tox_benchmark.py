import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.prepare_amp_benchmarks import (
    aggregate_records,
    assign_fasta_ids,
    build_final_length_matched_negative_library,
    build_conflict_report,
    clean_sequence_strict,
    cluster_aware_train_val_test_split_rows,
    is_length_ok,
    parse_fasta,
    record_template,
    run_cdhit_wsl,
    summarize_dataset,
    with_label_id,
    write_cdhit_input_fasta,
    write_csv,
    write_fasta,
    write_split_bundle,
)


LABEL_PATTERN = re.compile(r"\|([01])$")


def parse_tox_label_from_header(header: str) -> Optional[str]:
    match = LABEL_PATTERN.search(header.strip())
    if not match:
        return None
    return match.group(1)


def tox_source_split_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if stem == "train_data":
        return "provided_train"
    if stem == "test1":
        return "provided_test1"
    if stem == "test2":
        return "provided_test2"
    return f"provided_{stem}"


def task_label_name(label: str) -> str:
    if label == "positive":
        return "toxic"
    if label == "negative":
        return "non_toxic"
    raise ValueError(f"Unknown label: {label}")


def annotate_task_labels(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{**row, "task_label": task_label_name(row["label"])} for row in rows]


def load_tox_labeled_text_records(
    data_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]], List[str]]:
    positive_dir = data_root / "tox" / "positive"
    records: List[Dict[str, str]] = []
    file_stats: Dict[str, Dict[str, int]] = {}
    ignored_files: List[str] = []

    for path in sorted(positive_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            ignored_files.append(path.name)
            continue

        stats = Counter()
        source_split = tox_source_split_from_filename(path.name)
        for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(path), start=1):
            stats["raw_rows"] += 1

            raw_label = parse_tox_label_from_header(header)
            if raw_label is None:
                stats["skip_unparseable_label"] += 1
                continue

            cleaned_sequence, error = clean_sequence_strict(raw_sequence)
            if cleaned_sequence is None:
                if error == "empty_after_whitespace_cleanup":
                    stats["skip_empty_after_clean"] += 1
                else:
                    stats["skip_non_standard_sequence"] += 1
                continue

            if not is_length_ok(cleaned_sequence):
                stats["skip_length_filter"] += 1
                continue

            label = "positive" if raw_label == "1" else "negative"
            stats["kept_rows"] += 1
            stats[f"kept_label_{raw_label}"] += 1
            records.append(
                record_template(
                    sequence=cleaned_sequence,
                    label=label,
                    source_db="tox_labeled",
                    source_split=source_split,
                    source_file=path.name,
                    record_id=f"{path.name}:{sequence_index}",
                    record_name=header,
                    raw_sequence=raw_sequence,
                    raw_label=raw_label,
                )
            )

        file_stats[path.name] = dict(stats)

    return records, file_stats, ignored_files


def load_tox_negative_records(
    data_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]], List[str]]:
    negative_dir = data_root / "tox" / "negative"
    records: List[Dict[str, str]] = []
    file_stats: Dict[str, Dict[str, int]] = {}
    used_files: List[str] = []

    sequence_files = sorted(
        path
        for path in negative_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in {".txt", ".fa", ".fasta", ".faa", ".gz"}
            or path.name.lower().endswith(".fasta.gz")
        )
    )

    for path in sequence_files:
        stats = Counter()
        for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(path), start=1):
            stats["raw_rows"] += 1

            cleaned_sequence, error = clean_sequence_strict(raw_sequence)
            if cleaned_sequence is None:
                if error == "empty_after_whitespace_cleanup":
                    stats["skip_empty_after_clean"] += 1
                else:
                    stats["skip_non_standard_sequence"] += 1
                continue

            if not is_length_ok(cleaned_sequence):
                stats["skip_length_filter"] += 1
                continue

            stats["kept_rows"] += 1
            records.append(
                record_template(
                    sequence=cleaned_sequence,
                    label="negative",
                    source_db="tox_negative_import",
                    source_split="provided_negative",
                    source_file=path.name,
                    record_id=f"{path.name}:{sequence_index}",
                    record_name=header,
                    raw_sequence=raw_sequence,
                    raw_label="0",
                )
            )

        used_files.append(path.name)
        file_stats[path.name] = dict(stats)

    return records, file_stats, used_files


def write_tox_report(
    project_root: Path,
    base_dir: Path,
    summary: Dict[str, object],
) -> Path:
    reports_dir = base_dir / "reports"
    outputs = summary["outputs"]
    splits = summary["splits"]
    clustering = summary["clustering"]
    sources = summary["source_local_stats"]

    total = outputs["tox_benchmark_final"]["unique_sequences"]
    positive_total = outputs["toxic_positive_pool"]["unique_sequences"]
    negative_before_total = outputs["non_toxic_negative_pool_before_matching"]["unique_sequences"]
    negative_after_total = outputs["non_toxic_negative_pool_final"]["unique_sequences"]
    matching = summary["length_matching"]

    lines = [
        "TOX Final Benchmark Report",
        "Date: 2026-06-06",
        f"Project root: {project_root}",
        "",
        "1. Source policy",
        "",
        "Positive/toxic source:",
        "- data\\tox\\positive\\*.txt rows whose FASTA-like header ends with |1",
        "",
        "Negative/non-toxic sources:",
        "- data\\tox\\positive\\*.txt rows whose FASTA-like header ends with |0",
        "- data\\tox\\negative\\* sequence files",
        "",
        "Ignored source files in data\\tox\\positive:",
    ]
    for filename in sources["ignored_positive_files"]:
        lines.append(f"- {filename}")

    lines.extend(
        [
            "",
            "2. Cleaning and filtering rules",
            "",
            "All sources were processed with the same strict rules:",
            "1. Remove whitespace",
            "2. Convert sequence to uppercase",
            "3. If any non-standard residue appears, drop the entire sequence",
            "4. Keep only sequences with length 10-50 aa",
            "5. Deduplicate after cleaning",
            "6. If the same cleaned sequence appears in both toxic and non-toxic pools, remove it completely from the benchmark",
            "",
            "Allowed amino acids:",
            "ACDEFGHIKLMNPQRSTVWY",
            "",
            "3. Per-file retention statistics",
            "",
            "Labeled text files:",
        ]
    )
    for filename, stats in sources["labeled_text_files"].items():
        lines.append(
            f"- {filename}: raw_rows = {stats.get('raw_rows', 0)}, kept_rows = {stats.get('kept_rows', 0)}, "
            f"kept_label_1 = {stats.get('kept_label_1', 0)}, kept_label_0 = {stats.get('kept_label_0', 0)}, "
            f"skip_non_standard_sequence = {stats.get('skip_non_standard_sequence', 0)}, "
            f"skip_length_filter = {stats.get('skip_length_filter', 0)}"
        )

    lines.extend(["", "Imported negative files:"])
    for filename, stats in sources["negative_files"].items():
        lines.append(
            f"- {filename}: raw_rows = {stats.get('raw_rows', 0)}, kept_rows = {stats.get('kept_rows', 0)}, "
            f"skip_non_standard_sequence = {stats.get('skip_non_standard_sequence', 0)}, "
            f"skip_length_filter = {stats.get('skip_length_filter', 0)}"
        )

    lines.extend(
        [
            "",
            "4. Final benchmark result",
            "",
            f"- unique_sequences = {outputs['tox_benchmark_final']['unique_sequences']}",
            f"- toxic_positive_unique = {outputs['tox_benchmark_final']['positive_unique']}",
            f"- non_toxic_negative_unique = {outputs['tox_benchmark_final']['negative_unique']}",
            f"- negative_to_positive_ratio = {outputs['tox_benchmark_final']['negative_to_positive_ratio']}",
            f"- support_sum = {outputs['tox_benchmark_final']['support_sum']}",
            f"- duplicate_records_collapsed = {outputs['tox_benchmark_final']['duplicate_records_collapsed']}",
            f"- conflicting_sequences_removed = {outputs['conflicting_sequences_removed']}",
            "",
            "Component pools after conflict removal and before length matching:",
            f"- toxic_positive_pool = {positive_total}",
            f"- non_toxic_negative_pool_before_matching = {negative_before_total}",
            f"- negative_to_positive_ratio_before_matching = {round(negative_before_total / positive_total, 6) if positive_total else 0.0}",
            "",
            "Final non-toxic negative library after length matching:",
            f"- non_toxic_negative_pool_final = {negative_after_total}",
            f"- negative_to_positive_ratio_after_matching = {round(negative_after_total / positive_total, 6) if positive_total else 0.0}",
            "",
            "5. Length-distribution matching diagnostics",
            "",
            f"- exact_match_capacity = {matching['exact_match_capacity']}",
            f"- exact_match_capacity_ratio_vs_positive = {matching['exact_match_capacity_ratio_vs_positive']}",
            f"- positive_mean_length = {matching['positive_mean_length']}",
            f"- negative_pool_mean_length_before = {matching['negative_pool_mean_length_before']}",
            f"- final_negative_mean_length_after = {matching['final_negative_mean_length_after']}",
            f"- total_variation_distance_before = {matching['tv_distance_before']}",
            f"- total_variation_distance_after = {matching['tv_distance_after']}",
            f"- js_divergence_before = {matching['js_divergence_before']}",
            f"- js_divergence_after = {matching['js_divergence_after']}",
            f"- exact_length_bins_matched = {matching['exact_length_bins_matched']} / {matching['length_bin_count']}",
            f"- relocated_count = {matching['relocated_count']}",
            "",
            "6. CD-HIT clustering and split",
            "",
            f"- cluster_count = {clustering['cluster_count']}",
            f"- cluster_file_membership_count = {splits['cluster_file_membership_count']}",
            f"- fallback_singletons = {splits['fallback_singletons']}",
            "",
            "Train:",
            f"- unique_sequences = {splits['train']['unique_sequences']}",
            f"- toxic_positive_unique = {splits['train']['positive_unique']}",
            f"- non_toxic_negative_unique = {splits['train']['negative_unique']}",
            "",
            "Val:",
            f"- unique_sequences = {splits['val']['unique_sequences']}",
            f"- toxic_positive_unique = {splits['val']['positive_unique']}",
            f"- non_toxic_negative_unique = {splits['val']['negative_unique']}",
            "",
            "Test:",
            f"- unique_sequences = {splits['test']['unique_sequences']}",
            f"- toxic_positive_unique = {splits['test']['positive_unique']}",
            f"- non_toxic_negative_unique = {splits['test']['negative_unique']}",
            "",
            "Approximate split ratios by total sequence count:",
            f"- train = {splits['train']['unique_sequences']} / {total} = {splits['train']['unique_sequences'] / total:.4f}",
            f"- val = {splits['val']['unique_sequences']} / {total} = {splits['val']['unique_sequences'] / total:.4f}",
            f"- test = {splits['test']['unique_sequences']} / {total} = {splits['test']['unique_sequences'] / total:.4f}",
            "",
            "7. Main artifacts",
            "",
            f"- benchmark_csv = {summary['artifacts']['benchmark_csv']}",
            f"- toxic_positive_csv = {summary['artifacts']['toxic_positive_csv']}",
            f"- non_toxic_negative_csv = {summary['artifacts']['non_toxic_negative_csv']}",
            f"- non_toxic_negative_raw_csv = {summary['artifacts']['non_toxic_negative_raw_csv']}",
            f"- cdhit_input_fasta = {summary['artifacts']['cdhit_input_fasta']}",
            f"- cluster_file = {summary['artifacts']['cluster_file']}",
            f"- train_csv = {summary['artifacts']['train_csv']}",
            f"- val_csv = {summary['artifacts']['val_csv']}",
            f"- test_csv = {summary['artifacts']['test_csv']}",
            f"- length_distribution_report = {summary['artifacts']['length_distribution_report']}",
            f"- summary_json = {summary['artifacts']['summary_json']}",
            f"- conflict_report = {summary['artifacts']['conflict_report']}",
            "",
            "8. One-sentence handoff summary",
            "",
            "This TOX benchmark merges labeled rows from data/tox/positive and imported non-toxic sequences from data/tox/negative, then applies strict cleaning, conflict removal, exact 1:1 length-matched negative construction, CD-HIT 80% clustering, and cluster-aware train/val/test splitting.",
        ]
    )

    report_path = reports_dir / "tox_final_benchmark_report_20260606.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_tox_readme(base_dir: Path, summary: Dict[str, object]) -> None:
    outputs = summary["outputs"]
    matching = summary["length_matching"]
    readme = [
        "# TOX Final Benchmark",
        "",
        "This folder contains the final TOX benchmark prepared on 2026-06-06.",
        "",
        "## Label semantics",
        "",
        "- `label = positive`, `label_id = 1`, `task_label = toxic`",
        "- `label = negative`, `label_id = 0`, `task_label = non_toxic`",
        "",
        "## Source policy",
        "",
        "- Toxic positives come from `data/tox/positive/*.txt` rows labeled `|1`",
        "- Non-toxic negatives come from `data/tox/positive/*.txt` rows labeled `|0` plus imported files under `data/tox/negative`",
        "",
        "## Shared filtering rules",
        "",
        "1. Remove whitespace",
        "2. Convert to uppercase",
        "3. Drop the entire sequence if any non-standard amino acid appears",
        "4. Keep only sequences with length `10-50 aa`",
        "5. Deduplicate after cleaning",
        "6. Drop all toxic/non-toxic label-conflict sequences",
        "",
        "Allowed amino acids:",
        "",
        "`ACDEFGHIKLMNPQRSTVWY`",
        "",
        "## Final benchmark summary",
        "",
        f"- Total unique sequences: `{outputs['tox_benchmark_final']['unique_sequences']}`",
        f"- Toxic positives: `{outputs['tox_benchmark_final']['positive_unique']}`",
        f"- Non-toxic negatives: `{outputs['tox_benchmark_final']['negative_unique']}`",
        f"- Negative-to-positive ratio: `{outputs['tox_benchmark_final']['negative_to_positive_ratio']}`",
        "",
        "## Length matching",
        "",
        "- The final non-toxic library is built to match the toxic positive length distribution",
        f"- Exact match capacity at equal total was sufficient: `{matching['exact_match_capacity']} >= {matching['positive_total']}`",
        f"- Final TV distance after matching: `{matching['tv_distance_after']}`",
        f"- Final JS divergence after matching: `{matching['js_divergence_after']}`",
        "",
        "## Main files",
        "",
        "- `tox_benchmark_final.csv`",
        "- `toxic_positive.csv`",
        "- `non_toxic_negative.csv`",
        "- `non_toxic_negative_raw.csv`",
        "- `tox_for_cdhit_final.fasta`",
        "",
        "## Splits",
        "",
        "- `splits/train.csv`",
        "- `splits/val.csv`",
        "- `splits/test.csv`",
        "",
        "## Reports",
        "",
        "- `reports/tox_final_benchmark_report_20260606.txt`",
        "- `reports/tox_final_benchmark_summary.json`",
        "- `reports/conflicting_sequences_tox.csv`",
        "- `reports/tox_length_distribution_matching.csv`",
    ]
    (base_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def prepare_tox_benchmark() -> Dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    tox_root = data_root / "tox"
    base_dir = project_root / "data--final" / "tox"
    cdhit_dir = base_dir / "cdhit"
    splits_dir = base_dir / "splits"
    reports_dir = base_dir / "reports"
    cdhit_binary = project_root / "tools" / "cdhit" / "cd-hit"

    labeled_records, labeled_stats, ignored_positive_files = load_tox_labeled_text_records(data_root)
    imported_negative_records, imported_negative_stats, used_negative_files = load_tox_negative_records(data_root)

    all_records = labeled_records + imported_negative_records
    conflicts, conflict_rows = build_conflict_report(all_records)
    non_conflict_records = [record for record in all_records if record["sequence"] not in conflicts]

    toxic_positive_rows = aggregate_records(
        record for record in non_conflict_records if record["label"] == "positive"
    )
    non_toxic_negative_rows_raw = aggregate_records(
        record for record in non_conflict_records if record["label"] == "negative"
    )
    non_toxic_negative_rows, length_matching_report, length_distribution_rows = build_final_length_matched_negative_library(
        toxic_positive_rows,
        non_toxic_negative_rows_raw,
        random_state=42,
    )
    non_toxic_negative_rows = sorted(non_toxic_negative_rows, key=lambda row: row["sequence"])
    final_rows = sorted(
        toxic_positive_rows + non_toxic_negative_rows,
        key=lambda row: (0 if row["label"] == "positive" else 1, row["sequence"]),
    )
    final_rows = annotate_task_labels(with_label_id(assign_fasta_ids(final_rows, prefix="toxseq")))
    toxic_positive_rows = annotate_task_labels(toxic_positive_rows)
    non_toxic_negative_rows = annotate_task_labels(non_toxic_negative_rows)
    non_toxic_negative_rows_raw = annotate_task_labels(non_toxic_negative_rows_raw)

    write_csv(base_dir / "tox_benchmark_final.csv", final_rows)
    write_csv(base_dir / "toxic_positive.csv", toxic_positive_rows)
    write_fasta(base_dir / "toxic_positive.fa", toxic_positive_rows)
    write_csv(base_dir / "non_toxic_negative.csv", non_toxic_negative_rows)
    write_fasta(base_dir / "non_toxic_negative.fa", non_toxic_negative_rows)
    write_csv(base_dir / "non_toxic_negative_raw.csv", non_toxic_negative_rows_raw)
    write_fasta(base_dir / "non_toxic_negative_raw.fa", non_toxic_negative_rows_raw)
    write_cdhit_input_fasta(base_dir / "tox_for_cdhit_final.fasta", final_rows)
    write_csv(reports_dir / "conflicting_sequences_tox.csv", conflict_rows)
    write_csv(reports_dir / "tox_length_distribution_matching.csv", length_distribution_rows)

    cdhit_paths = run_cdhit_wsl(
        cdhit_binary=cdhit_binary,
        input_fasta=base_dir / "tox_for_cdhit_final.fasta",
        output_prefix=cdhit_dir / "internal_tox_c80",
        identity=0.8,
        word_length=5,
    )

    split_rows, split_summary = cluster_aware_train_val_test_split_rows(
        final_rows,
        cdhit_paths["clustered_clstr"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )
    split_rows = {split_name: annotate_task_labels(rows) for split_name, rows in split_rows.items()}
    write_split_bundle(splits_dir, split_rows)

    cluster_count = sum(
        1
        for line in Path(cdhit_paths["clustered_clstr"]).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith(">Cluster ")
    )

    summary = {
        "policy": {
            "name": "tox_final_benchmark",
            "positive_semantics": "toxic",
            "negative_semantics": "non_toxic",
            "source_rule": (
                "Rows labeled |1 in data/tox/positive/*.txt are treated as toxic positives; "
                "rows labeled |0 there plus sequence files under data/tox/negative are treated as non-toxic negatives."
            ),
            "original_splits_merged": ["train_data.txt", "test1.txt", "test2.txt"],
            "re_split_strategy": "cluster_aware_train_val_test_split_rows",
        },
        "filter_rules": {
            "uppercase_after_whitespace_cleanup": True,
            "non_standard_sequence_policy": "drop_entire_sequence",
            "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "length_range": [10, 50],
            "drop_label_conflicts": True,
        },
        "source_local_stats": {
            "labeled_text_files": labeled_stats,
            "negative_files": imported_negative_stats,
            "ignored_positive_files": ignored_positive_files,
            "used_negative_files": used_negative_files,
        },
        "outputs": {
            "tox_benchmark_final": summarize_dataset(final_rows),
            "toxic_positive_pool": summarize_dataset(toxic_positive_rows),
            "non_toxic_negative_pool_before_matching": summarize_dataset(non_toxic_negative_rows_raw),
            "non_toxic_negative_pool_final": summarize_dataset(non_toxic_negative_rows),
            "conflicting_sequences_removed": len(conflicts),
        },
        "length_matching": length_matching_report,
        "clustering": {
            "tool": "CD-HIT",
            "identity_threshold": 0.8,
            "word_length": 5,
            "cluster_count": cluster_count,
            "output_prefix": str(Path(cdhit_paths["clustered_fasta"]).relative_to(project_root)),
            "cluster_file": str(Path(cdhit_paths["clustered_clstr"]).relative_to(project_root)),
        },
        "splits": split_summary,
        "artifacts": {
            "base_dir": str(base_dir.relative_to(project_root)),
            "benchmark_csv": str((base_dir / "tox_benchmark_final.csv").relative_to(project_root)),
            "toxic_positive_csv": str((base_dir / "toxic_positive.csv").relative_to(project_root)),
            "non_toxic_negative_csv": str((base_dir / "non_toxic_negative.csv").relative_to(project_root)),
            "non_toxic_negative_raw_csv": str((base_dir / "non_toxic_negative_raw.csv").relative_to(project_root)),
            "cdhit_input_fasta": str((base_dir / "tox_for_cdhit_final.fasta").relative_to(project_root)),
            "cluster_file": str((cdhit_dir / "internal_tox_c80.clstr").relative_to(project_root)),
            "train_csv": str((splits_dir / "train.csv").relative_to(project_root)),
            "val_csv": str((splits_dir / "val.csv").relative_to(project_root)),
            "test_csv": str((splits_dir / "test.csv").relative_to(project_root)),
            "length_distribution_report": str((reports_dir / "tox_length_distribution_matching.csv").relative_to(project_root)),
            "summary_json": str((reports_dir / "tox_final_benchmark_summary.json").relative_to(project_root)),
            "conflict_report": str((reports_dir / "conflicting_sequences_tox.csv").relative_to(project_root)),
            "report_txt": str((reports_dir / "tox_final_benchmark_report_20260606.txt").relative_to(project_root)),
        },
    }

    with (reports_dir / "tox_final_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path = write_tox_report(project_root, base_dir, summary)
    summary["artifacts"]["report_txt"] = str(report_path.relative_to(project_root))
    with (reports_dir / "tox_final_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_tox_readme(base_dir, summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare_tox_benchmark(), ensure_ascii=False, indent=2))
