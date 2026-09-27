import csv
import gzip
import json
import math
import random
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DBAASP_POSITIVE_KEYWORDS = (
    "GRAM+",
    "GRAM-",
    "FUNG",
    "VIRUS",
    "PARASIT",
    "BACTER",
    "BIOFILM",
)
DRAMP_POSITIVE_KEYWORDS = (
    "ANTIMICROBIAL",
    "ANTIBACTERIAL",
    "ANTI-GRAM",
    "ANTIFUNGAL",
    "ANTIVIRAL",
    "ANTIPARASITIC",
    "ANTIBIOFILM",
    "ANTIMYCOBACTERIAL",
)
MIN_LEN = 10
MAX_LEN = 50


def normalize_whitespace_and_case(raw_sequence: str) -> str:
    return re.sub(r"\s+", "", (raw_sequence or "").upper())


def clean_sequence_strict(raw_sequence: str) -> Tuple[Optional[str], Optional[str]]:
    sequence = normalize_whitespace_and_case(raw_sequence)
    if not sequence:
        return None, "empty_after_whitespace_cleanup"

    invalid_chars = sorted({char for char in sequence if char not in STANDARD_AA})
    if invalid_chars:
        return None, f"contains_non_standard_residue:{''.join(invalid_chars)}"

    return sequence, None


def is_length_ok(sequence: str) -> bool:
    return MIN_LEN <= len(sequence) <= MAX_LEN


def has_dbaasp_positive_label(target_group: str) -> bool:
    upper = (target_group or "").upper()
    return any(keyword in upper for keyword in DBAASP_POSITIVE_KEYWORDS)


def has_dramp_positive_label(activity: str) -> bool:
    upper = (activity or "").upper()
    return any(keyword in upper for keyword in DRAMP_POSITIVE_KEYWORDS)


def parse_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".fasta.gz") else open
    header = None
    sequence_chunks: List[str] = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_chunks)
                header = line[1:].strip()
                sequence_chunks = []
            else:
                sequence_chunks.append(line)
        if header is not None:
            yield header, "".join(sequence_chunks)


def parse_uniprot_header(header: str) -> Tuple[str, str]:
    parts = header.split("|")
    if len(parts) >= 3:
        accession = parts[1].strip()
        name = parts[2].strip()
        return accession, name
    return header.strip(), header.strip()


def excel_column_to_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    value = 0
    for char in letters:
        value = (value * 26) + (ord(char) - ord("A") + 1)
    return value - 1


def load_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in archive.namelist():
        return []

    shared_root = ET.fromstring(archive.read(shared_path))
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings: List[str] = []

    for item in shared_root:
        texts = [node.text or "" for node in item.iter(f"{namespace}t")]
        strings.append("".join(texts))

    return strings


def read_excel_cell(cell: ET.Element, shared_strings: List[str]) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.iter(f"{namespace}t")]
        return "".join(texts)

    value_node = cell.find(f"{namespace}v")
    if value_node is None:
        return ""

    if cell_type == "s":
        return shared_strings[int(value_node.text)]

    return value_node.text or ""


def iter_xlsx_rows(path: Path, sheet_name: str = None) -> Iterator[Dict[str, str]]:
    workbook_ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rel_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    sheet_data_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

    with zipfile.ZipFile(path) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook_root.find("a:sheets", workbook_ns)
        if sheets is None or not list(sheets):
            return

        selected_sheet = None
        for sheet in sheets:
            if sheet_name is None or sheet.attrib.get("name") == sheet_name:
                selected_sheet = sheet
                break
        if selected_sheet is None:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}")

        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in rel_root.findall("p:Relationship", rel_ns)
        }
        worksheet_path = "xl/" + rel_map[selected_sheet.attrib[rel_attr]].lstrip("/")
        shared_strings = load_shared_strings(archive)
        worksheet_root = ET.fromstring(archive.read(worksheet_path))
        sheet_data = worksheet_root.find(f"{sheet_data_ns}sheetData")
        if sheet_data is None:
            return

        header: List[str] = []
        for row_index, row in enumerate(sheet_data.findall(f"{sheet_data_ns}row")):
            cell_map: Dict[int, str] = {}
            for cell in row.findall(f"{sheet_data_ns}c"):
                cell_ref = cell.attrib.get("r", "")
                if not cell_ref:
                    continue
                cell_map[excel_column_to_index(cell_ref)] = read_excel_cell(cell, shared_strings)

            max_index = max(cell_map) if cell_map else -1
            row_values = [cell_map.get(index, "") for index in range(max_index + 1)]

            if row_index == 0:
                header = row_values
                continue

            if not header:
                continue

            yield {
                header[index]: row_values[index] if index < len(row_values) else ""
                for index in range(len(header))
            }


def record_template(
    *,
    sequence: str,
    label: str,
    source_db: str,
    source_split: str,
    source_file: str,
    record_id: str,
    record_name: str,
    raw_sequence: str,
    raw_label: str,
) -> Dict[str, str]:
    return {
        "sequence": sequence,
        "length": str(len(sequence)),
        "label": label,
        "source_db": source_db,
        "source_split": source_split,
        "source_file": source_file,
        "record_id": record_id,
        "record_name": record_name,
        "raw_sequence": raw_sequence,
        "raw_label": raw_label,
    }


def load_dbaasp_records(data_root: Path) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats = Counter()
    records: List[Dict[str, str]] = []

    for csv_path in sorted((data_root / "dbaasp").glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stats["raw_rows"] += 1

                if row.get("COMPLEXITY", "").strip() != "Monomer":
                    stats["skip_non_monomer"] += 1
                    continue

                raw_label = row.get("TARGET GROUP", "")
                if not has_dbaasp_positive_label(raw_label):
                    stats["skip_missing_positive_label"] += 1
                    continue

                cleaned_sequence, error = clean_sequence_strict(row.get("SEQUENCE", ""))
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
                        label="positive",
                        source_db="dbaasp",
                        source_split="internal",
                        source_file=csv_path.name,
                        record_id=row.get("ID", ""),
                        record_name=row.get("NAME", ""),
                        raw_sequence=row.get("SEQUENCE", ""),
                        raw_label=raw_label,
                    )
                )

    return records, dict(stats)


def load_dramp_records(data_root: Path) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats = Counter()
    records: List[Dict[str, str]] = []
    dramp_path = data_root / "dramp3" / "dramp.xlsx"

    for row in iter_xlsx_rows(dramp_path, sheet_name="general_amps"):
        stats["raw_rows"] += 1

        raw_label = row.get("Activity", "")
        if not has_dramp_positive_label(raw_label):
            stats["skip_missing_positive_label"] += 1
            continue

        cleaned_sequence, error = clean_sequence_strict(row.get("Sequence", ""))
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
                label="positive",
                source_db="dramp3",
                source_split="internal",
                source_file=dramp_path.name,
                record_id=row.get("DRAMP_ID", ""),
                record_name=row.get("Name", ""),
                raw_sequence=row.get("Sequence", ""),
                raw_label=raw_label,
            )
        )

    return records, dict(stats)


def load_amplify_records(data_root: Path) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]], List[str]]:
    specs = [
        ("AMPlify_AMP_train_common.fa", "positive", "internal_train"),
        ("AMPlify_AMP_test_common.fa", "positive", "external_test"),
        ("AMPlify_non_AMP_train_balanced.fa", "negative", "internal_train"),
        ("AMPlify_non_AMP_train_imbalanced.fa", "negative", "internal_train"),
        ("AMPlify_non_AMP_test_balanced.fa", "negative", "external_test"),
    ]
    ignored_files = ["AMPlify_non_AMP_test_imbalanced.fa"]

    all_records: List[Dict[str, str]] = []
    all_stats: Dict[str, Dict[str, int]] = {}

    for filename, label, split_name in specs:
        fasta_path = data_root / "amplify" / filename
        stats = Counter()

        for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(fasta_path), start=1):
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
            all_records.append(
                record_template(
                    sequence=cleaned_sequence,
                    label=label,
                    source_db="amplify",
                    source_split=split_name,
                    source_file=filename,
                    record_id=f"{filename}:{sequence_index}",
                    record_name=header,
                    raw_sequence=raw_sequence,
                    raw_label=label,
                )
            )

        all_stats[filename] = dict(stats)

    return all_records, all_stats, ignored_files


def load_uniprot_negative_records(data_root: Path) -> Tuple[List[Dict[str, str]], Dict[str, int], str]:
    stats = Counter()
    records: List[Dict[str, str]] = []
    uniprot_path = data_root / "uniprot" / "uniprotkb_reviewed_true_AND_length_10_T_2026_06_05.fasta.gz"

    for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(uniprot_path), start=1):
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

        accession, name = parse_uniprot_header(header)
        stats["kept_rows"] += 1
        records.append(
            record_template(
                sequence=cleaned_sequence,
                label="negative",
                source_db="uniprot",
                source_split="internal_train",
                source_file=uniprot_path.name,
                record_id=accession or f"{uniprot_path.name}:{sequence_index}",
                record_name=name,
                raw_sequence=raw_sequence,
                raw_label="negative",
            )
        )

    return records, dict(stats), uniprot_path.name


def load_amplify_negative_records_all(
    data_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]]]:
    specs = [
        "AMPlify_non_AMP_train_balanced.fa",
        "AMPlify_non_AMP_train_imbalanced.fa",
        "AMPlify_non_AMP_test_balanced.fa",
        "AMPlify_non_AMP_test_imbalanced.fa",
    ]

    all_records: List[Dict[str, str]] = []
    all_stats: Dict[str, Dict[str, int]] = {}

    for filename in specs:
        fasta_path = data_root / "amplify" / filename
        stats = Counter()

        for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(fasta_path), start=1):
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
            all_records.append(
                record_template(
                    sequence=cleaned_sequence,
                    label="negative",
                    source_db="amplify",
                    source_split="internal_train",
                    source_file=filename,
                    record_id=f"{filename}:{sequence_index}",
                    record_name=header,
                    raw_sequence=raw_sequence,
                    raw_label="negative",
                )
            )

        all_stats[filename] = dict(stats)

    return all_records, all_stats


def load_new_negative_records(
    data_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]], List[str]]:
    newnegative_dir = data_root / "newnegative"
    records: List[Dict[str, str]] = []
    all_stats: Dict[str, Dict[str, int]] = {}
    used_files: List[str] = []

    fasta_paths = sorted(
        path
        for path in newnegative_dir.iterdir()
        if path.is_file() and (
            path.suffix.lower() in {".fa", ".fasta", ".faa"}
            or path.name.lower().endswith(".fasta.gz")
        )
    )

    for fasta_path in fasta_paths:
        stats = Counter()

        for sequence_index, (header, raw_sequence) in enumerate(parse_fasta(fasta_path), start=1):
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
                    source_db="newnegative",
                    source_split="internal_train",
                    source_file=fasta_path.name,
                    record_id=f"{fasta_path.name}:{sequence_index}",
                    record_name=header,
                    raw_sequence=raw_sequence,
                    raw_label="negative",
                )
            )

        used_files.append(fasta_path.name)
        all_stats[fasta_path.name] = dict(stats)

    return records, all_stats, used_files


def build_conflict_report(records: Iterable[Dict[str, str]]) -> Tuple[set, List[Dict[str, str]]]:
    label_by_sequence: Dict[str, set] = defaultdict(set)
    sources_by_sequence: Dict[str, set] = defaultdict(set)
    source_files_by_sequence: Dict[str, set] = defaultdict(set)
    ids_by_sequence: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        sequence = record["sequence"]
        label_by_sequence[sequence].add(record["label"])
        sources_by_sequence[sequence].add(f"{record['source_db']}:{record['source_split']}")
        source_files_by_sequence[sequence].add(record["source_file"])
        ids_by_sequence[sequence].append(record["record_id"])

    conflicts = {sequence for sequence, labels in label_by_sequence.items() if len(labels) > 1}
    report_rows: List[Dict[str, str]] = []
    for sequence in sorted(conflicts):
        report_rows.append(
            {
                "sequence": sequence,
                "length": str(len(sequence)),
                "labels": ";".join(sorted(label_by_sequence[sequence])),
                "sources": ";".join(sorted(sources_by_sequence[sequence])),
                "source_files": ";".join(sorted(source_files_by_sequence[sequence])),
                "record_ids": ";".join(ids_by_sequence[sequence]),
            }
        )

    return conflicts, report_rows


def aggregate_records(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["sequence"], record["label"])].append(record)

    aggregated_rows: List[Dict[str, str]] = []
    for (sequence, label), group in sorted(grouped.items()):
        source_dbs = sorted({entry["source_db"] for entry in group})
        source_splits = sorted({entry["source_split"] for entry in group})
        source_files = sorted({entry["source_file"] for entry in group})
        record_ids = [entry["record_id"] for entry in group if entry["record_id"]]
        raw_labels = sorted({entry["raw_label"] for entry in group if entry["raw_label"]})
        names = [entry["record_name"] for entry in group if entry["record_name"]]

        aggregated_rows.append(
            {
                "sequence": sequence,
                "length": str(len(sequence)),
                "label": label,
                "support_count": str(len(group)),
                "source_dbs": ";".join(source_dbs),
                "source_splits": ";".join(source_splits),
                "source_files": ";".join(source_files),
                "record_ids": ";".join(record_ids),
                "example_name": names[0] if names else "",
                "raw_labels": " || ".join(raw_labels),
            }
        )

    return aggregated_rows


def assign_fasta_ids(rows: List[Dict[str, str]], prefix: str = "seq") -> List[Dict[str, str]]:
    rows_with_ids: List[Dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        fasta_id = f"{prefix}_{index:06d}"
        rows_with_ids.append({"fasta_id": fasta_id, **row})
    return rows_with_ids


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: List[Dict[str, str]], header_field: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(rows, start=1):
            if header_field:
                header = row[header_field]
            else:
                header = (
                    f"{index}|label={row['label']}|sources={row['source_dbs']}|"
                    f"splits={row['source_splits']}|length={row['length']}"
                )
            handle.write(f">{header}\n{row['sequence']}\n")


def write_cdhit_input_fasta(path: Path, rows_with_ids: List[Dict[str, str]]) -> None:
    write_fasta(path, rows_with_ids, header_field="fasta_id")


def label_to_id(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return 0
    raise ValueError(f"Unknown label: {label}")


def with_label_id(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows_with_label_id: List[Dict[str, str]] = []
    for row in rows:
        rows_with_label_id.append({**row, "label_id": str(label_to_id(row["label"]))})
    return rows_with_label_id


def summarize_dataset(rows: List[Dict[str, str]]) -> Dict[str, float]:
    label_counter = Counter(row["label"] for row in rows)
    support_sum = sum(int(row["support_count"]) for row in rows)
    positives = label_counter.get("positive", 0)
    negatives = label_counter.get("negative", 0)

    return {
        "unique_sequences": len(rows),
        "positive_unique": positives,
        "negative_unique": negatives,
        "support_sum": support_sum,
        "duplicate_records_collapsed": support_sum - len(rows),
        "negative_to_positive_ratio": round(negatives / positives, 6) if positives else None,
    }


def length_counts(rows: Iterable[Dict[str, str]]) -> Counter:
    counts = Counter()
    for row in rows:
        counts[int(row["length"])] += 1
    return counts


def distribute_target_counts_with_capacity(
    probabilities: Dict[int, float],
    capacities: Dict[int, int],
    total_target: int,
) -> Dict[int, int]:
    raw_targets = {length: probabilities[length] * total_target for length in probabilities}
    allocated = {
        length: min(int(raw_targets[length]), capacities.get(length, 0))
        for length in probabilities
    }

    remainder = total_target - sum(allocated.values())
    while remainder > 0:
        candidates = []
        for length in probabilities:
            capacity = capacities.get(length, 0)
            if allocated[length] < capacity:
                fractional = raw_targets[length] - int(raw_targets[length])
                slack = capacity - allocated[length]
                candidates.append((fractional, probabilities[length], slack, length))

        if not candidates:
            break

        candidates.sort(reverse=True)
        for _, _, _, length in candidates:
            if remainder <= 0:
                break
            if allocated[length] < capacities.get(length, 0):
                allocated[length] += 1
                remainder -= 1

    return allocated


def exact_length_matched_total(
    positive_counts: Dict[int, int],
    negative_counts: Dict[int, int],
) -> int:
    total_positive = sum(positive_counts.values())
    if total_positive == 0:
        return 0

    ratios = []
    for length, pos_count in positive_counts.items():
        if pos_count <= 0:
            continue
        neg_count = negative_counts.get(length, 0)
        if neg_count <= 0:
            return 0
        probability = pos_count / total_positive
        ratios.append(neg_count / probability)

    return int(min(ratios)) if ratios else 0


def negative_sampling_priority_weight(row: Dict[str, str]) -> float:
    sources = set(filter(None, row["source_dbs"].split(";")))
    if "amplify" in sources and "uniprot" not in sources:
        return 3.0
    if "amplify" in sources and "uniprot" in sources:
        return 2.5
    if "uniprot" in sources and len(sources) == 1:
        return 1.0
    return 1.5


def weighted_sample_without_replacement(
    rows: List[Dict[str, str]],
    sample_size: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    if sample_size <= 0:
        return []
    if sample_size >= len(rows):
        return list(rows)

    scored_rows = []
    for row in rows:
        weight = max(negative_sampling_priority_weight(row), 1e-9)
        key = rng.random() ** (1.0 / weight)
        scored_rows.append((key, row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored_rows[:sample_size]]


def source_group_name(row: Dict[str, str]) -> str:
    sources = set(filter(None, row["source_dbs"].split(";")))
    if "amplify" in sources and "uniprot" in sources:
        return "amplify_plus_uniprot"
    if "amplify" in sources:
        return "amplify_only"
    if "uniprot" in sources:
        return "uniprot_only"
    return "other_negative"


def source_signature(row: Dict[str, str]) -> str:
    sources = sorted(filter(None, row["source_dbs"].split(";")))
    return "+".join(sources) if sources else "unknown"


def build_length_matched_internal_benchmark(
    internal_rows_with_ids: List[Dict[str, str]],
    random_state: int = 42,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    positives = [row for row in internal_rows_with_ids if row["label"] == "positive"]
    negatives = [row for row in internal_rows_with_ids if row["label"] == "negative"]

    positive_counts = length_counts(positives)
    negative_counts = length_counts(negatives)
    total_positive = sum(positive_counts.values())
    probabilities = {
        length: positive_counts[length] / total_positive
        for length in sorted(positive_counts)
    }

    target_negative_total = exact_length_matched_total(positive_counts, negative_counts)
    target_counts = distribute_target_counts_with_capacity(
        probabilities=probabilities,
        capacities=negative_counts,
        total_target=target_negative_total,
    )

    negatives_by_length: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in negatives:
        negatives_by_length[int(row["length"])].append(row)

    rng = random.Random(random_state)
    selected_negatives: List[Dict[str, str]] = []
    for length in sorted(target_counts):
        selected_negatives.extend(
            weighted_sample_without_replacement(
                negatives_by_length[length],
                target_counts[length],
                rng,
            )
        )

    selected_negatives_by_id = {row["fasta_id"]: row for row in selected_negatives}
    selected_negatives = [selected_negatives_by_id[row["fasta_id"]] for row in negatives if row["fasta_id"] in selected_negatives_by_id]

    matched_rows = positives + selected_negatives
    matched_rows.sort(key=lambda row: (0 if row["label"] == "positive" else 1, row["sequence"]))

    matched_negative_counts = length_counts(selected_negatives)
    source_group_counter = Counter(source_group_name(row) for row in selected_negatives)
    report = {
        "target_negative_total_exact_match": target_negative_total,
        "positive_length_counts": {str(length): positive_counts[length] for length in sorted(positive_counts)},
        "negative_length_counts_before": {str(length): negative_counts[length] for length in sorted(negative_counts)},
        "negative_length_counts_after": {str(length): matched_negative_counts[length] for length in sorted(matched_negative_counts)},
        "negative_source_group_counts_after": dict(source_group_counter),
    }

    return matched_rows, report


def largest_remainder_target_counts(
    probabilities: Dict[int, float],
    total_target: int,
) -> Dict[int, int]:
    raw_targets = {length: probabilities.get(length, 0.0) * total_target for length in probabilities}
    allocated = {length: int(raw_targets[length]) for length in probabilities}
    remainder = total_target - sum(allocated.values())
    if remainder > 0:
        ranked = sorted(
            probabilities,
            key=lambda length: (raw_targets[length] - allocated[length], probabilities[length], -length),
            reverse=True,
        )
        for length in ranked[:remainder]:
            allocated[length] += 1
    return allocated


def fit_counts_to_capacity_by_nearest_length(
    desired_counts: Dict[int, int],
    capacities: Dict[int, int],
) -> Tuple[Dict[int, int], List[Dict[str, int]]]:
    lengths = sorted(set(desired_counts) | set(capacities))
    selected = {
        length: min(desired_counts.get(length, 0), capacities.get(length, 0))
        for length in lengths
    }
    remaining_capacity = {
        length: capacities.get(length, 0) - selected[length]
        for length in lengths
    }

    relocations: List[Dict[str, int]] = []
    for need_length in lengths:
        deficit = desired_counts.get(need_length, 0) - selected[need_length]
        if deficit <= 0:
            continue

        donor_lengths = sorted(
            [length for length in lengths if remaining_capacity.get(length, 0) > 0 and length != need_length],
            key=lambda length: (abs(length - need_length), length),
        )
        for donor_length in donor_lengths:
            if deficit <= 0:
                break
            available = remaining_capacity.get(donor_length, 0)
            if available <= 0:
                continue

            taken = min(deficit, available)
            selected[donor_length] += taken
            remaining_capacity[donor_length] -= taken
            deficit -= taken
            relocations.append(
                {
                    "target_length": need_length,
                    "sampled_length": donor_length,
                    "count": taken,
                    "distance": abs(donor_length - need_length),
                }
            )

        if deficit > 0:
            raise ValueError(
                f"Unable to fit target counts into available capacities. "
                f"Length {need_length} remains short by {deficit}."
            )

    return selected, relocations


def sample_rows_by_length_counts(
    rows: List[Dict[str, str]],
    target_counts: Dict[int, int],
    random_state: int = 42,
) -> List[Dict[str, str]]:
    rows_by_length: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_length[int(row["length"])].append(row)

    rng = random.Random(random_state)
    selected_rows: List[Dict[str, str]] = []
    for length in sorted(target_counts):
        candidates = list(rows_by_length.get(length, []))
        target = target_counts[length]
        if target <= 0:
            continue
        if target > len(candidates):
            raise ValueError(
                f"Requested {target} rows of length {length}, but only {len(candidates)} are available."
            )
        rng.shuffle(candidates)
        selected_rows.extend(candidates[:target])

    return selected_rows


def probabilities_from_counts(counts: Dict[int, int], lengths: List[int]) -> Dict[int, float]:
    total = sum(counts.get(length, 0) for length in lengths)
    if total <= 0:
        return {length: 0.0 for length in lengths}
    return {length: counts.get(length, 0) / total for length in lengths}


def total_variation_distance(
    probs_a: Dict[int, float],
    probs_b: Dict[int, float],
    lengths: List[int],
) -> float:
    return 0.5 * sum(abs(probs_a.get(length, 0.0) - probs_b.get(length, 0.0)) for length in lengths)


def jensen_shannon_divergence(
    probs_a: Dict[int, float],
    probs_b: Dict[int, float],
    lengths: List[int],
) -> float:
    def kl_divergence(p: Dict[int, float], q: Dict[int, float]) -> float:
        value = 0.0
        for length in lengths:
            p_value = p.get(length, 0.0)
            q_value = q.get(length, 0.0)
            if p_value <= 0.0 or q_value <= 0.0:
                continue
            value += p_value * math.log(p_value / q_value)
        return value

    midpoint = {
        length: 0.5 * (probs_a.get(length, 0.0) + probs_b.get(length, 0.0))
        for length in lengths
    }
    return 0.5 * kl_divergence(probs_a, midpoint) + 0.5 * kl_divergence(probs_b, midpoint)


def mean_length(rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["length"]) for row in rows) / len(rows)


def build_final_length_matched_negative_library(
    positive_rows: List[Dict[str, str]],
    negative_rows: List[Dict[str, str]],
    random_state: int = 42,
) -> Tuple[List[Dict[str, str]], Dict[str, object], List[Dict[str, str]]]:
    lengths = list(range(MIN_LEN, MAX_LEN + 1))
    positive_counts = length_counts(positive_rows)
    negative_counts = length_counts(negative_rows)
    total_positive = len(positive_rows)
    total_negative = len(negative_rows)
    exact_capacity = exact_length_matched_total(positive_counts, negative_counts)

    positive_probabilities = probabilities_from_counts(positive_counts, lengths)
    desired_negative_counts = largest_remainder_target_counts(
        positive_probabilities,
        total_target=total_positive,
    )
    selected_negative_counts, relocations = fit_counts_to_capacity_by_nearest_length(
        desired_negative_counts,
        negative_counts,
    )
    selected_negative_rows = sample_rows_by_length_counts(
        negative_rows,
        selected_negative_counts,
        random_state=random_state,
    )

    selected_negative_length_counts = length_counts(selected_negative_rows)
    negative_pool_probabilities = probabilities_from_counts(negative_counts, lengths)
    selected_negative_probabilities = probabilities_from_counts(selected_negative_length_counts, lengths)

    relocation_distance_total = sum(item["count"] * item["distance"] for item in relocations)
    relocated_total = sum(item["count"] for item in relocations)

    length_report_rows: List[Dict[str, str]] = []
    for length in lengths:
        length_report_rows.append(
            {
                "length": str(length),
                "positive_count": str(positive_counts.get(length, 0)),
                "positive_probability": f"{positive_probabilities.get(length, 0.0):.8f}",
                "negative_pool_count": str(negative_counts.get(length, 0)),
                "negative_pool_probability": f"{negative_pool_probabilities.get(length, 0.0):.8f}",
                "final_negative_count": str(selected_negative_length_counts.get(length, 0)),
                "final_negative_probability": f"{selected_negative_probabilities.get(length, 0.0):.8f}",
                "abs_probability_gap": f"{abs(positive_probabilities.get(length, 0.0) - selected_negative_probabilities.get(length, 0.0)):.8f}",
            }
        )

    report = {
        "strategy": "approximate_length_matching_with_fixed_negative_total_equal_to_positive_total",
        "positive_total": total_positive,
        "negative_pool_total_before_matching": total_negative,
        "exact_match_capacity": exact_capacity,
        "exact_match_capacity_ratio_vs_positive": round(exact_capacity / total_positive, 6) if total_positive else None,
        "chosen_negative_total": len(selected_negative_rows),
        "final_negative_to_positive_ratio": round(len(selected_negative_rows) / total_positive, 6) if total_positive else None,
        "positive_mean_length": round(mean_length(positive_rows), 6),
        "negative_pool_mean_length_before": round(mean_length(negative_rows), 6),
        "final_negative_mean_length_after": round(mean_length(selected_negative_rows), 6),
        "tv_distance_before": round(total_variation_distance(positive_probabilities, negative_pool_probabilities, lengths), 6),
        "tv_distance_after": round(total_variation_distance(positive_probabilities, selected_negative_probabilities, lengths), 6),
        "js_divergence_before": round(jensen_shannon_divergence(positive_probabilities, negative_pool_probabilities, lengths), 6),
        "js_divergence_after": round(jensen_shannon_divergence(positive_probabilities, selected_negative_probabilities, lengths), 6),
        "exact_length_bins_matched": sum(
            1 for length in lengths if selected_negative_length_counts.get(length, 0) == positive_counts.get(length, 0)
        ),
        "length_bin_count": len(lengths),
        "relocated_count": relocated_total,
        "relocation_distance_total": relocation_distance_total,
        "average_relocation_distance": round(relocation_distance_total / relocated_total, 6) if relocated_total else 0.0,
        "top_limiting_lengths": [
            {
                "length": length,
                "positive_count": positive_counts.get(length, 0),
                "negative_pool_count": negative_counts.get(length, 0),
            }
            for length in sorted(
                lengths,
                key=lambda item: (
                    positive_counts.get(item, 0) - min(positive_counts.get(item, 0), negative_counts.get(item, 0)),
                    positive_counts.get(item, 0),
                ),
                reverse=True,
            )[:10]
            if positive_counts.get(length, 0) > negative_counts.get(length, 0)
        ],
    }

    return selected_negative_rows, report, length_report_rows


def windows_path_to_wsl(path: Path) -> str:
    resolved = str(path.resolve())
    drive = resolved[0].lower()
    tail = resolved[2:].replace("\\", "/")
    return f"/mnt/{drive}{tail}"


def run_cdhit_wsl(
    cdhit_binary: Path,
    input_fasta: Path,
    output_prefix: Path,
    identity: float = 0.8,
    word_length: int = 5,
) -> Dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cdhit_wsl = windows_path_to_wsl(cdhit_binary)
    input_wsl = windows_path_to_wsl(input_fasta)
    output_wsl = windows_path_to_wsl(output_prefix)

    command = (
        f"'{cdhit_wsl}' -i '{input_wsl}' -o '{output_wsl}' "
        f"-c {identity} -n {word_length} -M 0 -T 0 -d 0"
    )
    subprocess.run(["wsl", "bash", "-lc", command], check=True)

    return {
        "clustered_fasta": str(output_prefix),
        "clustered_clstr": str(output_prefix) + ".clstr",
    }


def cluster_aware_train_val_test_split_rows(
    rows_with_ids: List[Dict[str, str]],
    cdhit_clstr_file_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, object]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    cluster_to_ids = parse_cdhit_clusters(cdhit_clstr_file_path)
    id_to_row = {row["fasta_id"]: row for row in rows_with_ids}

    clusters = []
    missing_ids = []
    for cluster_id, member_ids in cluster_to_ids.items():
        member_rows = []
        for member_id in member_ids:
            row = id_to_row.get(member_id)
            if row is None:
                missing_ids.append(member_id)
                continue
            member_rows.append(row)
        if member_rows:
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "rows": member_rows,
                    "size": len(member_rows),
                    "positive": sum(1 for row in member_rows if row["label"] == "positive"),
                    "negative": sum(1 for row in member_rows if row["label"] == "negative"),
                }
            )

    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} sequence IDs from row table. First few: {missing_ids[:10]}")

    covered_ids = {row["fasta_id"] for cluster in clusters for row in cluster["rows"]}
    uncovered_rows = [row for row in rows_with_ids if row["fasta_id"] not in covered_ids]
    for row in uncovered_rows:
        clusters.append(
            {
                "cluster_id": f"fallback_singleton::{row['fasta_id']}",
                "rows": [row],
                "size": 1,
                "positive": 1 if row["label"] == "positive" else 0,
                "negative": 1 if row["label"] == "negative" else 0,
            }
        )

    total = len(rows_with_ids)
    total_positive = sum(1 for row in rows_with_ids if row["label"] == "positive")
    total_negative = sum(1 for row in rows_with_ids if row["label"] == "negative")

    targets = {
        "train": {
            "total": total * train_ratio,
            "positive": total_positive * train_ratio,
            "negative": total_negative * train_ratio,
        },
        "val": {
            "total": total * val_ratio,
            "positive": total_positive * val_ratio,
            "negative": total_negative * val_ratio,
        },
        "test": {
            "total": total * test_ratio,
            "positive": total_positive * test_ratio,
            "negative": total_negative * test_ratio,
        },
    }

    rng = random.Random(random_state)
    rng.shuffle(clusters)
    clusters.sort(key=lambda cluster: (cluster["size"], cluster["positive"], cluster["negative"]), reverse=True)

    assigned_clusters = {"train": [], "val": [], "test": []}
    counts = {
        "train": {"total": 0, "positive": 0, "negative": 0},
        "val": {"total": 0, "positive": 0, "negative": 0},
        "test": {"total": 0, "positive": 0, "negative": 0},
    }

    for cluster in clusters:
        best_split = None
        best_score = None
        for split_name in ("train", "val", "test"):
            projected_counts = {
                name: counts[name].copy()
                for name in ("train", "val", "test")
            }
            projected_counts[split_name]["total"] += cluster["size"]
            projected_counts[split_name]["positive"] += cluster["positive"]
            projected_counts[split_name]["negative"] += cluster["negative"]

            score = 0.0
            for name in ("train", "val", "test"):
                score += abs(projected_counts[name]["total"] - targets[name]["total"]) / max(targets[name]["total"], 1)
                score += abs(projected_counts[name]["positive"] - targets[name]["positive"]) / max(targets[name]["positive"], 1)
                score += abs(projected_counts[name]["negative"] - targets[name]["negative"]) / max(targets[name]["negative"], 1)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assigned_clusters[best_split].append(cluster)
        counts[best_split]["total"] += cluster["size"]
        counts[best_split]["positive"] += cluster["positive"]
        counts[best_split]["negative"] += cluster["negative"]

    split_rows = {
        split_name: [row for cluster in assigned_clusters[split_name] for row in cluster["rows"]]
        for split_name in ("train", "val", "test")
    }

    split_summary = {}
    for split_name in ("train", "val", "test"):
        split_summary[split_name] = {
            "clusters": len(assigned_clusters[split_name]),
            "unique_sequences": len(split_rows[split_name]),
            "positive_unique": sum(1 for row in split_rows[split_name] if row["label"] == "positive"),
            "negative_unique": sum(1 for row in split_rows[split_name] if row["label"] == "negative"),
        }

    split_summary["fallback_singletons"] = len(uncovered_rows)
    split_summary["cluster_file_membership_count"] = len(covered_ids)
    split_summary["total_sequences_after_fallback"] = len(rows_with_ids)

    return split_rows, split_summary


def write_split_bundle(base_dir: Path, split_rows: Dict[str, List[Dict[str, str]]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        write_csv(base_dir / f"{split_name}.csv", rows)
        write_fasta(base_dir / f"{split_name}.fa", rows, header_field="fasta_id")


def prepare_policy_v5_db_dramp_uniprot(
    project_root: Path,
    data_root: Path,
    dbaasp_records: List[Dict[str, str]],
    dramp_records: List[Dict[str, str]],
    uniprot_records: List[Dict[str, str]],
) -> Dict[str, object]:
    base_dir = data_root / "benchmark_v5_db_dramp_uniprot"
    cdhit_dir = base_dir / "cdhit"
    splits_dir = base_dir / "splits"
    reports_dir = base_dir / "reports"
    cdhit_binary = project_root / "tools" / "cdhit" / "cd-hit"

    policy_records = dbaasp_records + dramp_records + uniprot_records
    conflicts, conflict_rows = build_conflict_report(policy_records)
    non_conflict_records = [record for record in policy_records if record["sequence"] not in conflicts]

    positive_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] in {"dbaasp", "dramp3"}
    )
    negative_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "uniprot"
    )
    internal_rows = aggregate_records(non_conflict_records)
    internal_rows_with_ids = with_label_id(assign_fasta_ids(internal_rows, prefix="v5seq"))

    write_csv(base_dir / "internal_benchmark_v5.csv", internal_rows_with_ids)
    write_csv(base_dir / "positive_v5.csv", positive_rows)
    write_fasta(base_dir / "positive_v5.fa", positive_rows)
    write_csv(base_dir / "negative_v5.csv", negative_rows)
    write_fasta(base_dir / "negative_v5.fa", negative_rows)
    write_cdhit_input_fasta(base_dir / "internal_for_cdhit_v5.fasta", internal_rows_with_ids)
    write_csv(reports_dir / "conflicting_sequences_v5.csv", conflict_rows)

    cdhit_paths = run_cdhit_wsl(
        cdhit_binary=cdhit_binary,
        input_fasta=base_dir / "internal_for_cdhit_v5.fasta",
        output_prefix=cdhit_dir / "internal_v5_c80",
        identity=0.8,
        word_length=5,
    )

    split_rows, split_summary = cluster_aware_train_val_test_split_rows(
        internal_rows_with_ids,
        cdhit_paths["clustered_clstr"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )
    write_split_bundle(splits_dir, split_rows)

    cluster_count = sum(1 for line in Path(cdhit_paths["clustered_clstr"]).read_text(encoding="utf-8", errors="ignore").splitlines() if line.startswith(">Cluster "))

    summary = {
        "policy": {
            "name": "v5_db_dramp_uniprot",
            "positives": ["DBAASP", "DRAMP3"],
            "negatives": ["UniProt reviewed short-peptide import"],
            "excluded_sources": ["AMPlify train", "AMPlify test"],
            "external_test_used": False,
        },
        "filter_rules": {
            "uppercase_after_whitespace_cleanup": True,
            "non_standard_sequence_policy": "drop_entire_sequence",
            "allowed_amino_acids": "".join(sorted(STANDARD_AA)),
            "length_range": [MIN_LEN, MAX_LEN],
            "drop_label_conflicts": True,
        },
        "outputs": {
            "internal_benchmark_v5": summarize_dataset(internal_rows_with_ids),
            "positive_pool_v5": summarize_dataset(positive_rows),
            "negative_pool_v5": summarize_dataset(negative_rows),
            "conflicting_sequences_removed": len(conflicts),
        },
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
            "internal_benchmark_v5_csv": str((base_dir / "internal_benchmark_v5.csv").relative_to(project_root)),
            "internal_for_cdhit_v5_fasta": str((base_dir / "internal_for_cdhit_v5.fasta").relative_to(project_root)),
            "train_csv": str((splits_dir / "train.csv").relative_to(project_root)),
            "val_csv": str((splits_dir / "val.csv").relative_to(project_root)),
            "test_csv": str((splits_dir / "test.csv").relative_to(project_root)),
            "conflict_report": str((reports_dir / "conflicting_sequences_v5.csv").relative_to(project_root)),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "benchmark_summary_v5.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def write_benchmark_report_v6(
    project_root: Path,
    base_dir: Path,
    summary: Dict[str, object],
    amplify_negative_stats: Dict[str, Dict[str, int]],
    newnegative_stats: Dict[str, Dict[str, int]],
    newnegative_files: List[str],
    uniprot_filename: str,
) -> Path:
    reports_dir = base_dir / "reports"
    outputs = summary["outputs"]
    clustering = summary["clustering"]
    splits = summary["splits"]
    policy = summary["policy"]

    def split_ratio(split_name: str) -> float:
        total = outputs["internal_benchmark_v6"]["unique_sequences"]
        count = splits[split_name]["unique_sequences"]
        return count / total if total else 0.0

    lines = [
        "AMP Benchmark Report V6",
        "Date: 2026-06-06",
        f"Project root: {project_root}",
        "",
        "1. Policy",
        "",
        "This V6 benchmark uses the following source policy only:",
        "1. Positive sources:",
        "   - DBAASP",
        "   - DRAMP3",
        "2. Negative sources:",
        "   - UniProt reviewed short-peptide import",
        "   - All AMPlify negative files",
        "   - newnegative user-added FASTA files",
        "3. AMPlify positive files:",
        "   - Excluded from this version",
        "4. External test set:",
        "   - Not used in this version",
        "5. Assumption for this version:",
        "   - Because AMPlify test is no longer reserved as an external benchmark, all AMPlify negative files were repurposed into the internal negative pool.",
        "",
        "2. Input files",
        "",
        "Positive raw inputs:",
        "- data\\amp\\dbaasp\\dbaasp1.csv ... dbaasp13.csv",
        "- data\\amp\\dramp3\\dramp.xlsx",
        "",
        "Negative raw inputs:",
        f"- data\\amp\\uniprot\\{uniprot_filename}",
        "- data\\amp\\amplify\\AMPlify_non_AMP_train_balanced.fa",
        "- data\\amp\\amplify\\AMPlify_non_AMP_train_imbalanced.fa",
        "- data\\amp\\amplify\\AMPlify_non_AMP_test_balanced.fa",
        "- data\\amp\\amplify\\AMPlify_non_AMP_test_imbalanced.fa",
    ]

    for filename in newnegative_files:
        lines.append(f"- data\\amp\\newnegative\\{filename}")

    lines.extend(
        [
            "",
            "3. Cleaning and filtering rules",
            "",
            "All sources were processed with the same strict rules:",
            "1. Remove whitespace",
            "2. Convert sequence to uppercase",
            "3. If any non-standard residue appears, drop the entire sequence",
            "4. Keep only sequences with length 10-50 aa",
            "5. Deduplicate after cleaning",
            "6. If the same cleaned sequence appears in both positive and negative pools, remove it completely from the benchmark",
            "",
            "Allowed amino acids:",
            "".join(sorted(STANDARD_AA)),
            "",
            "4. Positive labeling rules",
            "",
            "DBAASP:",
            "- Keep only rows with COMPLEXITY == Monomer",
            "- Keep only rows whose TARGET GROUP contains antimicrobial-related target keywords",
            "",
            "DRAMP3:",
            "- Keep only rows whose Activity contains antimicrobial-related keywords",
            "",
            "Negative sources:",
            "- UniProt, AMPlify negatives, and newnegative FASTA files are all treated as negative candidates in this policy version",
            "",
            "5. Source-local retention statistics after strict filtering",
            "",
            "UniProt:",
            f"- used_file = {uniprot_filename}",
            f"- kept_rows = {summary['source_local_stats']['uniprot']['kept_rows']}",
            f"- raw_rows = {summary['source_local_stats']['uniprot']['raw_rows']}",
            "",
            "AMPlify negative files:",
        ]
    )

    for filename, stats in amplify_negative_stats.items():
        lines.append(
            f"- {filename}: raw_rows = {stats.get('raw_rows', 0)}, kept_rows = {stats.get('kept_rows', 0)}, "
            f"skip_non_standard_sequence = {stats.get('skip_non_standard_sequence', 0)}, "
            f"skip_length_filter = {stats.get('skip_length_filter', 0)}"
        )

    lines.extend(["", "newnegative files:"])
    for filename, stats in newnegative_stats.items():
        lines.append(
            f"- {filename}: raw_rows = {stats.get('raw_rows', 0)}, kept_rows = {stats.get('kept_rows', 0)}, "
            f"skip_non_standard_sequence = {stats.get('skip_non_standard_sequence', 0)}, "
            f"skip_length_filter = {stats.get('skip_length_filter', 0)}"
        )

    lines.extend(
        [
            "",
            "6. Internal benchmark construction result",
            "",
            "Main file:",
            f"- {summary['artifacts']['internal_benchmark_v6_csv']}",
            "",
            "Statistics:",
            f"- unique_sequences = {outputs['internal_benchmark_v6']['unique_sequences']}",
            f"- positive_unique = {outputs['internal_benchmark_v6']['positive_unique']}",
            f"- negative_unique = {outputs['internal_benchmark_v6']['negative_unique']}",
            f"- support_sum = {outputs['internal_benchmark_v6']['support_sum']}",
            f"- duplicate_records_collapsed = {outputs['internal_benchmark_v6']['duplicate_records_collapsed']}",
            f"- negative_to_positive_ratio = {outputs['internal_benchmark_v6']['negative_to_positive_ratio']}",
            "",
            "Positive pool only:",
            f"- unique_sequences = {outputs['positive_pool_v6']['unique_sequences']}",
            f"- support_sum = {outputs['positive_pool_v6']['support_sum']}",
            f"- duplicate_records_collapsed = {outputs['positive_pool_v6']['duplicate_records_collapsed']}",
            "",
            "Negative pool only:",
            f"- unique_sequences = {outputs['negative_pool_v6']['unique_sequences']}",
            f"- support_sum = {outputs['negative_pool_v6']['support_sum']}",
            f"- duplicate_records_collapsed = {outputs['negative_pool_v6']['duplicate_records_collapsed']}",
            "",
            "Negative-source local breakdown after strict filtering and conflict removal:",
            f"- UniProt local unique = {outputs['uniprot_negative_local']['unique_sequences']}",
            f"- AMPlify local unique = {outputs['amplify_negative_local']['unique_sequences']}",
            f"- newnegative local unique = {outputs['newnegative_negative_local']['unique_sequences']}",
            "",
            "Conflict removal:",
            f"- conflicting_sequences_removed = {outputs['conflicting_sequences_removed']}",
            "",
            "7. Output directory layout",
            "",
            "Base directory:",
            f"- {summary['artifacts']['base_dir']}",
            "",
            "Main artifacts:",
            "- internal_benchmark_v6.csv",
            "- internal_for_cdhit_v6.fasta",
            "- positive_v6.csv",
            "- positive_v6.fa",
            "- negative_v6.csv",
            "- negative_v6.fa",
            "",
            "CD-HIT artifacts:",
            "- cdhit\\internal_v6_c80",
            "- cdhit\\internal_v6_c80.clstr",
            "",
            "Split artifacts:",
            "- splits\\train.csv",
            "- splits\\train.fa",
            "- splits\\val.csv",
            "- splits\\val.fa",
            "- splits\\test.csv",
            "- splits\\test.fa",
            "",
            "Reports:",
            "- reports\\benchmark_summary_v6.json",
            "- reports\\conflicting_sequences_v6.csv",
            "- reports\\benchmark_report_v6_20260606.txt",
            "",
            "8. CD-HIT execution",
            "",
            "Executable used:",
            "- tools\\cdhit\\cd-hit",
            "",
            "Clustering command parameters:",
            f"- identity_threshold = {clustering['identity_threshold']}",
            f"- word_length = {clustering['word_length']}",
            "- M = 0",
            "- T = 0",
            "- d = 0",
            "",
            "Cluster result:",
            f"- total input sequences to CD-HIT = {splits['cluster_file_membership_count']}",
            f"- cluster_count = {clustering['cluster_count']}",
            "",
            "Important note about short sequences:",
            f"- CD-HIT cluster membership covered {splits['cluster_file_membership_count']} sequences",
            f"- Total benchmark sequences after cleaning were {splits['total_sequences_after_fallback']}",
            f"- Therefore {splits['fallback_singletons']} sequences were not present in the .clstr membership output",
            "- These sequences were added back as fallback singleton clusters so they still participate in cluster-aware splitting without crossing splits",
            "",
            "9. Cluster-aware train/val/test split",
            "",
            "Split method:",
            "- Entire clusters were assigned to only one split",
            "- Split target ratios:",
            "  - train = 80%",
            "  - val = 10%",
            "  - test = 10%",
            "- Assignment uses a greedy global deviation score over:",
            "  - total sequence count",
            "  - positive count",
            "  - negative count",
            "",
            "Split summary:",
            "",
            "Train:",
            f"- clusters = {splits['train']['clusters']}",
            f"- unique_sequences = {splits['train']['unique_sequences']}",
            f"- positive_unique = {splits['train']['positive_unique']}",
            f"- negative_unique = {splits['train']['negative_unique']}",
            "",
            "Val:",
            f"- clusters = {splits['val']['clusters']}",
            f"- unique_sequences = {splits['val']['unique_sequences']}",
            f"- positive_unique = {splits['val']['positive_unique']}",
            f"- negative_unique = {splits['val']['negative_unique']}",
            "",
            "Test:",
            f"- clusters = {splits['test']['clusters']}",
            f"- unique_sequences = {splits['test']['unique_sequences']}",
            f"- positive_unique = {splits['test']['positive_unique']}",
            f"- negative_unique = {splits['test']['negative_unique']}",
            "",
            "Approximate split ratios by total sequence count:",
            f"- train = {splits['train']['unique_sequences']} / {outputs['internal_benchmark_v6']['unique_sequences']} = {split_ratio('train'):.4f}",
            f"- val = {splits['val']['unique_sequences']} / {outputs['internal_benchmark_v6']['unique_sequences']} = {split_ratio('val'):.4f}",
            f"- test = {splits['test']['unique_sequences']} / {outputs['internal_benchmark_v6']['unique_sequences']} = {split_ratio('test'):.4f}",
            "",
            "10. Key implementation file",
            "",
            "The current pipeline logic lives in:",
            "- src\\utils\\prepare_amp_benchmarks.py",
            "",
            "This script now includes:",
            "1. raw source loaders",
            "2. strict cleaning and conflict removal",
            "3. V5 benchmark generation",
            "4. V6 benchmark generation",
            "5. WSL-based CD-HIT execution",
            "6. cluster parsing",
            "7. cluster-aware train/val/test splitting",
            "",
            "11. Recommended files for another LLM",
            "",
            "If another LLM needs to continue from this version, give it these files first:",
            "1. src\\utils\\prepare_amp_benchmarks.py",
            f"2. {summary['artifacts']['internal_benchmark_v6_csv']}",
            f"3. {summary['artifacts']['internal_for_cdhit_v6_fasta']}",
            f"4. {summary['artifacts']['cluster_file']}",
            f"5. {summary['artifacts']['train_csv']}",
            f"6. {summary['artifacts']['val_csv']}",
            f"7. {summary['artifacts']['test_csv']}",
            f"8. {summary['artifacts']['summary_json']}",
            f"9. {summary['artifacts']['conflict_report']}",
            f"10. {summary['artifacts']['report_txt']}",
            "",
            "12. One-sentence handoff summary",
            "",
            "V6 is a fully rebuilt internal AMP benchmark using DBAASP+DRAMP3 positives and a combined UniProt+AMPlify+newnegative negative pool, processed with strict whole-sequence filtering and global positive/negative conflict removal, then clustered by CD-HIT at 80% identity and split cluster-wise into train/val/test.",
        ]
    )

    report_path = reports_dir / "benchmark_report_v6_20260606.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def prepare_policy_v6_db_dramp_all_negative(
    project_root: Path,
    data_root: Path,
    dbaasp_records: List[Dict[str, str]],
    dramp_records: List[Dict[str, str]],
    uniprot_records: List[Dict[str, str]],
    uniprot_stats: Dict[str, int],
    amplify_negative_records: List[Dict[str, str]],
    amplify_negative_stats: Dict[str, Dict[str, int]],
    newnegative_records: List[Dict[str, str]],
    newnegative_stats: Dict[str, Dict[str, int]],
    newnegative_files: List[str],
    uniprot_filename: str,
) -> Dict[str, object]:
    base_dir = data_root / "benchmark_v6_db_dramp_all_negative"
    cdhit_dir = base_dir / "cdhit"
    splits_dir = base_dir / "splits"
    reports_dir = base_dir / "reports"
    cdhit_binary = project_root / "tools" / "cdhit" / "cd-hit"

    policy_records = (
        dbaasp_records
        + dramp_records
        + uniprot_records
        + amplify_negative_records
        + newnegative_records
    )
    conflicts, conflict_rows = build_conflict_report(policy_records)
    non_conflict_records = [record for record in policy_records if record["sequence"] not in conflicts]

    positive_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] in {"dbaasp", "dramp3"}
    )
    negative_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["label"] == "negative"
    )
    uniprot_negative_local = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "uniprot"
    )
    amplify_negative_local = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "amplify"
    )
    newnegative_negative_local = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "newnegative"
    )
    internal_rows = aggregate_records(non_conflict_records)
    internal_rows_with_ids = with_label_id(assign_fasta_ids(internal_rows, prefix="v6seq"))

    write_csv(base_dir / "internal_benchmark_v6.csv", internal_rows_with_ids)
    write_csv(base_dir / "positive_v6.csv", positive_rows)
    write_fasta(base_dir / "positive_v6.fa", positive_rows)
    write_csv(base_dir / "negative_v6.csv", negative_rows)
    write_fasta(base_dir / "negative_v6.fa", negative_rows)
    write_cdhit_input_fasta(base_dir / "internal_for_cdhit_v6.fasta", internal_rows_with_ids)
    write_csv(reports_dir / "conflicting_sequences_v6.csv", conflict_rows)

    cdhit_paths = run_cdhit_wsl(
        cdhit_binary=cdhit_binary,
        input_fasta=base_dir / "internal_for_cdhit_v6.fasta",
        output_prefix=cdhit_dir / "internal_v6_c80",
        identity=0.8,
        word_length=5,
    )

    split_rows, split_summary = cluster_aware_train_val_test_split_rows(
        internal_rows_with_ids,
        cdhit_paths["clustered_clstr"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )
    write_split_bundle(splits_dir, split_rows)

    cluster_count = sum(
        1
        for line in Path(cdhit_paths["clustered_clstr"]).read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
        if line.startswith(">Cluster ")
    )

    summary = {
        "policy": {
            "name": "v6_db_dramp_all_negative",
            "positives": ["DBAASP", "DRAMP3"],
            "negatives": [
                "UniProt reviewed short-peptide import",
                "AMPlify non-AMP train/test balanced/imbalanced files",
                "newnegative user-added FASTA files",
            ],
            "excluded_sources": ["All AMPlify positive files"],
            "external_test_used": False,
            "assumption": (
                "AMPlify test negatives were repurposed into the internal negative pool because "
                "AMPlify test is no longer being reserved as an external benchmark in this version."
            ),
        },
        "filter_rules": {
            "uppercase_after_whitespace_cleanup": True,
            "non_standard_sequence_policy": "drop_entire_sequence",
            "allowed_amino_acids": "".join(sorted(STANDARD_AA)),
            "length_range": [MIN_LEN, MAX_LEN],
            "drop_label_conflicts": True,
        },
        "source_local_stats": {
            "uniprot": uniprot_stats,
            "amplify_negative_files": amplify_negative_stats,
            "newnegative_files": newnegative_stats,
        },
        "outputs": {
            "internal_benchmark_v6": summarize_dataset(internal_rows_with_ids),
            "positive_pool_v6": summarize_dataset(positive_rows),
            "negative_pool_v6": summarize_dataset(negative_rows),
            "uniprot_negative_local": summarize_dataset(uniprot_negative_local),
            "amplify_negative_local": summarize_dataset(amplify_negative_local),
            "newnegative_negative_local": summarize_dataset(newnegative_negative_local),
            "conflicting_sequences_removed": len(conflicts),
        },
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
            "internal_benchmark_v6_csv": str((base_dir / "internal_benchmark_v6.csv").relative_to(project_root)),
            "internal_for_cdhit_v6_fasta": str((base_dir / "internal_for_cdhit_v6.fasta").relative_to(project_root)),
            "train_csv": str((splits_dir / "train.csv").relative_to(project_root)),
            "val_csv": str((splits_dir / "val.csv").relative_to(project_root)),
            "test_csv": str((splits_dir / "test.csv").relative_to(project_root)),
            "cluster_file": str((cdhit_dir / "internal_v6_c80.clstr").relative_to(project_root)),
            "summary_json": str((reports_dir / "benchmark_summary_v6.json").relative_to(project_root)),
            "conflict_report": str((reports_dir / "conflicting_sequences_v6.csv").relative_to(project_root)),
            "report_txt": str((reports_dir / "benchmark_report_v6_20260606.txt").relative_to(project_root)),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "benchmark_summary_v6.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path = write_benchmark_report_v6(
        project_root=project_root,
        base_dir=base_dir,
        summary=summary,
        amplify_negative_stats=amplify_negative_stats,
        newnegative_stats=newnegative_stats,
        newnegative_files=newnegative_files,
        uniprot_filename=uniprot_filename,
    )
    summary["artifacts"]["report_txt"] = str(report_path.relative_to(project_root))
    with (reports_dir / "benchmark_summary_v6.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def write_final_open_release_report(
    project_root: Path,
    base_dir: Path,
    summary: Dict[str, object],
) -> Path:
    reports_dir = base_dir / "reports"
    outputs = summary["outputs"]
    matching = summary["length_matching"]
    clustering = summary["clustering"]
    splits = summary["splits"]

    def split_ratio(split_name: str) -> float:
        total = outputs["final_benchmark"]["unique_sequences"]
        count = splits[split_name]["unique_sequences"]
        return count / total if total else 0.0

    positive_total = outputs["positive_pool"]["unique_sequences"]
    negative_before_total = outputs["negative_pool_before_matching"]["unique_sequences"]
    negative_after_total = outputs["negative_pool_final"]["unique_sequences"]

    before_by_source = summary["negative_source_breakdown"]["before_matching_by_signature"]
    after_by_source = summary["negative_source_breakdown"]["after_matching_by_signature"]

    lines = [
        "AMP Final Open Release Report",
        "Date: 2026-06-06",
        f"Project root: {project_root}",
        "",
        "1. Final release policy",
        "",
        "Positive sources:",
        "- DBAASP",
        "- DRAMP3",
        "",
        "Negative candidate sources:",
        "- UniProt reviewed short-peptide import",
        "- All AMPlify negative files",
        "- All FASTA files in data\\amp\\newnegative",
        "",
        "Excluded from the final release:",
        "- All AMPlify positive files",
        "- All AMPlify external-test semantics; AMPlify negatives are treated only as internal negative candidates here",
        "",
        "2. Common cleaning rules",
        "",
        "All sources were processed with the same strict rules:",
        "1. Remove whitespace",
        "2. Convert sequence to uppercase",
        "3. If any non-standard residue appears, drop the entire sequence",
        "4. Keep only sequences with length 10-50 aa",
        "5. Deduplicate after cleaning",
        "6. If the same cleaned sequence appears in both positive and negative pools, remove it completely from the release",
        "",
        "Allowed amino acids:",
        "".join(sorted(STANDARD_AA)),
        "",
        "3. Why the final negative library uses approximate matching instead of exact matching",
        "",
        (
            f"The candidate negative pool is large enough overall ({matching['negative_pool_total_before_matching']} rows), "
            f"but it is still short on several AMP-like short lengths. If exact per-length matching were enforced, "
            f"the negative library would collapse to only {matching['exact_match_capacity']} rows, which is "
            f"{matching['exact_match_capacity_ratio_vs_positive']}x of the positive pool."
        ),
        "",
        "Therefore the final open-release data uses this strategy:",
        "- Fix the final negative total to equal the positive total",
        "- Use the positive empirical length distribution as the target probability distribution",
        "- Fit the negative histogram as closely as possible under available capacities",
        "- When a target length is undersupplied, reassign that mass to the nearest available lengths",
        "",
        "4. Final benchmark result",
        "",
        f"- final_unique_sequences = {outputs['final_benchmark']['unique_sequences']}",
        f"- final_positive_unique = {outputs['final_benchmark']['positive_unique']}",
        f"- final_negative_unique = {outputs['final_benchmark']['negative_unique']}",
        f"- final_negative_to_positive_ratio = {outputs['final_benchmark']['negative_to_positive_ratio']}",
        "",
        "Positive pool before length matching:",
        f"- unique_sequences = {outputs['positive_pool']['unique_sequences']}",
        "",
        "Negative candidate pool before length matching:",
        f"- unique_sequences = {outputs['negative_pool_before_matching']['unique_sequences']}",
        f"- negative_to_positive_ratio_vs_positive_pool = {round(negative_before_total / positive_total, 6) if positive_total else 0.0}",
        "",
        "Final negative library after length matching:",
        f"- unique_sequences = {outputs['negative_pool_final']['unique_sequences']}",
        f"- negative_to_positive_ratio_vs_positive_pool = {round(negative_after_total / positive_total, 6) if positive_total else 0.0}",
        "",
        f"- conflicting_sequences_removed = {outputs['conflicting_sequences_removed']}",
        "",
        "5. Length-distribution matching diagnostics",
        "",
        f"- positive_mean_length = {matching['positive_mean_length']}",
        f"- negative_pool_mean_length_before = {matching['negative_pool_mean_length_before']}",
        f"- final_negative_mean_length_after = {matching['final_negative_mean_length_after']}",
        f"- total_variation_distance_before = {matching['tv_distance_before']}",
        f"- total_variation_distance_after = {matching['tv_distance_after']}",
        f"- js_divergence_before = {matching['js_divergence_before']}",
        f"- js_divergence_after = {matching['js_divergence_after']}",
        f"- exact_length_bins_matched = {matching['exact_length_bins_matched']} / {matching['length_bin_count']}",
        f"- relocated_count = {matching['relocated_count']}",
        f"- average_relocation_distance = {matching['average_relocation_distance']}",
        "",
        "Top limiting lengths in the negative pool:",
    ]

    for item in matching["top_limiting_lengths"]:
        lines.append(
            f"- length {item['length']}: positive_count = {item['positive_count']}, "
            f"negative_pool_count = {item['negative_pool_count']}"
        )

    lines.extend(
        [
            "",
            "6. Negative-source composition",
            "",
            "Before length matching (non-exclusive signature counts):",
        ]
    )
    for signature, count in sorted(before_by_source.items()):
        lines.append(f"- {signature}: {count}")

    lines.extend(["", "After length matching (non-exclusive signature counts):"])
    for signature, count in sorted(after_by_source.items()):
        lines.append(f"- {signature}: {count}")

    lines.extend(
        [
            "",
            "7. CD-HIT clustering and split",
            "",
            f"- cluster_count = {clustering['cluster_count']}",
            f"- cluster_file_membership_count = {splits['cluster_file_membership_count']}",
            f"- fallback_singletons = {splits['fallback_singletons']}",
            "",
            "Train:",
            f"- unique_sequences = {splits['train']['unique_sequences']}",
            f"- positive_unique = {splits['train']['positive_unique']}",
            f"- negative_unique = {splits['train']['negative_unique']}",
            "",
            "Val:",
            f"- unique_sequences = {splits['val']['unique_sequences']}",
            f"- positive_unique = {splits['val']['positive_unique']}",
            f"- negative_unique = {splits['val']['negative_unique']}",
            "",
            "Test:",
            f"- unique_sequences = {splits['test']['unique_sequences']}",
            f"- positive_unique = {splits['test']['positive_unique']}",
            f"- negative_unique = {splits['test']['negative_unique']}",
            "",
            "Approximate split ratios by total sequence count:",
            f"- train = {splits['train']['unique_sequences']} / {outputs['final_benchmark']['unique_sequences']} = {split_ratio('train'):.4f}",
            f"- val = {splits['val']['unique_sequences']} / {outputs['final_benchmark']['unique_sequences']} = {split_ratio('val'):.4f}",
            f"- test = {splits['test']['unique_sequences']} / {outputs['final_benchmark']['unique_sequences']} = {split_ratio('test'):.4f}",
            "",
            "8. Final directory layout",
            "",
            f"- base_dir = {summary['artifacts']['base_dir']}",
            f"- benchmark_csv = {summary['artifacts']['benchmark_csv']}",
            f"- positive_csv = {summary['artifacts']['positive_csv']}",
            f"- negative_csv = {summary['artifacts']['negative_csv']}",
            f"- cdhit_input_fasta = {summary['artifacts']['cdhit_input_fasta']}",
            f"- cluster_file = {summary['artifacts']['cluster_file']}",
            f"- train_csv = {summary['artifacts']['train_csv']}",
            f"- val_csv = {summary['artifacts']['val_csv']}",
            f"- test_csv = {summary['artifacts']['test_csv']}",
            f"- length_distribution_report = {summary['artifacts']['length_distribution_report']}",
            f"- summary_json = {summary['artifacts']['summary_json']}",
            f"- conflict_report = {summary['artifacts']['conflict_report']}",
            "",
            "9. One-sentence handoff summary",
            "",
            "This final open-release dataset uses DBAASP+DRAMP3 positives and a pooled UniProt+AMPlify+newnegative negative candidate set, then builds a 1:1 final negative library by approximate length-distribution matching before CD-HIT 80% cluster-aware train/val/test splitting.",
        ]
    )

    report_path = reports_dir / "final_open_release_report_20260606.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def prepare_final_open_release(
    project_root: Path,
    data_root: Path,
    dbaasp_records: List[Dict[str, str]],
    dramp_records: List[Dict[str, str]],
    uniprot_records: List[Dict[str, str]],
    uniprot_stats: Dict[str, int],
    amplify_negative_records: List[Dict[str, str]],
    amplify_negative_stats: Dict[str, Dict[str, int]],
    newnegative_records: List[Dict[str, str]],
    newnegative_stats: Dict[str, Dict[str, int]],
    newnegative_files: List[str],
    uniprot_filename: str,
) -> Dict[str, object]:
    base_dir = project_root / "data--final"
    cdhit_dir = base_dir / "cdhit"
    splits_dir = base_dir / "splits"
    reports_dir = base_dir / "reports"
    cdhit_binary = project_root / "tools" / "cdhit" / "cd-hit"

    policy_records = (
        dbaasp_records
        + dramp_records
        + uniprot_records
        + amplify_negative_records
        + newnegative_records
    )
    conflicts, conflict_rows = build_conflict_report(policy_records)
    non_conflict_records = [record for record in policy_records if record["sequence"] not in conflicts]

    positive_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] in {"dbaasp", "dramp3"}
    )
    negative_candidate_rows = aggregate_records(
        record for record in non_conflict_records if record["label"] == "negative"
    )

    final_negative_rows, length_matching_report, length_distribution_rows = build_final_length_matched_negative_library(
        positive_rows=positive_rows,
        negative_rows=negative_candidate_rows,
        random_state=42,
    )
    final_negative_rows = sorted(final_negative_rows, key=lambda row: row["sequence"])

    final_rows = sorted(
        positive_rows + final_negative_rows,
        key=lambda row: (0 if row["label"] == "positive" else 1, row["sequence"]),
    )
    final_rows_with_ids = with_label_id(assign_fasta_ids(final_rows, prefix="finalseq"))

    write_csv(base_dir / "internal_benchmark_final.csv", final_rows_with_ids)
    write_csv(base_dir / "positive_final.csv", positive_rows)
    write_fasta(base_dir / "positive_final.fa", positive_rows)
    write_csv(base_dir / "negative_final.csv", final_negative_rows)
    write_fasta(base_dir / "negative_final.fa", final_negative_rows)
    write_cdhit_input_fasta(base_dir / "internal_for_cdhit_final.fasta", final_rows_with_ids)
    write_csv(reports_dir / "conflicting_sequences_final.csv", conflict_rows)
    write_csv(reports_dir / "length_distribution_matching_final.csv", length_distribution_rows)

    cdhit_paths = run_cdhit_wsl(
        cdhit_binary=cdhit_binary,
        input_fasta=base_dir / "internal_for_cdhit_final.fasta",
        output_prefix=cdhit_dir / "internal_final_c80",
        identity=0.8,
        word_length=5,
    )

    split_rows, split_summary = cluster_aware_train_val_test_split_rows(
        final_rows_with_ids,
        cdhit_paths["clustered_clstr"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )
    write_split_bundle(splits_dir, split_rows)

    cluster_count = sum(
        1
        for line in Path(cdhit_paths["clustered_clstr"]).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith(">Cluster ")
    )

    before_signature_counts = Counter(source_signature(row) for row in negative_candidate_rows)
    after_signature_counts = Counter(source_signature(row) for row in final_negative_rows)

    summary = {
        "policy": {
            "name": "final_open_release",
            "positives": ["DBAASP", "DRAMP3"],
            "negative_candidates": [
                "UniProt reviewed short-peptide import",
                "AMPlify negative files",
                "newnegative user-added FASTA files",
            ],
            "negative_library_strategy": length_matching_report["strategy"],
            "external_test_used": False,
            "notes": (
                "The final release fixes the negative total to equal the positive total and then fits the negative "
                "length histogram to the positive histogram as closely as possible under available capacities."
            ),
        },
        "filter_rules": {
            "uppercase_after_whitespace_cleanup": True,
            "non_standard_sequence_policy": "drop_entire_sequence",
            "allowed_amino_acids": "".join(sorted(STANDARD_AA)),
            "length_range": [MIN_LEN, MAX_LEN],
            "drop_label_conflicts": True,
        },
        "source_local_stats": {
            "uniprot_file": uniprot_filename,
            "uniprot_stats": uniprot_stats,
            "amplify_negative_stats": amplify_negative_stats,
            "newnegative_files": newnegative_files,
            "newnegative_stats": newnegative_stats,
        },
        "outputs": {
            "final_benchmark": summarize_dataset(final_rows_with_ids),
            "positive_pool": summarize_dataset(positive_rows),
            "negative_pool_before_matching": summarize_dataset(negative_candidate_rows),
            "negative_pool_final": summarize_dataset(final_negative_rows),
            "conflicting_sequences_removed": len(conflicts),
        },
        "length_matching": length_matching_report,
        "negative_source_breakdown": {
            "before_matching_by_signature": dict(before_signature_counts),
            "after_matching_by_signature": dict(after_signature_counts),
        },
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
            "benchmark_csv": str((base_dir / "internal_benchmark_final.csv").relative_to(project_root)),
            "positive_csv": str((base_dir / "positive_final.csv").relative_to(project_root)),
            "negative_csv": str((base_dir / "negative_final.csv").relative_to(project_root)),
            "cdhit_input_fasta": str((base_dir / "internal_for_cdhit_final.fasta").relative_to(project_root)),
            "cluster_file": str((cdhit_dir / "internal_final_c80.clstr").relative_to(project_root)),
            "train_csv": str((splits_dir / "train.csv").relative_to(project_root)),
            "val_csv": str((splits_dir / "val.csv").relative_to(project_root)),
            "test_csv": str((splits_dir / "test.csv").relative_to(project_root)),
            "length_distribution_report": str((reports_dir / "length_distribution_matching_final.csv").relative_to(project_root)),
            "summary_json": str((reports_dir / "final_open_release_summary.json").relative_to(project_root)),
            "conflict_report": str((reports_dir / "conflicting_sequences_final.csv").relative_to(project_root)),
            "report_txt": str((reports_dir / "final_open_release_report_20260606.txt").relative_to(project_root)),
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "final_open_release_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path = write_final_open_release_report(
        project_root=project_root,
        base_dir=base_dir,
        summary=summary,
    )
    summary["artifacts"]["report_txt"] = str(report_path.relative_to(project_root))
    with (reports_dir / "final_open_release_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def parse_cdhit_clusters(cdhit_clstr_file_path: str) -> Dict[str, List[str]]:
    cluster_to_ids: Dict[str, List[str]] = defaultdict(list)
    current_cluster_id = None

    with open(cdhit_clstr_file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">Cluster "):
                current_cluster_id = line.replace(">Cluster ", "").strip()
                continue

            if current_cluster_id is None:
                raise ValueError("Encountered sequence line before any cluster header.")

            match = re.search(r">(.+?)\.\.\.", line)
            if not match:
                match = re.search(r">(\S+)", line)
            if not match:
                raise ValueError(f"Could not parse sequence identifier from CD-HIT line: {line}")

            cluster_to_ids[current_cluster_id].append(match.group(1))

    return dict(cluster_to_ids)


def cluster_aware_train_val_test_split(
    df,
    cdhit_clstr_file_path,
    id_column: str = "fasta_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    """
    Split a pandas-like DataFrame into train/val/test by assigning entire CD-HIT clusters.

    Requirements:
    1. `df` must contain a unique identifier column matching the FASTA headers used for CD-HIT.
    2. `cdhit_clstr_file_path` must point to a .clstr file generated from the same FASTA.
    3. No cluster will span more than one split.

    Returns:
    (train_df, val_df, test_df)
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    columns = set(getattr(df, "columns", []))
    if id_column not in columns:
        raise ValueError(f"DataFrame must contain the identifier column {id_column!r}")

    cluster_to_ids = parse_cdhit_clusters(cdhit_clstr_file_path)
    id_to_cluster: Dict[str, str] = {}
    for cluster_id, member_ids in cluster_to_ids.items():
        for member_id in member_ids:
            if member_id in id_to_cluster:
                raise ValueError(f"Sequence identifier {member_id!r} appears in multiple clusters.")
            id_to_cluster[member_id] = cluster_id

    split_df = df.copy()
    split_df["cluster_id"] = split_df[id_column].map(id_to_cluster)
    if split_df["cluster_id"].isna().any():
        missing_ids = split_df.loc[split_df["cluster_id"].isna(), id_column].tolist()
        preview = missing_ids[:10]
        raise ValueError(
            "Some sequence identifiers are missing from the CD-HIT cluster file. "
            f"First missing identifiers: {preview}"
        )

    cluster_sizes = split_df.groupby("cluster_id").size().to_dict()
    rng = random.Random(random_state)
    cluster_items = list(cluster_sizes.items())
    rng.shuffle(cluster_items)
    cluster_items.sort(key=lambda item: item[1], reverse=True)

    total_sequences = len(split_df)
    targets = {
        "train": total_sequences * train_ratio,
        "val": total_sequences * val_ratio,
        "test": total_sequences * test_ratio,
    }
    assignments: Dict[str, str] = {}
    current_counts = {"train": 0, "val": 0, "test": 0}

    for cluster_id, cluster_size in cluster_items:
        best_split = None
        best_score = None

        for split_name in ("train", "val", "test"):
            projected_counts = current_counts.copy()
            projected_counts[split_name] += cluster_size
            score = sum(abs(projected_counts[name] - targets[name]) for name in projected_counts)

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assignments[cluster_id] = best_split
        current_counts[best_split] += cluster_size

    split_df["split"] = split_df["cluster_id"].map(assignments)

    train_df = split_df.loc[split_df["split"] == "train"].copy()
    val_df = split_df.loc[split_df["split"] == "val"].copy()
    test_df = split_df.loc[split_df["split"] == "test"].copy()

    return train_df, val_df, test_df


def prepare_amp_benchmarks() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "amp"
    positive_dir = data_root / "positive"
    negative_dir = data_root / "negative"
    report_dir = data_root / "reports"

    dbaasp_records, dbaasp_stats = load_dbaasp_records(data_root)
    dramp_records, dramp_stats = load_dramp_records(data_root)
    amplify_records, amplify_stats, ignored_amplify_files = load_amplify_records(data_root)
    uniprot_records, uniprot_stats, uniprot_filename = load_uniprot_negative_records(data_root)
    amplify_negative_all_records, amplify_negative_all_stats = load_amplify_negative_records_all(data_root)
    newnegative_records, newnegative_stats, newnegative_files = load_new_negative_records(data_root)

    all_records = dbaasp_records + dramp_records + amplify_records + uniprot_records
    conflicts, conflict_rows = build_conflict_report(all_records)
    non_conflict_records = [record for record in all_records if record["sequence"] not in conflicts]

    dbaasp_positive_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "dbaasp"
    )
    dramp3_positive_rows = aggregate_records(
        record for record in non_conflict_records if record["source_db"] == "dramp3"
    )
    amplify_train_positive_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_file"] == "AMPlify_AMP_train_common.fa"
    )
    amplify_test_positive_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_file"] == "AMPlify_AMP_test_common.fa"
    )
    amplify_train_negative_balanced_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_file"] == "AMPlify_non_AMP_train_balanced.fa"
    )
    amplify_train_negative_imbalanced_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_file"] == "AMPlify_non_AMP_train_imbalanced.fa"
    )
    amplify_train_negative_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["label"] == "negative" and record["source_db"] == "amplify" and record["source_split"] == "internal_train"
    )
    amplify_test_negative_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_file"] == "AMPlify_non_AMP_test_balanced.fa"
    )
    uniprot_negative_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_db"] == "uniprot" and record["source_split"] == "internal_train"
    )

    internal_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_split"] in {"internal", "internal_train"}
    )
    external_rows = aggregate_records(
        record
        for record in non_conflict_records
        if record["source_split"] == "external_test"
    )
    internal_rows_with_ids = assign_fasta_ids(internal_rows, prefix="seq")
    internal_rows_with_ids = with_label_id(internal_rows_with_ids)
    matched_internal_rows_with_ids, length_match_report = build_length_matched_internal_benchmark(
        internal_rows_with_ids,
        random_state=42,
    )

    write_csv(positive_dir / "dbaasp_positive.csv", dbaasp_positive_rows)
    write_fasta(positive_dir / "dbaasp_positive.fa", dbaasp_positive_rows)
    write_csv(positive_dir / "dramp3_positive.csv", dramp3_positive_rows)
    write_fasta(positive_dir / "dramp3_positive.fa", dramp3_positive_rows)
    write_csv(positive_dir / "amplify_train_positive.csv", amplify_train_positive_rows)
    write_fasta(positive_dir / "amplify_train_positive.fa", amplify_train_positive_rows)
    write_csv(positive_dir / "amplify_test_positive_external.csv", amplify_test_positive_rows)
    write_fasta(positive_dir / "amplify_test_positive_external.fa", amplify_test_positive_rows)
    write_csv(positive_dir / "internal_benchmark_positive.csv", [row for row in internal_rows if row["label"] == "positive"])
    write_fasta(positive_dir / "internal_benchmark_positive.fa", [row for row in internal_rows if row["label"] == "positive"])
    write_csv(positive_dir / "external_test_positive.csv", [row for row in external_rows if row["label"] == "positive"])
    write_fasta(positive_dir / "external_test_positive.fa", [row for row in external_rows if row["label"] == "positive"])

    write_csv(negative_dir / "amplify_train_negative_balanced.csv", amplify_train_negative_balanced_rows)
    write_fasta(negative_dir / "amplify_train_negative_balanced.fa", amplify_train_negative_balanced_rows)
    write_csv(negative_dir / "amplify_train_negative_imbalanced.csv", amplify_train_negative_imbalanced_rows)
    write_fasta(negative_dir / "amplify_train_negative_imbalanced.fa", amplify_train_negative_imbalanced_rows)
    write_csv(negative_dir / "amplify_train_negative.csv", amplify_train_negative_rows)
    write_fasta(negative_dir / "amplify_train_negative.fa", amplify_train_negative_rows)
    write_csv(negative_dir / "uniprot_reviewed_negative.csv", uniprot_negative_rows)
    write_fasta(negative_dir / "uniprot_reviewed_negative.fa", uniprot_negative_rows)
    write_csv(negative_dir / "amplify_test_negative_external.csv", amplify_test_negative_rows)
    write_fasta(negative_dir / "amplify_test_negative_external.fa", amplify_test_negative_rows)
    write_csv(negative_dir / "internal_benchmark_negative.csv", [row for row in internal_rows if row["label"] == "negative"])
    write_fasta(negative_dir / "internal_benchmark_negative.fa", [row for row in internal_rows if row["label"] == "negative"])
    write_csv(negative_dir / "external_test_negative.csv", [row for row in external_rows if row["label"] == "negative"])
    write_fasta(negative_dir / "external_test_negative.fa", [row for row in external_rows if row["label"] == "negative"])

    write_csv(data_root / "internal_benchmark.csv", internal_rows)
    write_csv(data_root / "internal_benchmark_v2.csv", internal_rows_with_ids)
    write_csv(data_root / "internal_benchmark_v3.csv", internal_rows_with_ids)
    write_csv(data_root / "internal_benchmark_v4_length_matched.csv", matched_internal_rows_with_ids)
    write_csv(data_root / "external_test.csv", external_rows)
    write_cdhit_input_fasta(data_root / "internal_for_cdhit.fasta", internal_rows_with_ids)
    write_cdhit_input_fasta(data_root / "internal_for_cdhit_v3.fasta", internal_rows_with_ids)
    write_cdhit_input_fasta(data_root / "internal_for_cdhit_v4_length_matched.fasta", matched_internal_rows_with_ids)

    write_csv(report_dir / "conflicting_sequences.csv", conflict_rows)
    write_csv(report_dir / "conflicting_sequences_v2.csv", conflict_rows)
    write_csv(report_dir / "conflicting_sequences_v3.csv", conflict_rows)
    length_distribution_rows = []
    positive_length_counts = length_match_report["positive_length_counts"]
    negative_length_counts_before = length_match_report["negative_length_counts_before"]
    negative_length_counts_after = length_match_report["negative_length_counts_after"]
    for length in range(MIN_LEN, MAX_LEN + 1):
        length_distribution_rows.append(
            {
                "length": str(length),
                "positive_count": str(positive_length_counts.get(str(length), 0)),
                "negative_count_before": str(negative_length_counts_before.get(str(length), 0)),
                "negative_count_after": str(negative_length_counts_after.get(str(length), 0)),
            }
        )
    write_csv(report_dir / "length_distribution_matching_v4.csv", length_distribution_rows)

    summary = {
        "label_audit": {
            "dbaasp": {
                "label_field": "TARGET GROUP",
                "notes": (
                    "DBAASP does not expose a binary AMP label here; positive candidates were kept only "
                    "when TARGET GROUP contains antimicrobial-related targets and COMPLEXITY is Monomer."
                ),
                "stats": dbaasp_stats,
            },
            "dramp3": {
                "label_field": "Activity",
                "notes": (
                    "DRAMP3 general_amps was filtered conservatively to rows whose Activity contains "
                    "antimicrobial-related keywords."
                ),
                "stats": dramp_stats,
            },
            "amplify": {
                "label_field": "Filename-derived label",
                "notes": (
                    "The internal negative pool includes both the balanced and imbalanced train negatives. "
                    "The external test remains separated and uses the balanced test files."
                ),
                "used_files": list(amplify_stats.keys()),
                "ignored_files": ignored_amplify_files,
                "stats": amplify_stats,
            },
            "uniprot": {
                "label_field": "Filename-derived negative source from UniProt query results",
                "notes": (
                    "Reviewed UniProt entries from the user-supplied query were treated as internal negatives "
                    "and processed with the same strict whole-sequence filtering, deduplication, and conflict removal rules."
                ),
                "used_file": uniprot_filename,
                "stats": uniprot_stats,
            },
        },
        "filter_rules": {
            "uppercase_after_whitespace_cleanup": True,
            "non_standard_sequence_policy": "drop_entire_sequence",
            "allowed_amino_acids": "".join(sorted(STANDARD_AA)),
            "length_range": [MIN_LEN, MAX_LEN],
            "drop_label_conflicts": True,
        },
        "outputs": {
            "internal_benchmark": summarize_dataset(internal_rows),
            "internal_benchmark_v2": summarize_dataset(internal_rows_with_ids),
            "internal_benchmark_v4_length_matched": summarize_dataset(matched_internal_rows_with_ids),
            "external_test": summarize_dataset(external_rows),
            "dbaasp_positive": summarize_dataset(dbaasp_positive_rows),
            "dramp3_positive": summarize_dataset(dramp3_positive_rows),
            "amplify_train_positive": summarize_dataset(amplify_train_positive_rows),
            "amplify_train_negative_union": summarize_dataset(amplify_train_negative_rows),
            "amplify_train_negative_balanced": summarize_dataset(amplify_train_negative_balanced_rows),
            "amplify_train_negative_imbalanced": summarize_dataset(amplify_train_negative_imbalanced_rows),
            "uniprot_negative": summarize_dataset(uniprot_negative_rows),
            "conflicting_sequences_removed": len(conflicts),
            "cdhit_input_sequences": len(internal_rows_with_ids),
        },
        "length_distribution_matching": {
            **length_match_report,
            "matched_cdhit_input_sequences": len(matched_internal_rows_with_ids),
            "notes": (
                "Negatives were downsampled without replacement to the largest total count that still allows "
                "an exact match to the positive length histogram. Within each length bin, sampling weights "
                "prefer AMPlify-backed negatives over UniProt-only negatives."
            ),
        },
        "clustering_status": {
            "cdhit_available": False,
            "cdhit_execution_status": "not_run",
            "reason": (
                "No CD-HIT executable is available in the current Windows environment, and an attempted conda "
                "installation did not find a win-64 package. Cluster-aware splitting logic remains in the script, "
                "and the v4 length-matched CD-HIT input FASTA is ready for use once CD-HIT is available."
            ),
        },
        "artifacts": {
            "internal_benchmark_v2_csv": str((data_root / "internal_benchmark_v2.csv").relative_to(project_root)),
            "internal_benchmark_v3_csv": str((data_root / "internal_benchmark_v3.csv").relative_to(project_root)),
            "internal_benchmark_v4_length_matched_csv": str((data_root / "internal_benchmark_v4_length_matched.csv").relative_to(project_root)),
            "internal_for_cdhit_fasta": str((data_root / "internal_for_cdhit.fasta").relative_to(project_root)),
            "internal_for_cdhit_v3_fasta": str((data_root / "internal_for_cdhit_v3.fasta").relative_to(project_root)),
            "internal_for_cdhit_v4_length_matched_fasta": str((data_root / "internal_for_cdhit_v4_length_matched.fasta").relative_to(project_root)),
            "conflict_report": str((report_dir / "conflicting_sequences_v2.csv").relative_to(project_root)),
            "conflict_report_v3": str((report_dir / "conflicting_sequences_v3.csv").relative_to(project_root)),
            "length_distribution_report_v4": str((report_dir / "length_distribution_matching_v4.csv").relative_to(project_root)),
            "cluster_split_logic": "cluster_aware_train_val_test_split(df, cdhit_clstr_file_path)",
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "benchmark_prep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (report_dir / "benchmark_prep_summary_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (report_dir / "benchmark_prep_summary_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (report_dir / "benchmark_prep_summary_v4_length_matched.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    policy_v5_summary = prepare_policy_v5_db_dramp_uniprot(
        project_root=project_root,
        data_root=data_root,
        dbaasp_records=dbaasp_records,
        dramp_records=dramp_records,
        uniprot_records=uniprot_records,
    )
    policy_v6_summary = prepare_policy_v6_db_dramp_all_negative(
        project_root=project_root,
        data_root=data_root,
        dbaasp_records=dbaasp_records,
        dramp_records=dramp_records,
        uniprot_records=uniprot_records,
        uniprot_stats=uniprot_stats,
        amplify_negative_records=amplify_negative_all_records,
        amplify_negative_stats=amplify_negative_all_stats,
        newnegative_records=newnegative_records,
        newnegative_stats=newnegative_stats,
        newnegative_files=newnegative_files,
        uniprot_filename=uniprot_filename,
    )
    final_open_release_summary = prepare_final_open_release(
        project_root=project_root,
        data_root=data_root,
        dbaasp_records=dbaasp_records,
        dramp_records=dramp_records,
        uniprot_records=uniprot_records,
        uniprot_stats=uniprot_stats,
        amplify_negative_records=amplify_negative_all_records,
        amplify_negative_stats=amplify_negative_all_stats,
        newnegative_records=newnegative_records,
        newnegative_stats=newnegative_stats,
        newnegative_files=newnegative_files,
        uniprot_filename=uniprot_filename,
    )

    print(
        json.dumps(
            {
                "legacy_and_intermediate_summary": summary,
                "policy_v5_db_dramp_uniprot": policy_v5_summary,
                "policy_v6_db_dramp_all_negative": policy_v6_summary,
                "policy_final_open_release": final_open_release_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare_amp_benchmarks()
