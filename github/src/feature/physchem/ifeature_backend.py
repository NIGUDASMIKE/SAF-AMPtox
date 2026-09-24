from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from iFeatureOmegaCLI.iFeatureOmegaCLI import iProtein


def build_fasta(records: list[tuple[str, str]], fasta_path: Path) -> None:
    with fasta_path.open("w", encoding="utf-8") as handle:
        for fasta_id, sequence in records:
            handle.write(f">{fasta_id}\n{sequence}\n")


def extract_ifeature_descriptor(
    records: list[tuple[str, str]],
    descriptor_name: str,
) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False, encoding="utf-8") as handle:
        tmp_path = Path(handle.name)
    try:
        build_fasta(records, tmp_path)
        protein = iProtein(str(tmp_path))
        protein.get_descriptor(descriptor_name)
        if protein.encodings is None:
            raise ValueError(f"Descriptor returned no encoding table: {descriptor_name}")
        encodings = protein.encodings.copy()
        for column in encodings.columns:
            encodings[column] = pd.to_numeric(encodings[column], errors="coerce")
        if encodings.isnull().any().any():
            encodings = encodings.fillna(0.0)
        return encodings.astype("float32")
    finally:
        tmp_path.unlink(missing_ok=True)
