from __future__ import annotations

import math
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from config import VALID_AA
from ifeature_backend import extract_ifeature_descriptor


AA_LIST = list(VALID_AA)
DIPEPTIDES = ["".join(pair) for pair in product(AA_LIST, repeat=2)]

KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

BOMAN_SCALE = {
    "A": -0.36, "R": 2.58, "N": 1.34, "D": 3.05, "C": -0.41,
    "Q": 0.73, "E": 3.11, "G": -0.20, "H": 0.81, "I": -1.75,
    "L": -1.76, "K": 2.80, "M": -0.62, "F": -1.33, "P": -0.40,
    "S": 0.22, "T": 0.11, "W": -0.60, "Y": 0.35, "V": -1.27,
}

PKA_SCALE = {
    "N_term": 8.6,
    "K": 10.8,
    "R": 12.5,
    "H": 6.0,
    "C_term": 3.6,
    "D": 3.9,
    "E": 4.1,
    "C": 8.3,
    "Y": 10.9,
}

CTD_PROPERTIES = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary": ("EALMQKRH", "VIYCWFT", "GNPSD"),
}

CTD_PERCENTILES = (1, 25, 50, 75, 100)


def _clean_sequence(seq: str) -> str:
    return "".join([aa for aa in str(seq).upper() if aa in VALID_AA])


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def calculate_precise_charge(seq: str, target_pH: float = 7.4) -> float:
    if not seq:
        return 0.0
    charge = 1.0 / (1.0 + 10 ** (target_pH - PKA_SCALE["N_term"]))
    charge += sum(1.0 / (1.0 + 10 ** (target_pH - PKA_SCALE[aa])) for aa in seq if aa in {"K", "R", "H"})
    charge -= 1.0 / (1.0 + 10 ** (PKA_SCALE["C_term"] - target_pH))
    charge -= sum(1.0 / (1.0 + 10 ** (PKA_SCALE[aa] - target_pH)) for aa in seq if aa in {"D", "E", "C", "Y"})
    return charge


def calculate_pi(seq: str) -> float:
    left, right = 0.0, 14.0
    mid = 7.0
    for _ in range(100):
        mid = (left + right) / 2.0
        if calculate_precise_charge(seq, mid) > 0:
            left = mid
        else:
            right = mid
    return mid


def calculate_hydrophobic_moment(seq: str) -> float:
    if not seq:
        return 0.0
    delta = 100.0 * math.pi / 180.0
    sum_cos = sum(KD_SCALE.get(aa, 0.0) * math.cos(delta * i) for i, aa in enumerate(seq))
    sum_sin = sum(KD_SCALE.get(aa, 0.0) * math.sin(delta * i) for i, aa in enumerate(seq))
    return math.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(seq)


def _phys8_row(seq: str) -> list[float]:
    clean_seq = _clean_sequence(seq)
    length = len(clean_seq)
    if length == 0:
        return [0.0] * 8

    hydrophobicity = sum(KD_SCALE[aa] for aa in clean_seq) / length / 4.5
    boman = sum(BOMAN_SCALE[aa] for aa in clean_seq) / length / 3.0
    aliphatic = (
        clean_seq.count("A")
        + 2.9 * clean_seq.count("V")
        + 3.9 * (clean_seq.count("I") + clean_seq.count("L"))
    ) / length * 100.0 / 150.0
    return [
        length / 100.0,
        calculate_precise_charge(clean_seq, target_pH=7.4) / 10.0,
        hydrophobicity,
        boman,
        aliphatic,
        calculate_pi(clean_seq) / 14.0,
        (clean_seq.count("F") + clean_seq.count("Y") + clean_seq.count("W")) / length,
        calculate_hydrophobic_moment(clean_seq) / 2.0,
    ]


def extract_phys8(sequences: list[str]) -> pd.DataFrame:
    columns = [
        "phys8_length_norm",
        "phys8_net_charge_ph74",
        "phys8_hydrophobicity",
        "phys8_boman",
        "phys8_aliphatic_index",
        "phys8_pi_norm",
        "phys8_aromaticity",
        "phys8_hydrophobic_moment",
    ]
    rows = [_phys8_row(seq) for seq in sequences]
    return pd.DataFrame(rows, columns=columns)


def extract_aac(sequences: list[str]) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        clean_seq = _clean_sequence(seq)
        counts = Counter(clean_seq)
        length = len(clean_seq)
        rows.append([_safe_divide(counts[aa], length) for aa in AA_LIST])
    columns = [f"aac_{aa}" for aa in AA_LIST]
    return pd.DataFrame(rows, columns=columns)


def extract_dpc(sequences: list[str]) -> pd.DataFrame:
    rows = []
    for seq in sequences:
        clean_seq = _clean_sequence(seq)
        pairs = [clean_seq[index:index + 2] for index in range(max(len(clean_seq) - 1, 0))]
        counts = Counter(pair for pair in pairs if len(pair) == 2)
        total = len(pairs)
        rows.append([_safe_divide(counts[pair], total) for pair in DIPEPTIDES])
    columns = [f"dpc_{pair}" for pair in DIPEPTIDES]
    return pd.DataFrame(rows, columns=columns)


def _group_lookup(property_groups: tuple[str, str, str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for group_id, residues in enumerate(property_groups, start=1):
        for residue in residues:
            lookup[residue] = group_id
    return lookup


def extract_ctdc(sequences: list[str]) -> pd.DataFrame:
    rows = []
    columns = []
    for property_name in CTD_PROPERTIES:
        for group_id in (1, 2, 3):
            columns.append(f"ctdc_{property_name}_g{group_id}")

    for seq in sequences:
        clean_seq = _clean_sequence(seq)
        length = len(clean_seq)
        values: list[float] = []
        for property_groups in CTD_PROPERTIES.values():
            lookup = _group_lookup(property_groups)
            memberships = [lookup[aa] for aa in clean_seq]
            for group_id in (1, 2, 3):
                values.append(_safe_divide(sum(1 for member in memberships if member == group_id), length))
        rows.append(values)
    return pd.DataFrame(rows, columns=columns)


def extract_ctdt(sequences: list[str]) -> pd.DataFrame:
    rows = []
    columns = []
    for property_name in CTD_PROPERTIES:
        for transition in ("12", "13", "23"):
            columns.append(f"ctdt_{property_name}_{transition}")

    for seq in sequences:
        clean_seq = _clean_sequence(seq)
        values: list[float] = []
        for property_groups in CTD_PROPERTIES.values():
            lookup = _group_lookup(property_groups)
            memberships = [lookup[aa] for aa in clean_seq]
            transitions = list(zip(memberships, memberships[1:]))
            total = len(transitions)
            counts = {
                "12": sum(1 for left, right in transitions if {left, right} == {1, 2}),
                "13": sum(1 for left, right in transitions if {left, right} == {1, 3}),
                "23": sum(1 for left, right in transitions if {left, right} == {2, 3}),
            }
            for transition in ("12", "13", "23"):
                values.append(_safe_divide(counts[transition], total))
        rows.append(values)
    return pd.DataFrame(rows, columns=columns)


def _distribution_position(indices: list[int], sequence_length: int, percentile: int) -> float:
    if not indices or sequence_length == 0:
        return 0.0
    if percentile == 1:
        target_index = indices[0]
    else:
        rank = math.ceil(percentile / 100.0 * len(indices)) - 1
        target_index = indices[max(rank, 0)]
    return target_index / sequence_length * 100.0


def extract_ctdd(sequences: list[str]) -> pd.DataFrame:
    rows = []
    columns = []
    for property_name in CTD_PROPERTIES:
        for group_id in (1, 2, 3):
            for percentile in CTD_PERCENTILES:
                columns.append(f"ctdd_{property_name}_g{group_id}_p{percentile}")

    for seq in sequences:
        clean_seq = _clean_sequence(seq)
        sequence_length = len(clean_seq)
        values: list[float] = []
        for property_groups in CTD_PROPERTIES.values():
            lookup = _group_lookup(property_groups)
            memberships = [lookup[aa] for aa in clean_seq]
            for group_id in (1, 2, 3):
                indices = [index + 1 for index, member in enumerate(memberships) if member == group_id]
                for percentile in CTD_PERCENTILES:
                    values.append(_distribution_position(indices, sequence_length, percentile))
        rows.append(values)
    return pd.DataFrame(rows, columns=columns)


EXTRACTOR_MAP = {
    "PHYS8": extract_phys8,
}


def extract_manual_feature_group(group: str, sequences: list[str]) -> pd.DataFrame:
    if group not in EXTRACTOR_MAP:
        raise ValueError(f"Manual feature group is not implemented: {group}")
    return EXTRACTOR_MAP[group](sequences).astype("float32")


def extract_feature_group(
    group: str,
    sequences: list[str],
    backend: str,
    extractor_key: str,
    fasta_ids: list[str] | None = None,
) -> pd.DataFrame:
    if backend == "manual":
        return extract_manual_feature_group(group, sequences)
    if backend == "ifeature":
        if fasta_ids is None:
            raise ValueError(f"fasta_ids are required for iFeature extraction: {group}")
        return extract_ifeature_descriptor(list(zip(fasta_ids, sequences)), extractor_key)
    raise ValueError(f"Unsupported backend: {backend}")
