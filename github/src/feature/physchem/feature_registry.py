from __future__ import annotations

from dataclasses import dataclass

from config import DEFAULT_FEATURE_GROUPS


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    backend: str
    extractor_key: str | None
    family: str
    description: str
    status: str = "ready"
    enabled: bool = True


FEATURE_SPECS = (
    FeatureSpec("PHYS8", "manual", "PHYS8", "physchem", "Eight handcrafted physicochemical descriptors."),
    FeatureSpec("AAC", "ifeature", "AAC", "composition", "Amino acid composition."),
    FeatureSpec("DPC", "ifeature", "DPC type 1", "composition", "Normalized dipeptide composition."),
    FeatureSpec("DDE", "ifeature", "DDE", "composition", "Dipeptide deviation from expected mean."),
    FeatureSpec("TPC", "ifeature", "TPC type 1", "composition", "Normalized tripeptide composition.", status="excluded_runtime", enabled=False),
    FeatureSpec("GAAC", "ifeature", "GAAC", "composition", "Grouped amino acid composition."),
    FeatureSpec("EGAAC", "ifeature", "EGAAC", "composition", "Enhanced grouped amino acid composition.", status="excluded_variable_length", enabled=False),
    FeatureSpec("CKSAAP", "ifeature", "CKSAAP type 1", "k_spaced", "Composition of k-spaced amino acid pairs."),
    FeatureSpec("CKSAAGP", "ifeature", "CKSAAGP type 1", "k_spaced", "Composition of k-spaced grouped amino acid pairs."),
    FeatureSpec("GDPC", "ifeature", "GDPC type 1", "composition", "Grouped dipeptide composition."),
    FeatureSpec("GTPC", "ifeature", "GTPC type 1", "composition", "Grouped tripeptide composition."),
    FeatureSpec("ASDC", "ifeature", "ASDC", "composition", "Adaptive skip dipeptide composition."),
    FeatureSpec("DistancePair", "ifeature", "DistancePair", "composition", "Distance-pair composition."),
    FeatureSpec("Moran", "ifeature", "Moran", "autocorrelation", "Moran autocorrelation."),
    FeatureSpec("Geary", "ifeature", "Geary", "autocorrelation", "Geary autocorrelation."),
    FeatureSpec("NMBroto", "ifeature", "NMBroto", "autocorrelation", "Normalized Moreau-Broto autocorrelation."),
    FeatureSpec("SOCNumber", "ifeature", "SOCNumber", "sequence_order", "Sequence order coupling number."),
    FeatureSpec("QSOrder", "ifeature", "QSOrder", "sequence_order", "Quasi-sequence order descriptor."),
    FeatureSpec("AC", "ifeature", "AC", "autocorrelation", "Auto covariance descriptor."),
    FeatureSpec("CC", "ifeature", "CC", "autocorrelation", "Cross covariance descriptor."),
    FeatureSpec("ACC", "ifeature", "ACC", "autocorrelation", "Auto-cross covariance descriptor."),
    FeatureSpec("CTDC", "ifeature", "CTDC", "ctd", "CTD composition descriptor."),
    FeatureSpec("CTDT", "ifeature", "CTDT", "ctd", "CTD transition descriptor."),
    FeatureSpec("CTDD", "ifeature", "CTDD", "ctd", "CTD distribution descriptor."),
    FeatureSpec("CTriad", "ifeature", "CTriad", "ctd", "Conjoint triad descriptor."),
    FeatureSpec("KSCTriad", "ifeature", "KSCTriad", "k_spaced", "k-spaced conjoint triad."),
    FeatureSpec("PAAC", "ifeature", "PAAC", "pseudo_aac", "Pseudo amino acid composition."),
    FeatureSpec("APAAC", "ifeature", "APAAC", "pseudo_aac", "Amphiphilic pseudo amino acid composition."),
    FeatureSpec("PseKRAAC_type1", "ifeature", "PseKRAAC type 1", "reduced_alphabet", "Pseudo KRAAC type 1."),
    FeatureSpec("PseKRAAC_type2", "ifeature", "PseKRAAC type 2", "reduced_alphabet", "Pseudo KRAAC type 2."),
    FeatureSpec("PseKRAAC_type3A", "ifeature", "PseKRAAC type 3A", "reduced_alphabet", "Pseudo KRAAC type 3A."),
    FeatureSpec("PseKRAAC_type3B", "ifeature", "PseKRAAC type 3B", "reduced_alphabet", "Pseudo KRAAC type 3B."),
    FeatureSpec("PseKRAAC_type4", "ifeature", "PseKRAAC type 4", "reduced_alphabet", "Pseudo KRAAC type 4."),
    FeatureSpec("PseKRAAC_type5", "ifeature", "PseKRAAC type 5", "reduced_alphabet", "Pseudo KRAAC type 5."),
    FeatureSpec("PseKRAAC_type6A", "ifeature", "PseKRAAC type 6A", "reduced_alphabet", "Pseudo KRAAC type 6A."),
    FeatureSpec("PseKRAAC_type6B", "ifeature", "PseKRAAC type 6B", "reduced_alphabet", "Pseudo KRAAC type 6B."),
    FeatureSpec("PseKRAAC_type6C", "ifeature", "PseKRAAC type 6C", "reduced_alphabet", "Pseudo KRAAC type 6C."),
    FeatureSpec("PseKRAAC_type7", "ifeature", "PseKRAAC type 7", "reduced_alphabet", "Pseudo KRAAC type 7."),
    FeatureSpec("PseKRAAC_type8", "ifeature", "PseKRAAC type 8", "reduced_alphabet", "Pseudo KRAAC type 8."),
    FeatureSpec("PseKRAAC_type9", "ifeature", "PseKRAAC type 9", "reduced_alphabet", "Pseudo KRAAC type 9."),
    FeatureSpec("PseKRAAC_type10", "ifeature", "PseKRAAC type 10", "reduced_alphabet", "Pseudo KRAAC type 10."),
    FeatureSpec("PseKRAAC_type11", "ifeature", "PseKRAAC type 11", "reduced_alphabet", "Pseudo KRAAC type 11."),
    FeatureSpec("PseKRAAC_type12", "ifeature", "PseKRAAC type 12", "reduced_alphabet", "Pseudo KRAAC type 12."),
    FeatureSpec("PseKRAAC_type13", "ifeature", "PseKRAAC type 13", "reduced_alphabet", "Pseudo KRAAC type 13."),
    FeatureSpec("PseKRAAC_type14", "ifeature", "PseKRAAC type 14", "reduced_alphabet", "Pseudo KRAAC type 14."),
    FeatureSpec("PseKRAAC_type15", "ifeature", "PseKRAAC type 15", "reduced_alphabet", "Pseudo KRAAC type 15."),
    FeatureSpec("PseKRAAC_type16", "ifeature", "PseKRAAC type 16", "reduced_alphabet", "Pseudo KRAAC type 16."),
)

FEATURE_SPEC_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}


def resolve_feature_groups(groups: list[str] | None) -> list[str]:
    if not groups:
        return list(DEFAULT_FEATURE_GROUPS)
    resolved = []
    for group in groups:
        if group not in FEATURE_SPEC_BY_NAME:
            raise ValueError(f"Unknown feature group: {group}")
        resolved.append(group)
    return resolved


def ready_feature_groups() -> list[str]:
    return [spec.name for spec in FEATURE_SPECS if spec.enabled and spec.status == "ready"]
