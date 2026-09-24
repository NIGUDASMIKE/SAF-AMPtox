# Physicochemical Feature Pipeline

This directory contains the compact CCD/physicochemical feature workflow used by
SAF-AMPTox for AMP and TOX benchmarks.

## Scope

- load AMP and TOX train/validation/test splits
- export descriptor matrices
- evaluate single descriptor groups
- select a shared feature union with LightGBM
- build compact shared-union tables for downstream fusion models

## Main Feature Groups

- `PHYS8`: eight handcrafted physicochemical descriptors
- `AAC`, `DPC`, `DDE`, `GAAC`, `CKSAAP`, `CTDD`, `KSCTriad`, `APAAC`
- additional iFeature-style groups registered in `feature_registry.py`

## Typical Workflow

Export feature matrices:

```bash
python src/feature/physchem/extract_features.py --tasks amp tox --all-ready --overwrite
```

Run single-group evaluation:

```bash
python src/feature/physchem/evaluate_single_features.py --tasks amp tox --groups AAC DPC CTDD
```

Select task-aware union groups:

```bash
python src/feature/physchem/select_union_groups_by_threshold.py --suggest-only
python src/feature/physchem/select_union_groups_by_threshold.py --amp-roc-threshold 0.80 --tox-roc-threshold 0.75
```

Build union matrices:

```bash
python src/feature/physchem/build_union_matrix.py --selection-json data--final/feature_physchem/reports/selected_groups_union.json
```

Run the no-PHYS8 compact pipeline:

```bash
python src/feature/run_ccd_no_phys8_pipeline.py --include-hard
```

Outputs are written under `data--final/feature_physchem/` by default, or under
the directory specified by the `CCD_FEATURE_OUTPUT_ROOT` environment variable.
