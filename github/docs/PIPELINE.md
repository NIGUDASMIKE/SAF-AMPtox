# SAF-AMPTox Pipeline

## Stage 1: Benchmark Construction

The AMP and TOX benchmark scripts clean peptide sequences, remove non-standard
residues, enforce a 10-50 aa length range, deduplicate sequences, remove label
conflicts, construct length-matched negatives, and create cluster-aware splits.

Main scripts:

- `src/utils/prepare_amp_benchmarks.py`
- `src/utils/prepare_tox_benchmark.py`
- `src/utils/build_hard_test.py`
- `src/utils/profile_shortcut_bias.py`

## Stage 2: Feature Extraction

The feature branch exports compact physicochemical descriptor tables and ESM-2
mean-pooled embeddings. The selected 832-feature CCD union is stored in
`resources/feature_union_832.json`.

Main scripts:

- `src/feature/physchem/extract_features.py`
- `src/feature/physchem/select_union_features_lgbm.py`
- `src/feature/extract_esm2.py`

## Stage 3: Fusion Discriminator

`src/feature/train_fusion_models.py` implements CCD-only, ESM-only, direct
concat, gated, cross-attention, and residual cross-attention classifiers.

The main SAF-AMPTox discriminator is:

```text
cross_attention_residual
```

## Stage 4: Generation And Screening

The generative branch samples candidate peptides, extracts candidate CCD and
ESM-2 features, scores candidates with AMP and TOX fusion ensembles, and applies
QC plus dual-task ranking.

Main scripts:

- `src/generative/sample_sft_evodiff_candidates.py`
- `src/generative/extract_candidate_ccd_features.py`
- `src/generative/extract_candidate_esm2_features.py`
- `src/generative/score_candidates_with_fusion_predictor.py`
- `src/generative/screen_sft_evodiff_pool.py`
- `src/generative/summarize_main_residual_generation.py`

## Stage 5: External Baselines And Figures

External SOTA comparisons and final evidence plots are separated from the core
training code:

- `src/benchmark/run_sota_duel.py`
- `src/feature/build_fusion_evidence_package.py`
- `src/figures/`
