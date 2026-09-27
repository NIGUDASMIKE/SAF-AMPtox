# SAF-AMPTox

SAF-AMPTox is a leakage-aware AMP/toxicity peptide modelling workflow. The
main line combines curated AMP and TOX benchmarks, shortcut-controlled hard
tests, compact physicochemical descriptors, ESM-2 embeddings, a residual
cross-attention fusion classifier, and an EvoDiff-based candidate screening
loop.

This folder is a cleaned open-release code package. It intentionally excludes
local IDE files, caches, temporary artifacts, third-party source trees, and
large model weights.

## Repository Layout

```text
src/utils/                 Benchmark construction, shortcut audit, hard-test builders
src/feature/physchem/      CCD/physicochemical descriptor extraction and selection
src/feature/               ESM-2 extraction, fusion models, ablations, UMAP/evidence scripts
src/generative/            EvoDiff sampling, candidate feature extraction, scoring, screening
src/benchmark/             External SOTA paired-duel evaluation scripts
src/figures/               Figure-generation scripts for manuscript panels
src/md/                    MD trajectory helper scripts for reproducible DSSP/helix analysis
resources/                 Small reusable resources, including the 832-feature union list
data/                      Place final benchmark splits and feature tables here
checkpoints/fusion/        Optional location for trained Residual CA checkpoints
external/                  Optional location for external tools/models
```

## What Is Included

- Core Python scripts for the SAF-AMPTox main line.
- `resources/feature_union_832.json`, the compact shared CCD feature set used by
  the main Residual Cross-Attention discriminator.
- Reproducible MD secondary-structure helper scripts and provenance notes for
  the P1_00985 case-study trajectory.
- Placeholder folders for data, checkpoints, and external tools.
- Reproducibility-oriented command examples.

## What Is Not Included

- Full benchmark tables and feature matrices.
- Manuscript-only handoff folders, final figure PDFs, and raw MD trajectories.
- Trained EvoDiff/SFT/DPO weights.
- Third-party repositories such as EvoDiff, CD-HIT, AMPlify, ToxinPred3, or
  Macrel.
- Machine-specific manifests containing absolute local paths.

## Environment

Create an environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional components:

- ESM-2 extraction requires `fair-esm` and PyTorch.
- Full iFeature-style descriptor extraction requires `iFeatureOmegaCLI`.
- Generative sampling requires a local EvoDiff checkout available as
  `external/evodiff-main` or on `PYTHONPATH`.
- MD secondary-structure reproduction requires `MDAnalysis` and `mdtraj`.
- SOTA comparison scripts require the corresponding external tools.

Optional Python dependencies can be installed with:

```bash
pip install -r requirements-optional.txt
```

## Expected Data Layout

Place prepared benchmark files under `data--final/`:

```text
data--final/
  splits/train.csv
  splits/val.csv
  splits/test.csv
  splits/test_hard_amp.csv
  tox/splits/train.csv
  tox/splits/val.csv
  tox/splits/test.csv
  tox/splits/test_hard_tox.csv
```

Each split CSV should contain at least:

```text
fasta_id,sequence,label_id
```

## Main Workflow

Build or audit final benchmark splits:

```bash
python src/utils/prepare_amp_benchmarks.py
python src/utils/prepare_tox_benchmark.py
python src/utils/build_hard_test.py
python src/utils/profile_shortcut_bias.py --amp-splits train test test_hard_amp --tox-splits train test test_hard_tox
```

Extract CCD features and select the compact union:

```bash
python src/feature/physchem/extract_features.py --tasks amp tox --all-ready --overwrite
python src/feature/physchem/evaluate_single_features.py --tasks amp tox --all-ready
python src/feature/physchem/select_union_groups_by_threshold.py --suggest-only
python src/feature/physchem/build_union_matrix.py --selection-json data--final/feature_physchem/reports/selected_groups_union.json
python src/feature/physchem/select_union_features_lgbm.py --union-subdir union_final_groups --compact-roc-tolerance 0.0005 --compact-pr-tolerance 0.0005
```

Extract ESM-2 features:

```bash
python src/feature/extract_esm2.py --include-hard --device auto
```

Train the fusion discriminator:

```bash
python src/feature/train_fusion_models.py \
  --tasks amp tox \
  --models concat_mlp cross_attention_residual \
  --ccd-root data--final/feature_physchem \
  --ccd-union-subdir golden_shared_union \
  --ccd-feature-json resources/feature_union_832.json \
  --esm-root data--final/feature_esm2 \
  --output-root data--final/fusion_models_main \
  --seeds 13 29 47
```

Score generated candidates with trained checkpoints:

```bash
python src/generative/extract_candidate_ccd_features.py \
  --candidate-csv data--final/generative_main/sft_final/candidates.csv \
  --feature-json resources/feature_union_832.json \
  --out-parquet data--final/generative_main/sft_final/features/candidate_ccd_features.parquet

python src/generative/extract_candidate_esm2_features.py \
  --candidate-csv data--final/generative_main/sft_final/candidates.csv \
  --out-parquet data--final/generative_main/sft_final/features/candidate_esm2_features.parquet

python src/generative/score_candidates_with_fusion_predictor.py \
  --checkpoint-root checkpoints/fusion \
  --ccd-parquet data--final/generative_main/sft_final/features/candidate_ccd_features.parquet \
  --esm-parquet data--final/generative_main/sft_final/features/candidate_esm2_features.parquet \
  --out-csv data--final/generative_main/sft_final/screening/candidate_fusion_scores.csv
```

Reproduce the P1_00985 MD secondary-structure source data:

```bash
python src/md/extract_peptide_only.py \
  --topology data/md/p1/production_100ns.tpr \
  --trajectory data/md/p1/production_100ns_centered.xtc \
  --selection protein \
  --out-dir results/md/p1_dssp

python src/md/compute_mdtraj_dssp.py \
  --topology results/md/p1_dssp/peptide_only_topology.pdb \
  --trajectory results/md/p1_dssp/peptide_only_centered.xtc \
  --out-dir results/md/p1_dssp
```

See `docs/REPRODUCIBILITY.md` and
`docs/MD_SECONDARY_STRUCTURE_PROVENANCE.md` for release-safe details.

## Notes For Release

- Choose a license before publishing publicly.
- If model checkpoints are released, put small fusion checkpoints under
  `checkpoints/fusion/` or attach them as GitHub Releases. Large EvoDiff weights
  should use Git LFS or an external model hub.
- Keep generated `*.manifest.json` files out of commits unless they have been
  checked for local absolute paths.
