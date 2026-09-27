# Reproducibility Guide

This repository is a cleaned code package. Large benchmark tables, model
checkpoints, third-party repositories, and MD trajectories are intentionally not
committed. To reproduce analyses, place those artifacts in the documented
locations and run the scripts with project-relative paths.

## 1. Python Environment

Core SAF-AMPTox workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional generation, external-tool, and MD-analysis helpers:

```bash
pip install -r requirements-optional.txt
```

## 2. Benchmark And Model Inputs

Expected benchmark split layout:

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

Each split CSV should contain:

```text
fasta_id,sequence,label_id
```

Optional trained fusion checkpoints can be placed under:

```text
checkpoints/fusion/
```

## 3. Generated Candidate Screening

Candidate scoring requires:

- candidate sequence CSV
- candidate CCD feature parquet
- candidate ESM-2 feature parquet
- trained fusion checkpoints

Use the commands in `README.md` and `docs/PIPELINE.md`.

## 4. MD Secondary-Structure Reproduction

For the P1_00985 Fig.6E/MD Panel E source data, place trajectory inputs under:

```text
data/md/p1/
  production_100ns.tpr
  production_100ns_centered.xtc
```

Then run:

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

The key output for Panel E is:

```text
results/md/p1_dssp/helix_fraction.csv
```

The residue-by-time secondary-structure map is:

```text
results/md/p1_dssp/dssp_by_residue.csv
```

## 5. Path Hygiene

Do not commit:

- local absolute paths
- server credentials
- private SSH hostnames/passwords
- generated trajectory files
- full benchmark matrices
- large checkpoints

Use placeholders such as `<PROJECT_ROOT>` in documentation.

