# MD Secondary-Structure Provenance

This note documents the recovered provenance for the P1_00985 MD
secondary-structure panel used in the SAF-AMPTox CBC journal extension.

No server credentials, local user paths, or machine-specific absolute paths are
required to reproduce the method. Paths below are project-relative examples.

## Recovered Method

The original analysis was recovered from the compute-server project after a
local handoff audit initially marked the method unresolved.

The recovered workflow used:

- Python `3.10.12`
- MDTraj `1.10.3`
- MDAnalysis `2.9.0`
- GROMACS `2023.2` for trajectory preprocessing
- MDTraj `compute_dssp(simplified=True)` for three-state H/E/C assignment

The analysis did not use `mkdssp`, `dssp`, or `gmx do_dssp`; those binaries
were not found in the inspected server environment.

## Workflow

The recovered script performed these steps:

1. Load the membrane-centered trajectory with MDAnalysis.
2. Select peptide atoms with the MDAnalysis selection `protein`.
3. Write a peptide-only PDB topology.
4. Write a peptide-only XTC trajectory.
5. Load the peptide-only XTC/PDB with MDTraj.
6. Run `md.compute_dssp(traj, simplified=True)`.
7. Save per-frame alpha-helix fraction as `mean(DSSP_state == "H")`.
8. Save residue-by-time H/E/C assignments.

## Reproducing The CSVs

Place trajectory inputs under a local data directory, for example:

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

Expected outputs:

```text
results/md/p1_dssp/
  peptide_only_topology.pdb
  peptide_only_centered.xtc
  peptide_only_extraction_manifest.json
  helix_fraction.csv
  dssp_by_residue.csv
  secondary_structure_method.json
  secondary_structure_method.md
```

## Expected P1_00985 Values

For the recovered 100 ns P1_00985 trajectory, the original outputs contained:

- `2001` frames from `0.0` to `100.0` ns
- `17` peptide residues
- initial alpha-helix fraction: `0.8235294117647058`
- mean alpha-helix fraction over `80-100 ns`: `0.7290597036819715`
- final alpha-helix fraction: `0.7647058823529411`

These values are useful as a sanity check when reproducing Fig.6E from the same
trajectory.

## Manuscript Wording

Recommended wording:

> Secondary structure was assigned from a peptide-only trajectory using MDTraj
> v1.10.3 `compute_dssp(simplified=True)`, yielding three-state H/E/C
> assignments. The alpha-helix fraction was calculated as the fraction of
> peptide residues assigned H at each frame.

Avoid:

- `mkdssp`
- `gmx do_dssp`
- external DSSP binary
- `method unresolved`

unless a separate external-DSSP run is performed and documented.

## Provenance Caveat

An older first-pass analysis had skipped secondary-structure assignment because
MDTraj/DSSP support was not yet available in that run. It was superseded by a
later paper-ready analysis script that generated the final `helix_fraction.csv`
and `dssp_by_residue.csv` with MDTraj.

