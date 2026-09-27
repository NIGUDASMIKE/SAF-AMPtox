# MD Analysis Helpers

This folder contains small, data-agnostic scripts for reproducing the P1_00985
secondary-structure panel from trajectory files.

The recovered Fig.6E workflow is:

1. Use MDAnalysis to select the peptide/protein atoms from a centered trajectory.
2. Write a peptide-only PDB topology and peptide-only XTC trajectory.
3. Use MDTraj `compute_dssp(simplified=True)` to assign H/E/C states.
4. Compute alpha-helix fraction as the fraction of residues assigned `H`.

Example:

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

- `helix_fraction.csv`
- `dssp_by_residue.csv`
- `secondary_structure_method.json`
- `secondary_structure_method.md`

Do not describe this workflow as `gmx do_dssp`, `mkdssp`, or an external DSSP
binary run unless you perform and document that separate analysis.

