#!/usr/bin/env python3
"""Extract a peptide-only topology and trajectory from an MDAnalysis-readable system.

This mirrors the P1_00985 provenance workflow used before MDTraj DSSP
assignment: load the centered MD trajectory, select the peptide/protein atoms,
write a peptide-only PDB topology, and write a peptide-only XTC trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, help="Input topology, e.g. production_100ns.tpr or .pdb/.gro")
    parser.add_argument("--trajectory", required=True, help="Input trajectory, e.g. centered production .xtc")
    parser.add_argument("--selection", default="protein", help="MDAnalysis atom selection for peptide atoms")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--out-prefix", default="peptide_only", help="Output filename prefix")
    parser.add_argument("--stride", type=int, default=1, help="Write every Nth frame")
    parser.add_argument("--start", type=int, default=None, help="Optional first frame index")
    parser.add_argument("--stop", type=int, default=None, help="Optional stop frame index, Python-slice style")
    return parser.parse_args()


def residue_labels(atom_group) -> list[str]:
    labels = []
    for i, residue in enumerate(atom_group.residues, start=1):
        labels.append(f"{i}:{residue.resname}{residue.resid}")
    return labels


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise SystemExit("MDAnalysis is required. Install requirements-optional.txt.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"{args.out_prefix}_topology.pdb"
    out_xtc = out_dir / f"{args.out_prefix}_centered.xtc"
    manifest_path = out_dir / f"{args.out_prefix}_extraction_manifest.json"

    universe = mda.Universe(args.topology, args.trajectory)
    atoms = universe.select_atoms(args.selection)
    if atoms.n_atoms == 0:
        raise SystemExit(f"Selection matched zero atoms: {args.selection!r}")

    universe.trajectory[0]
    atoms.write(str(out_pdb))

    frames_written = 0
    times_ps: list[float] = []
    frame_slice = slice(args.start, args.stop, args.stride)
    with mda.Writer(str(out_xtc), atoms.n_atoms) as writer:
        for ts in universe.trajectory[frame_slice]:
            writer.write(atoms)
            frames_written += 1
            times_ps.append(float(ts.time))

    manifest = {
        "topology": str(Path(args.topology)),
        "trajectory": str(Path(args.trajectory)),
        "selection": args.selection,
        "stride": args.stride,
        "start": args.start,
        "stop": args.stop,
        "out_pdb": str(out_pdb),
        "out_xtc": str(out_xtc),
        "n_atoms": int(atoms.n_atoms),
        "n_residues": int(atoms.residues.n_residues),
        "residues": residue_labels(atoms),
        "frames_written": frames_written,
        "first_time_ps": times_ps[0] if times_ps else None,
        "last_time_ps": times_ps[-1] if times_ps else None,
        "software": {
            "MDAnalysis": getattr(mda, "__version__", "unknown"),
            "python": sys.version.split()[0],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

