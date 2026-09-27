#!/usr/bin/env python3
"""Compute MDTraj DSSP assignments and alpha-helix fraction for a peptide trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, help="Peptide-only PDB topology")
    parser.add_argument("--trajectory", required=True, help="Peptide-only XTC trajectory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--out-prefix", default="", help="Optional filename prefix")
    parser.add_argument(
        "--eight-state",
        action="store_true",
        help="Use MDTraj's original DSSP state labels instead of simplified H/E/C",
    )
    parser.add_argument(
        "--time-scale-to-ns",
        type=float,
        default=0.001,
        help="Scale MDTraj time values to ns. MDTraj XTC times are usually ps, so default is 0.001.",
    )
    return parser.parse_args()


def residue_label(index: int, residue) -> str:
    name = getattr(residue, "name", "")
    code = AA3_TO_1.get(str(name).upper())
    if not code:
        code = getattr(residue, "code", None) or "X"
        code = str(code).strip() or "X"
    return f"{index}{code[0]}"


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    try:
        import mdtraj as md
        import numpy as np
    except ImportError as exc:
        raise SystemExit("MDTraj and NumPy are required. Install requirements-optional.txt.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.out_prefix}_" if args.out_prefix else ""
    helix_csv = out_dir / f"{prefix}helix_fraction.csv"
    dssp_csv = out_dir / f"{prefix}dssp_by_residue.csv"
    method_json = out_dir / f"{prefix}secondary_structure_method.json"
    method_md = out_dir / f"{prefix}secondary_structure_method.md"

    traj = md.load_xtc(args.trajectory, top=args.topology)
    dssp = md.compute_dssp(traj, simplified=not args.eight_state)
    residue_labels = [residue_label(i, residue) for i, residue in enumerate(traj.topology.residues, start=1)]

    if dssp.shape[1] != len(residue_labels):
        raise RuntimeError(f"DSSP residue count mismatch: {dssp.shape[1]} vs {len(residue_labels)}")

    times_ns = [float(t) * args.time_scale_to_ns for t in traj.time]
    helix_rows = []
    map_rows = []
    for frame_i, time_ns in enumerate(times_ns):
        states = dssp[frame_i]
        helix_rows.append(
            {
                "time_ns": time_ns,
                "alpha_helix_fraction": float(np.mean(states == "H")),
            }
        )
        row = {"time_ns": time_ns}
        for label, state in zip(residue_labels, states):
            row[label] = str(state)
        map_rows.append(row)

    write_csv(helix_csv, helix_rows)
    write_csv(dssp_csv, map_rows)

    late_values = [r["alpha_helix_fraction"] for r in helix_rows if 80.0 <= r["time_ns"] <= 100.0]
    method = {
        "method": "MDTraj compute_dssp",
        "mdtraj_version": md.__version__,
        "python": sys.version.split()[0],
        "topology": str(Path(args.topology)),
        "trajectory": str(Path(args.trajectory)),
        "simplified": not args.eight_state,
        "state_codes": "H/E/C" if not args.eight_state else "DSSP eight-state labels",
        "helix_fraction_definition": "fraction of peptide residues assigned state H at each frame",
        "n_frames": int(traj.n_frames),
        "n_residues": int(traj.topology.n_residues),
        "first_time_ns": times_ns[0] if times_ns else None,
        "last_time_ns": times_ns[-1] if times_ns else None,
        "initial_helix_fraction": helix_rows[0]["alpha_helix_fraction"] if helix_rows else None,
        "final_helix_fraction": helix_rows[-1]["alpha_helix_fraction"] if helix_rows else None,
        "mean_80_100ns_helix_fraction": float(np.mean(late_values)) if late_values else None,
        "outputs": {
            "helix_fraction_csv": str(helix_csv),
            "dssp_by_residue_csv": str(dssp_csv),
        },
    }
    method_json.write_text(json.dumps(method, indent=2), encoding="utf-8")
    method_md.write_text(
        "\n".join(
            [
                "# Secondary-Structure Method",
                "",
                "Secondary structure was assigned from a peptide-only trajectory using "
                f"MDTraj `{md.__version__}` `compute_dssp(simplified={not args.eight_state})`.",
                "",
                "Alpha-helix fraction is the fraction of peptide residues assigned state `H` at each frame.",
                "",
                f"- Frames: `{method['n_frames']}`",
                f"- Residues: `{method['n_residues']}`",
                f"- Time range: `{method['first_time_ns']}` to `{method['last_time_ns']}` ns",
                f"- Initial helix fraction: `{method['initial_helix_fraction']}`",
                f"- Mean helix fraction, 80-100 ns: `{method['mean_80_100ns_helix_fraction']}`",
                f"- Final helix fraction: `{method['final_helix_fraction']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(method, indent=2), flush=True)


if __name__ == "__main__":
    main()
