from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from config import FEATURE_OUTPUT_ROOT, REPORT_DIR


PALETTE = {
    "blue_main": "#2F5D8C",
    "blue_secondary": "#A0B1BA",
    "green_2": "#AADCA9",
    "red_strong": "#C00000",
    "amp_color": "#E64B35",
    "tox_color": "#4DBBD5",
    "neutral_light": "#D8DEE3",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "gold": "#E7B800",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "bg_lilac": "#EEEAF6",
    "bg_aqua": "#EDF4F6",
    "bg_peach": "#F7EEE7",
}


FIGURE_DIR = FEATURE_OUTPUT_ROOT / "figures"
SOURCE_DATA_DIR = FEATURE_OUTPUT_ROOT / "source_data"


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def apply_publication_style(font_size: int = 8, axes_linewidth: float = 0.9) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "xtick.minor.width": axes_linewidth * 0.8,
        "ytick.minor.width": axes_linewidth * 0.8,
        "legend.frameon": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": PALETTE["neutral_black"],
        "text.color": PALETTE["neutral_black"],
        "axes.edgecolor": PALETTE["neutral_black"],
        "xtick.color": PALETTE["neutral_black"],
        "ytick.color": PALETTE["neutral_black"],
    })


def save_figure(fig: plt.Figure, basename: str) -> dict[str, str]:
    ensure_dirs()
    svg_path = FIGURE_DIR / f"{basename}.svg"
    pdf_path = FIGURE_DIR / f"{basename}.pdf"
    png_path = FIGURE_DIR / f"{basename}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=600)
    return {
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_source_table(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs()
    out_path = SOURCE_DATA_DIR / name
    df.to_csv(out_path, index=False)
    return out_path


def abbreviate_feature_name(name: str, max_len: int = 34) -> str:
    replacements = {
        "hydrophobicity_": "hydro.",
        "secondarystruct": "sec",
        "solventaccess": "solv",
        "normwaalsvolume": "vdw",
        "polarizability": "polariz.",
        "Hydrophobicity": "Hydro",
        "Hydrophilicity": "Hydrophil",
        "residue": "r",
        "CTDD_": "CTDD:",
        "CKSAAP_": "CK:",
        "KSCTriad_": "KSC:",
        "APAAC_Pc1.": "APAAC1:",
        "APAAC_Pc2.": "APAAC2:",
        "PAAC_Xc1.": "PAAC1:",
        "PAAC_Xc2.": "PAAC2:",
        "ASDC_": "ASDC:",
        "DDE_": "DDE:",
        "AAC_": "AAC:",
    }
    short = name
    for source, target in replacements.items():
        short = short.replace(source, target)
    if len(short) <= max_len:
        return short
    return short[: max_len - 3] + "..."


AMINO_ACID_LABELS = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}


PROPERTY_LABELS = {
    "solventaccess": "Solvent accessibility",
    "normwaalsvolume": "van der Waals volume",
    "charge": "Charge",
    "hydrophobicity_ENGD860101": "Hydrophobicity (ENGD)",
    "hydrophobicity_ZIMJ680101": "Hydrophobicity (ZIMJ)",
    "hydrophobicity_ARGP820101": "Hydrophobicity (ARGP)",
    "hydrophobicity_PRAM900101": "Hydrophobicity (PRAM)",
}


FEATURE_LABEL_OVERRIDES = {
    "APAAC_Pc1.M": "APAAC: Amphiphilicity (Met)",
    "APAAC_Pc1.K": "APAAC: Amphiphilicity (Lys)",
    "APAAC_Pc1.R": "APAAC: Amphiphilicity (Arg)",
    "APAAC_Pc1.C": "APAAC: Amphiphilicity (Cys)",
    "APAAC_Pc2.Hydrophobicity.2": "APAAC: Hydrophobicity lag 2",
    "APAAC_Pc2.Hydrophilicity.2": "APAAC: Hydrophilicity lag 2",
    "DP_C": "DistancePair: Cys spacing",
}


def _aa_label(code: str) -> str:
    return AMINO_ACID_LABELS.get(code, code)


def pretty_feature_label(name: str, max_len: int | None = None) -> str:
    label = FEATURE_LABEL_OVERRIDES.get(name)
    if label is None:
        if name.startswith("AAC_"):
            residue = name.split("_", 1)[1]
            label = f"AAC: {_aa_label(residue)} frequency"
        elif name.startswith("ASDC_"):
            pair = name.split("_", 1)[1]
            label = f"ASDC: {_aa_label(pair[0])}-{_aa_label(pair[1])} pair"
        elif name.startswith("DDE_"):
            pair = name.split("_", 1)[1]
            label = f"DDE: {_aa_label(pair[0])}-{_aa_label(pair[1])} deviation"
        else:
            match = re.match(r"CKSAAP_([A-Z]{2})\.gap(\d+)$", name)
            if match:
                pair, gap = match.groups()
                label = f"CKSAAP: {_aa_label(pair[0])}-{_aa_label(pair[1])} (gap {gap})"
            else:
                match = re.match(r"CTDD_([^\.]+)\.\d+\.residue(\d+)$", name)
                if match:
                    prop_key, percentile = match.groups()
                    prop_label = PROPERTY_LABELS.get(prop_key, prop_key.replace("_", " "))
                    label = f"CTDD: {prop_label} ({percentile}%)"
                else:
                    label = abbreviate_feature_name(name, max_len=max_len or 40)
    if max_len is not None and len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def report_path(name: str) -> Path:
    return REPORT_DIR / name
