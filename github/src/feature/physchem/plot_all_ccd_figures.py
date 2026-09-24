from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    for script_name in (
        "plot_feature_landscape.py",
        "plot_topk_elbow.py",
        "plot_importance_overlap.py",
    ):
        run(script_name)
        print(f"[OK] {script_name}")


if __name__ == "__main__":
    main()
