from __future__ import annotations

import sys
from pathlib import Path


PIPELINE = Path(__file__).resolve().parent / "run_ccd_no_phys8_pipeline.py"


if __name__ == "__main__":
    namespace = {"__name__": "__main__", "__file__": str(PIPELINE)}
    code = compile(PIPELINE.read_text(encoding="utf-8"), str(PIPELINE), "exec")
    sys.argv[0] = str(PIPELINE)
    exec(code, namespace)
