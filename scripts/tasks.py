"""Cross-platform task runner. The Makefile delegates here so Windows works without make.

Usage:  python scripts/tasks.py <task> [extra args]
Tasks:  demo | data | train | api | dashboard | test | lint | format
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

TASKS: dict[str, list[list[str]]] = {
    "demo": [[PY, "scripts/run_demo.py"]],
    "data": [[PY, "scripts/download_data.py"]],
    "train": [[PY, "scripts/build_dataset.py"], [PY, "scripts/train.py"]],
    "api": [[PY, "-m", "uvicorn", "akashnetra.api.main:app", "--reload", "--port", "8000"]],
    "dashboard": [[PY, "-m", "streamlit", "run", "src/akashnetra/dashboard/app.py"]],
    "test": [[PY, "-m", "pytest"]],
    "lint": [[PY, "-m", "ruff", "check", "."], [PY, "-m", "black", "--check", "."]],
    "format": [[PY, "-m", "ruff", "check", "--fix", "."], [PY, "-m", "black", "."]],
}


def main(argv: list[str]) -> int:
    """Run the requested task and return its exit code."""
    if not argv or argv[0] not in TASKS:
        sys.stderr.write(f"Usage: python scripts/tasks.py <{'|'.join(TASKS)}> [args]\n")
        return 2
    name, extra = argv[0], argv[1:]
    for i, cmd in enumerate(TASKS[name]):
        full = cmd + (extra if i == len(TASKS[name]) - 1 else [])
        code = subprocess.call(full, cwd=ROOT)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
