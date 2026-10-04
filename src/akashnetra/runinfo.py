"""Small run-metadata helpers: git hash, library versions, seeding."""

from __future__ import annotations

import logging
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "xarray",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "shap",
    "fastapi",
    "streamlit",
)


def get_git_hash(cwd: Path | None = None) -> str:
    """Return the short git commit hash, or ``"unknown"`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    """Installed versions of the key libraries (``"not installed"`` if absent)."""
    versions: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy RNGs. Model libraries receive the seed via their params."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (global seed for third-party code paths)
    logger.debug("Seeds set to %d", seed)
