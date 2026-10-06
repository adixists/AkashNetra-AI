"""Train models on the existing unified dataset (build it first with build_dataset.py)."""

from __future__ import annotations

import logging
import sys

from akashnetra.config import load_config
from akashnetra.logging_setup import setup_logging
from akashnetra.models.train import run_training

logger = logging.getLogger("train")


def main() -> int:
    """Train, evaluate and save artifacts; return a process exit code."""
    setup_logging()
    cfg = load_config()
    try:
        result = run_training(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Metrics: %s", result.metrics_path)
    logger.info("Reliability plot: %s", result.reliability_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
