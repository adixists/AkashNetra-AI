"""DEMO MODE: build the synthetic dataset end to end (M1; later milestones extend it).

Always uses SYNTHETIC data and says so loudly; it never touches real-data outputs because
results are written to the ``synthetic`` subdirectories.
"""

from __future__ import annotations

import logging
import sys

from akashnetra.config import load_config
from akashnetra.logging_setup import setup_logging
from akashnetra.models.train import run_training
from akashnetra.processing.dataset import build_dataset

logger = logging.getLogger("run_demo")


def main() -> int:
    """Run the synthetic demo pipeline; return a process exit code."""
    setup_logging()
    cfg = load_config()
    if cfg.data_mode != "synthetic":
        logger.warning("run_demo forces data_mode=synthetic (config said '%s')", cfg.data_mode)
        cfg = cfg.model_copy(update={"data_mode": "synthetic"})
    logger.warning("DEMO MODE: everything below is SYNTHETIC DEMO DATA, not real weather.")
    result = build_dataset(cfg)
    logger.info("Demo dataset ready: %s", result.table_path)
    logger.info("Bust thresholds (train years only): %s", result.thresholds_path)
    trained = run_training(cfg)
    logger.info("Demo models trained. Metrics (SYNTHETIC): %s", trained.metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
