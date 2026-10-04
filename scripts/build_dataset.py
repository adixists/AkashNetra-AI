"""Build the unified dataset for the configured data mode (real mode arrives in M6)."""

from __future__ import annotations

import logging
import sys

from akashnetra.config import load_config
from akashnetra.logging_setup import setup_logging
from akashnetra.processing.dataset import build_dataset

logger = logging.getLogger("build_dataset")


def main() -> int:
    """Build and label the dataset; return a process exit code."""
    setup_logging()
    cfg = load_config()
    try:
        result = build_dataset(cfg)
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Done: %s", result.table_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
