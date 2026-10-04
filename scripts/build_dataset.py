"""Placeholder entry point: not implemented until milestone M1/M2."""

from __future__ import annotations

import logging
import sys

from akashnetra.logging_setup import setup_logging

logger = logging.getLogger("build_dataset")


def main() -> int:
    """Exit with a clear message instead of pretending to work."""
    setup_logging()
    logger.error("build_dataset.py is not implemented yet (planned for milestones M1-M2).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
