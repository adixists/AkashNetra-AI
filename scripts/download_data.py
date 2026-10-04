"""Placeholder entry point: not implemented until milestone M6 (real data)."""

from __future__ import annotations

import logging
import sys

from akashnetra.logging_setup import setup_logging

logger = logging.getLogger("download_data")


def main() -> int:
    """Exit with a clear message instead of pretending to work."""
    setup_logging()
    logger.error("download_data.py is not implemented yet (planned for milestone M6).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
