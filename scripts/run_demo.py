"""Placeholder entry point: not implemented until milestone M1 (synthetic demo)."""

from __future__ import annotations

import logging
import sys

from akashnetra.logging_setup import setup_logging

logger = logging.getLogger("run_demo")


def main() -> int:
    """Exit with a clear message instead of pretending to work."""
    setup_logging()
    logger.error("run_demo.py is not implemented yet (planned for milestone M1).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
