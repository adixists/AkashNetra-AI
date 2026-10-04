"""Logging setup and run-header helper (no bare ``print`` anywhere in the package)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from akashnetra.config import AppConfig, secret_status
from akashnetra.runinfo import get_git_hash, library_versions

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_HANDLER_TAG = "_akashnetra_handler"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once; calling it again only updates level/handlers.

    Args:
        level: Logging level name or number.
        log_file: Optional file that also receives the log stream.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    # Keep chatty third-party loggers quiet unless the user asks for DEBUG.
    for noisy in ("botocore", "s3fs", "fsspec", "urllib3", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_run_header(logger: logging.Logger, cfg: AppConfig, title: str = "run") -> None:
    """Log versions, git hash, data mode and the full config at the start of a run.

    Secrets are never logged; only whether they are set.
    """
    logger.info("=== AkashNetra AI %s ===", title)
    logger.info("model_version=%s git=%s", cfg.project.model_version, get_git_hash(cfg.root))
    logger.info("data_mode=%s", cfg.data_mode)
    if cfg.data_mode == "synthetic":
        logger.warning("SYNTHETIC DEMO DATA: results are not real forecast skill.")
    logger.info("python=%s libs=%s", sys.version.split()[0], json.dumps(library_versions()))
    logger.info("env_vars_set=%s", json.dumps(secret_status()))
    logger.info("seed=%d", cfg.model.seed)
    logger.info("config=%s", json.dumps(cfg.to_log_dict(), sort_keys=True))
