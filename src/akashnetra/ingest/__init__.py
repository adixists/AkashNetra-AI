"""Forecast, observation and reanalysis sources."""

from __future__ import annotations

from akashnetra.config import AppConfig
from akashnetra.ingest.base import ForecastSource, ObservationSource, SourceError
from akashnetra.ingest.synthetic import SyntheticSource

__all__ = ["ForecastSource", "ObservationSource", "SourceError", "SyntheticSource", "get_source"]


def get_source(cfg: AppConfig) -> SyntheticSource:
    """Return the source for the configured data mode.

    There is deliberately NO fallback: asking for ``real`` data when it is not available
    raises instead of silently generating synthetic data.
    """
    if cfg.data_mode == "synthetic":
        return SyntheticSource(cfg)
    raise NotImplementedError(
        "data_mode 'real' needs GEFSReforecastSource and IMDGriddedObservation, which are "
        "built in milestone M6. Set data_mode: synthetic for DEMO MODE. "
        "No automatic fallback to synthetic data is performed."
    )
