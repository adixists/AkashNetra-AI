"""Abstract interfaces for forecast and observation sources.

Every source (GEFS, NCUM/NEPS, synthetic) must return tables in the same internal schema so
the rest of the pipeline is source-agnostic. The ``data_mode`` class attribute is what the
API and dashboard use to label results; it must be honest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from akashnetra.config import DataMode

#: Columns every forecast table must contain: one row per (init_date, lead_day, box_id).
FORECAST_COLUMNS: tuple[str, ...] = (
    "init_date",  # datetime64: forecast initialisation date
    "lead_day",  # int: 1..max_lead_day (24 h accumulation ending d days after init)
    "box_id",  # str: 1-degree box id (see processing.boxes)
    "valid_date",  # datetime64: init_date + lead_day days
    "fcst_rain_mean",  # float: ensemble-mean 24 h rainfall in the box [mm]
    "fcst_rain_spread",  # float: ensemble std of 24 h rainfall in the box [mm]
    "n_members",  # int: ensemble size used (not assumed constant across dates)
    "moisture_flux_850",  # float: specific humidity [g/kg] x wind speed [m/s] at 850 hPa
    "wind_shear_200_850",  # float: |V200 - V850| [m/s]
    "z500",  # float: 500 hPa geopotential height [gpm] (raw; anomalies come later)
    "t2m",  # float: 2 m temperature [K] (raw; anomalies come later)
)

#: Columns every observation table must contain: one row per (valid_date, box_id).
OBSERVATION_COLUMNS: tuple[str, ...] = ("valid_date", "box_id", "obs_rain")


class SourceError(RuntimeError):
    """Raised when a source cannot deliver the requested data (never silently ignored)."""


class ForecastSource(ABC):
    """A provider of NWP-style forecast tables on the 1° box grid."""

    #: Short machine name, e.g. ``"synthetic"`` or ``"gefs"``.
    name: ClassVar[str]
    #: Honest provenance label: ``"synthetic"`` or ``"real"``.
    data_mode: ClassVar[DataMode]

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-serialisable provenance description (stored in dataset metadata)."""

    @abstractmethod
    def load_forecasts(self, years: Sequence[int]) -> pd.DataFrame:
        """Return the forecast table for all initialisation dates of the given years.

        The result has exactly the columns in :data:`FORECAST_COLUMNS`.

        Raises:
            SourceError: If the data for some requested year is unavailable.
        """


class ObservationSource(ABC):
    """A provider of observed daily rainfall averaged onto the 1° box grid."""

    @abstractmethod
    def load_observations(self, years: Sequence[int]) -> pd.DataFrame:
        """Return observed rainfall for every valid date needed by forecasts of ``years``.

        The result has exactly the columns in :data:`OBSERVATION_COLUMNS`.

        Raises:
            SourceError: If the data for some requested year is unavailable.
        """


def check_columns(df: pd.DataFrame, required: Sequence[str], what: str) -> None:
    """Raise :class:`SourceError` if ``df`` lacks required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SourceError(f"{what} table is missing columns: {missing}")
