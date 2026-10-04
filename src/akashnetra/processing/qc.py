"""Quality control of the unified table. Violations raise; nothing is silently repaired."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Physical upper bound for a 24 h box-mean rainfall [mm]; anything above is a data error.
MAX_PLAUSIBLE_RAIN_MM = 1500.0

REQUIRED_COLUMNS: tuple[str, ...] = (
    "init_date",
    "lead_day",
    "box_id",
    "valid_date",
    "fcst_rain_mean",
    "fcst_rain_spread",
    "obs_rain",
    "abs_error",
)
KEY = ["init_date", "lead_day", "box_id"]


class QCError(ValueError):
    """Raised when the data violates a hard quality-control rule."""


def check_unified_table(df: pd.DataFrame, max_lead_day: int) -> dict[str, Any]:
    """Validate the unified table and return a summary of what was checked.

    Raises:
        QCError: On missing columns, NaNs in core columns, negative or implausible rain,
            negative spread, bad lead days, valid_date != init_date + lead_day, duplicated
            keys, or abs_error inconsistent with ``|fcst - obs|``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise QCError(f"Missing required columns: {missing}")
    if df.empty:
        raise QCError("The table is empty")

    for col in ("fcst_rain_mean", "fcst_rain_spread", "obs_rain", "abs_error"):
        n_nan = int(df[col].isna().sum())
        if n_nan:
            raise QCError(f"{n_nan} NaN values in core column '{col}'")
    for col in ("fcst_rain_mean", "obs_rain"):
        if (df[col] < 0).any():
            raise QCError(f"Negative rainfall in '{col}'")
        if (df[col] > MAX_PLAUSIBLE_RAIN_MM).any():
            raise QCError(f"Implausible rainfall (> {MAX_PLAUSIBLE_RAIN_MM} mm) in '{col}'")
    if (df["fcst_rain_spread"] < 0).any():
        raise QCError("Negative ensemble spread")
    if not df["lead_day"].between(1, max_lead_day).all():
        raise QCError(f"lead_day outside 1..{max_lead_day}")

    expected_valid = df["init_date"] + pd.to_timedelta(df["lead_day"].astype(int), unit="D")
    if not (df["valid_date"] == expected_valid).all():
        raise QCError("valid_date != init_date + lead_day for some rows")
    n_dup = int(df.duplicated(KEY).sum())
    if n_dup:
        raise QCError(f"{n_dup} duplicated (init_date, lead_day, box_id) keys")
    recomputed = np.abs(df["fcst_rain_mean"].astype("float64") - df["obs_rain"].astype("float64"))
    if not np.allclose(recomputed, df["abs_error"].astype("float64"), atol=1e-4):
        raise QCError("abs_error is inconsistent with |fcst_rain_mean - obs_rain|")

    other_nan = {
        c: int(df[c].isna().sum()) for c in df.select_dtypes("number").columns if df[c].isna().any()
    }
    if other_nan:
        logger.warning("NaNs in non-core columns: %s", other_nan)
    return {
        "n_rows": len(df),
        "n_boxes": int(df["box_id"].nunique()),
        "n_init_dates": int(df["init_date"].nunique()),
        "nan_in_non_core_columns": other_nan,
        "max_rain_mm": float(max(df["fcst_rain_mean"].max(), df["obs_rain"].max())),
    }
