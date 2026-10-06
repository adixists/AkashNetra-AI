"""Feature engineering on the unified table (issue-time information only).

Climatologies (for anomalies and regime tags) are fitted on TRAIN years only and then
applied unchanged to every row. ``bust_threshold``, ``obs_rain`` and ``abs_error`` are
label-side columns and are never used as features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from akashnetra.config import AppConfig
from akashnetra.features.regimes import add_regime_tags, fit_regime_climatology

#: Raw variable -> anomaly (z-score) column, climatology per (box, lead_day, month).
ANOMALY_VARS: dict[str, str] = {
    "fcst_rain_spread": "spread_anom",
    "moisture_flux_850": "moisture_flux_anom",
    "z500": "z500_anom",
    "t2m": "t2m_anom",
}
CLIM_KEYS = ["box_id", "lead_day", "month"]

BASE_FEATURES: list[str] = [
    "fcst_rain_mean",
    "fcst_rain_spread",
    "spread_anom",
    "moisture_flux_850",
    "moisture_flux_anom",
    "wind_shear_200_850",
    "z500_anom",
    "t2m_anom",
    "region_rain_z",
    "monsoon_phase",
    "lps_flag",
    "lat",
    "lon",
    "day_of_season",
    "lead_day",
]
ANALOG_FEATURES: list[str] = ["analog_bust_rate", "analog_mean_error"]
FEATURE_COLUMNS: list[str] = BASE_FEATURES + ANALOG_FEATURES

#: Columns that must never be model inputs (labels or label-derived).
LABEL_COLUMNS: frozenset[str] = frozenset({"obs_rain", "abs_error", "bust", "bust_threshold"})


@dataclass(frozen=True)
class FeatureClimatology:
    """Train-only climatologies needed to build features for any row."""

    anomalies: pd.DataFrame
    regimes: pd.DataFrame


def _with_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["init_date"].dt.month.astype("int8")
    return out


def fit_feature_climatology(df: pd.DataFrame, train_years: Sequence[int]) -> FeatureClimatology:
    """Fit anomaly and regime climatologies on TRAIN-year rows only."""
    train = _with_month(df[df["year"].isin(list(train_years))])
    if train.empty:
        raise ValueError("No training rows to fit the feature climatology")
    g = train.groupby(CLIM_KEYS, observed=True)
    parts = []
    for var in ANOMALY_VARS:
        stats = g[var].agg(["mean", "std"])
        stats.columns = [f"{var}__mean", f"{var}__std"]
        parts.append(stats)
    anomalies = pd.concat(parts, axis=1).reset_index()
    for var in ANOMALY_VARS:
        col = f"{var}__std"
        anomalies[col] = anomalies[col].clip(lower=1e-6).fillna(1e-6)
    return FeatureClimatology(anomalies, fit_regime_climatology(df, train_years))


def add_anomalies(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Add z-score anomaly columns from a fitted climatology.

    Raises:
        ValueError: If any row has no climatology for its (box, lead_day, month).
    """
    out = _with_month(df).merge(clim, on=CLIM_KEYS, how="left", validate="many_to_one")
    if out[f"{next(iter(ANOMALY_VARS))}__mean"].isna().any():
        raise ValueError("Missing feature climatology for some (box, lead_day, month) groups")
    for var, name in ANOMALY_VARS.items():
        out[name] = ((out[var] - out[f"{var}__mean"]) / out[f"{var}__std"]).astype("float32")
    return out.drop(columns=[c for c in clim.columns if "__" in c])


def add_calendar(df: pd.DataFrame, season_months: Sequence[int]) -> pd.DataFrame:
    """Add ``day_of_season`` (days since the first day of the season, by init date)."""
    out = df.copy()
    start = pd.to_datetime(
        {"year": out["init_date"].dt.year, "month": min(season_months), "day": 1}
    )
    out["day_of_season"] = (out["init_date"] - start).dt.days.astype("int16")
    return out


def build_base_features(df: pd.DataFrame, clim: FeatureClimatology, cfg: AppConfig) -> pd.DataFrame:
    """Add every non-analog feature to the unified table."""
    out = add_anomalies(df, clim.anomalies)
    out = add_regime_tags(out, clim.regimes, cfg.regimes)
    return add_calendar(out, cfg.season.months)


def check_no_label_features(columns: Sequence[str]) -> None:
    """Raise if any label-side column is used as a feature."""
    bad = LABEL_COLUMNS.intersection(columns)
    if bad:
        raise ValueError(f"Label columns used as features: {sorted(bad)}")
