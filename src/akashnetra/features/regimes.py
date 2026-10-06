"""Rule-based weather-regime tags derived from forecast fields (issue-time information only).

Two tags, both computed with thresholds from ``config.yaml`` (``regimes``):

* ``monsoon_phase`` (+1 active / 0 normal / -1 break): z-score of the region-mean ensemble-mean
  forecast rain for that (init_date, lead_day), against the TRAIN-year climatology per
  (lead_day, month).
* ``lps_flag`` (0/1): low-pressure proxy = box 500 hPa height anomaly well below normal AND
  moisture-flux anomaly well above normal.

These are transparent proxies, not an operational regime classifier. ERA5-based tags
(M6, optional) can replace them behind the same column names.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from akashnetra.config import RegimesConfig

REGIME_CLIM_KEYS = ["lead_day", "month"]


def _region_mean_rain(df: pd.DataFrame) -> pd.DataFrame:
    """Region-mean forecast rain per (init_date, lead_day), with month of the init date."""
    out = (
        df.groupby(["init_date", "lead_day"], observed=True)["fcst_rain_mean"]
        .mean()
        .rename("region_rain")
        .reset_index()
    )
    out["month"] = out["init_date"].dt.month.astype("int8")
    return out


def fit_regime_climatology(df: pd.DataFrame, train_years: Sequence[int]) -> pd.DataFrame:
    """Mean/std of region-mean forecast rain per (lead_day, month), TRAIN years only."""
    train = df[df["init_date"].dt.year.isin(list(train_years))]
    if train.empty:
        raise ValueError("No training rows to fit the regime climatology")
    rr = _region_mean_rain(train)
    clim = rr.groupby(REGIME_CLIM_KEYS)["region_rain"].agg(["mean", "std"]).reset_index()
    clim["std"] = clim["std"].clip(lower=1e-6).fillna(1e-6)
    return clim.rename(columns={"mean": "region_rain_mean", "std": "region_rain_std"})


def add_regime_tags(df: pd.DataFrame, clim: pd.DataFrame, cfg: RegimesConfig) -> pd.DataFrame:
    """Add ``region_rain_z``, ``monsoon_phase`` and ``lps_flag``.

    Requires ``z500_anom`` and ``moisture_flux_anom`` (see :mod:`akashnetra.features.build`).

    Raises:
        ValueError: If a (lead_day, month) group has no climatology.
    """
    rr = _region_mean_rain(df).merge(clim, on=REGIME_CLIM_KEYS, how="left")
    if rr["region_rain_mean"].isna().any():
        raise ValueError("Missing regime climatology for some (lead_day, month) groups")
    rr["region_rain_z"] = (rr["region_rain"] - rr["region_rain_mean"]) / rr["region_rain_std"]
    rr["monsoon_phase"] = np.select(
        [rr["region_rain_z"] > cfg.active_rain_z, rr["region_rain_z"] < cfg.break_rain_z],
        [1, -1],
        default=0,
    ).astype("int8")

    out = df.drop(columns=["region_rain_z", "monsoon_phase", "lps_flag"], errors="ignore")
    out = out.merge(
        rr[["init_date", "lead_day", "region_rain_z", "monsoon_phase"]],
        on=["init_date", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    out["region_rain_z"] = out["region_rain_z"].astype("float32")
    out["lps_flag"] = (
        (out["z500_anom"] < cfg.lps_z500_z) & (out["moisture_flux_anom"] > cfg.lps_moisture_z)
    ).astype("int8")
    return out
