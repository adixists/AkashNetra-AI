"""SHAP explanations for LightGBM model predictions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

FEATURE_NAMES = {
    "fcst_rain_mean": "Forecast rainfall",
    "fcst_rain_spread": "Ensemble spread",
    "spread_anom": "Ensemble spread anomaly",
    "moisture_flux_850": "Moisture flux (850 hPa)",
    "moisture_flux_anom": "Moisture flux anomaly",
    "wind_shear_200_850": "Vertical wind shear",
    "z500_anom": "500 hPa geopotential anomaly",
    "t2m_anom": "2 m temperature anomaly",
    "region_rain_z": "Regional rainfall anomaly",
    "monsoon_phase": "Monsoon active/break phase",
    "lps_flag": "Low pressure system flag",
    "lat": "Latitude",
    "lon": "Longitude",
    "day_of_season": "Day of season",
    "lead_day": "Lead day",
    "analog_bust_rate": "Past similar cases that busted",
    "analog_mean_error": "Historical analog mean error",
}


class Explainer:
    def __init__(self, model: LGBMClassifier) -> None:
        self.model = model
        # TreeExplainer works directly on the Booster object
        self.explainer = shap.TreeExplainer(self.model.booster_)

    def explain_row(self, row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the top N SHAP drivers for a single row.

        Args:
            row: DataFrame containing exactly 1 row with all features.
            top_n: Number of top features to return.

        Returns:
            List of dicts with ``feature`` (plain name), ``value`` (feature value),
            ``shap_value`` (contribution to log-odds), and ``raises_risk`` (bool).
        """
        if len(row) != 1:
            raise ValueError("explain_row expects exactly one row")

        shap_vals = self.explainer.shap_values(row)
        # In LightGBM binary classification, shap_values might be a list of arrays (one per class)
        # or a single array. Usually, shap.TreeExplainer returns a list [negative, positive].
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Take the positive class

        # Extract the single row's values
        vals = shap_vals[0]

        # Match features with SHAP values
        features = row.columns.tolist()

        # Sort by absolute magnitude
        idx = np.argsort(np.abs(vals))[::-1][:top_n]

        out = []
        for i in idx:
            feat_id = features[i]
            val = float(vals[i])
            out.append(
                {
                    "feature": FEATURE_NAMES.get(feat_id, feat_id),
                    "feature_id": feat_id,
                    "feature_value": float(row.iloc[0, i]),
                    "shap_value": val,
                    "raises_risk": val > 0,
                }
            )

        return out
