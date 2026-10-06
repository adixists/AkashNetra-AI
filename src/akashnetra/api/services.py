"""Data and ML services for the FastAPI backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from akashnetra.config import AppConfig, load_config
from akashnetra.explain.reasons import generate_reason
from akashnetra.explain.rules import check_physics
from akashnetra.explain.shap_explain import Explainer
from akashnetra.features.build import FEATURE_COLUMNS
from akashnetra.processing.boxes import BoxGrid

logger = logging.getLogger(__name__)

class ForecastService:
    """Provides data and explanations for the API."""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        if cfg is None:
            # Assume ROOT is two levels up from src/akashnetra/api
            # Wait, Path(__file__).parents[3] is AkashNetra AI
            root = Path(__file__).resolve().parents[3]
            cfg = load_config(root / "config.yaml")

        self.cfg = cfg
        self.data_mode = cfg.data_mode
        self.model_version = cfg.project.model_version
        self.proc_dir = cfg.processed_dir
        self.art_dir = cfg.model_artifacts_dir

        self.grid = BoxGrid.from_config(cfg.region)
        self._df: pd.DataFrame | None = None
        self._metrics: dict[str, Any] | None = None
        self._explainer: Explainer | None = None
        self._analog_index = None
        self._analog_neighbors: pd.DataFrame | None = None

        self.load_data()

    def load_data(self) -> None:
        """Load predictions, metrics, and models."""
        pred_path = self.proc_dir / "predictions.parquet"
        if pred_path.exists():
            self._df = pd.read_parquet(pred_path)
            # Ensure dates are strings for JSON
            self._df["init_date_str"] = self._df["init_date"].dt.strftime("%Y-%m-%d")
        else:
            logger.warning("predictions.parquet not found")
            self._df = pd.DataFrame()

        metrics_path = self.art_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                self._metrics = json.load(f)
        else:
            self._metrics = {}

        lgbm_path = self.art_dir / "lightgbm.joblib"
        if lgbm_path.exists():
            lgbm = joblib.load(lgbm_path)
            self._explainer = Explainer(lgbm)

        idx_path = self.art_dir / "analog_index.joblib"
        if idx_path.exists():
            self._analog_index = joblib.load(idx_path)

        nbr_path = self.proc_dir / "analog_neighbors.parquet"
        if nbr_path.exists():
            self._analog_neighbors = pd.read_parquet(nbr_path)

    def get_init_dates(self) -> list[str]:
        if self._df is None or self._df.empty:
            return []
        dates = self._df["init_date_str"].unique()
        return sorted(list(dates), reverse=True)

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics or {}

    def get_alerts(self, init_date: str, lead_day: int, threshold: float) -> list[dict[str, Any]]:
        """Return GeoJSON features for the given date and lead day."""
        if self._df is None or self._df.empty:
            return []

        mask = (self._df["init_date_str"] == init_date) & (self._df["lead_day"] == lead_day)
        sub = self._df[mask]

        features = []
        for _, row in sub.iterrows():
            prob = float(row["p_lightgbm"])
            conf = 1.0 - prob
            is_alert = prob >= threshold

            # Determine level
            level = "Low"
            if prob >= 0.8:
                level = "High"
            elif prob >= 0.5:
                level = "Medium"

            box_id = row["box_id"]

            lat_center = float(row["lat"])
            lon_center = float(row["lon"])
            half_size = self.grid.size / 2.0

            lat_min, lat_max = lat_center - half_size, lat_center + half_size
            lon_min, lon_max = lon_center - half_size, lon_center + half_size

            # GeoJSON Polygon
            coords = [[
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min]
            ]]

            props = {
                "box_id": box_id,
                "bust_prob": prob,
                "confidence": conf,
                "alert": is_alert,
                "fcst_rain_mm": float(row["fcst_rain_mean"]),
                "spread_mm": float(row["fcst_rain_spread"]),
                "level": level
            }

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": props
            })

        return features

    def get_summary(self, init_date: str, threshold: float) -> list[dict[str, Any]]:
        """Summary across all lead days for the D1-D10 strip."""
        if self._df is None or self._df.empty:
            return []

        sub = self._df[self._df["init_date_str"] == init_date]
        if sub.empty:
            return []

        tiles = []
        for lead in range(1, 11):
            g = sub[sub["lead_day"] == lead]
            if g.empty:
                continue
            probs = g["p_lightgbm"].to_numpy()
            n_alerts = int((probs >= threshold).sum())
            max_prob = float(probs.max()) if len(probs) > 0 else 0.0
            mean_conf = float((1.0 - probs).mean()) if len(probs) > 0 else 0.0

            tiles.append({
                "lead_day": lead,
                "n_alerts": n_alerts,
                "max_prob": max_prob,
                "mean_confidence": mean_conf
            })

        return tiles

    def get_box_detail(
        self, box_id: str, init_date: str, lead_day: int, threshold: float
    ) -> dict[str, Any]:
        """Full details for a single box."""
        if self._df is None or self._df.empty:
            raise ValueError("No data available")

        mask = (
            (self._df["init_date_str"] == init_date)
            & (self._df["lead_day"] == lead_day)
            & (self._df["box_id"] == box_id)
        )
        sub = self._df[mask]

        if sub.empty:
            raise ValueError(f"Box {box_id} not found for {init_date} lead {lead_day}")

        row = sub.iloc[0]
        row_dict = row.to_dict()

        prob = float(row["p_lightgbm"])

        drivers = []
        if self._explainer:
            # Need to pass just the feature columns as a single-row DataFrame
            feat_df = sub[FEATURE_COLUMNS]
            drivers = self._explainer.explain_row(feat_df)

        analogs = []
        if self._analog_index and self._analog_neighbors is not None:
            # find row_id matching this row.
            # query the index on the fly is safer!
            ids, dists = self._analog_index.query(sub)
            analogs = self._analog_index.evidence(ids[0], dists[0])

        physics = check_physics(row_dict)
        reason = generate_reason(drivers, analogs)

        return {
            "box_id": box_id,
            "init_date": init_date,
            "lead_day": lead_day,
            "bust_prob": prob,
            "confidence": 1.0 - prob,
            "alert": prob >= threshold,
            "drivers": drivers,
            "analogs": analogs,
            "physics": physics,
            "reason": reason,
            "data_mode": self.data_mode,
            "model_version": self.model_version
        }
