"""Assemble the unified (init_date, lead_day, box_id) table, label it, and save it."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from akashnetra.config import AppConfig
from akashnetra.features.labels import (
    THRESHOLD_FILENAME,
    apply_bust_labels,
    compute_bust_thresholds,
    save_thresholds,
    season_of,
)
from akashnetra.ingest import get_source
from akashnetra.ingest.base import ForecastSource, ObservationSource
from akashnetra.logging_setup import log_run_header
from akashnetra.processing.boxes import BoxGrid
from akashnetra.processing.qc import check_unified_table
from akashnetra.runinfo import get_git_hash, set_seeds

logger = logging.getLogger(__name__)

TABLE_FILENAME = "unified.parquet"
META_FILENAME = "dataset_meta.json"

COLUMN_ORDER: list[str] = [
    "init_date",
    "lead_day",
    "box_id",
    "valid_date",
    "lat",
    "lon",
    "fcst_rain_mean",
    "fcst_rain_spread",
    "n_members",
    "moisture_flux_850",
    "wind_shear_200_850",
    "z500",
    "t2m",
    "obs_rain",
    "abs_error",
    "bust_threshold",
    "bust",
    "year",
    "season",
    "split",
    "data_mode",
]


@dataclass(frozen=True)
class BuildResult:
    """Paths and summary of a dataset build."""

    table_path: Path
    thresholds_path: Path
    meta_path: Path
    summary: dict[str, Any]


def assemble_unified_table(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    grid: BoxGrid,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Join forecasts with observations and add error, year, season and split columns.

    Rows without an observation, outside the season or outside every split are dropped, and
    each drop is logged with its count (nothing is dropped silently).
    """
    table = forecasts.merge(observations, on=["valid_date", "box_id"], how="left")

    n_no_obs = int(table["obs_rain"].isna().sum())
    if n_no_obs:
        logger.warning("Dropping %d forecast rows without an observation", n_no_obs)
        table = table[table["obs_rain"].notna()]

    boxes = grid.table().set_index("box_id")
    table["lat"] = table["box_id"].map(boxes["lat"]).astype("float32")
    table["lon"] = table["box_id"].map(boxes["lon"]).astype("float32")
    table["abs_error"] = np.abs(
        table["fcst_rain_mean"].astype("float64") - table["obs_rain"].astype("float64")
    ).astype("float32")
    table["year"] = table["init_date"].dt.year.astype("int16")
    table["season"] = season_of(table["init_date"], cfg.season)

    n_off = int(table["season"].isna().sum())
    if n_off:
        logger.warning("Dropping %d rows outside season %s", n_off, cfg.season.name)
        table = table[table["season"].notna()]

    split = cfg.years.active_split
    table["split"] = table["year"].map(lambda y: split.split_of(int(y)))
    n_no_split = int(table["split"].isna().sum())
    if n_no_split:
        logger.warning("Dropping %d rows from years in no split", n_no_split)
        table = table[table["split"].notna()]

    table["data_mode"] = cfg.data_mode
    return table.sort_values(["init_date", "lead_day", "box_id"], ignore_index=True)


def summarise_busts(table: pd.DataFrame) -> dict[str, Any]:
    """Bust rates exactly as computed (overall by split and train rate per lead day)."""
    by_split = table.groupby("split")["bust"].agg(["mean", "size"])
    train = table[table["split"] == "train"]
    return {
        "bust_rate_by_split": {k: float(v) for k, v in by_split["mean"].items()},
        "n_rows_by_split": {k: int(v) for k, v in by_split["size"].items()},
        "train_bust_rate_by_lead_day": {
            int(k): float(v) for k, v in train.groupby("lead_day")["bust"].mean().items()
        },
    }


def build_dataset(
    cfg: AppConfig,
    source: ForecastSource | None = None,
    observations: ObservationSource | None = None,
) -> BuildResult:
    """Run the full M1 pipeline: load -> join -> QC -> train-only thresholds -> labels -> save.

    Args:
        cfg: Application config (``cfg.data_mode`` decides the source unless one is given).
        source: Forecast source override (tests).
        observations: Observation source override; defaults to ``source`` when it also is
            an :class:`ObservationSource` (the synthetic source is both).
    """
    log_run_header(logger, cfg, title="build_dataset")
    set_seeds(cfg.model.seed)
    source = source or get_source(cfg)
    obs_source = observations or (source if isinstance(source, ObservationSource) else None)
    if obs_source is None:
        raise ValueError("An observation source is required")
    if source.data_mode != cfg.data_mode:
        raise ValueError(
            f"Source data_mode '{source.data_mode}' does not match config '{cfg.data_mode}'"
        )

    grid = BoxGrid.from_config(cfg.region)
    split = cfg.years.active_split
    years = sorted({*split.train.years, *split.val.years, *split.test.years})
    logger.info("Years=%s boxes=%d lead_days=1..%d", years, grid.n_boxes, cfg.forecast.max_lead_day)

    forecasts = source.load_forecasts(years)
    observations_df = obs_source.load_observations(years)
    table = assemble_unified_table(forecasts, observations_df, grid, cfg)
    qc = check_unified_table(table, cfg.forecast.max_lead_day)
    logger.info("QC passed: %s", qc)

    thresholds = compute_bust_thresholds(table, cfg.bust, split.train.years)
    table = apply_bust_labels(table, thresholds, cfg.bust.per_lead_day)
    table = table[COLUMN_ORDER]

    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / TABLE_FILENAME
    table.to_parquet(table_path, index=False)
    thresholds_path = save_thresholds(
        thresholds,
        cfg.model_artifacts_dir / THRESHOLD_FILENAME,
        bust=cfg.bust,
        train_years=split.train.years,
        season=cfg.season,
        data_mode=cfg.data_mode,
    )

    summary = {
        "data_mode": cfg.data_mode,
        "data_label": cfg.data_label,
        "source": source.describe(),
        "years": {n: split.years_of(n) for n in ("train", "val", "test")},
        "active_year_mode": cfg.years.active,
        "bust_percentile": cfg.bust.percentile,
        "bust_per_lead_day": cfg.bust.per_lead_day,
        "n_rows": len(table),
        "n_boxes": grid.n_boxes,
        "qc": qc,
        **summarise_busts(table),
        "git_hash": get_git_hash(cfg.root),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": cfg.project.model_version,
    }
    meta_path = out_dir / META_FILENAME
    meta_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    logger.info("Wrote %s (%d rows), %s, %s", table_path, len(table), thresholds_path, meta_path)
    logger.info("Bust rate by split (as computed): %s", summary["bust_rate_by_split"])
    return BuildResult(table_path, thresholds_path, meta_path, summary)
