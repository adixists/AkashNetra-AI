"""Bust labelling with TRAIN-ONLY percentile thresholds.

Definition (see README): ``bust = 1`` if ``abs_error`` is strictly above the P-th percentile
(default 90) of that box's errors in that season. The percentile is estimated on the
training years only, stored as an artifact, and then applied unchanged to validation and
test years so no information from them leaks into the labels.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from akashnetra.config import BustConfig, SeasonConfig
from akashnetra.runinfo import get_git_hash

logger = logging.getLogger(__name__)

THRESHOLD_FILENAME = "bust_thresholds.json"


class LabelError(ValueError):
    """Raised when thresholds cannot be computed or applied."""


def season_of(dates: pd.Series | pd.DatetimeIndex, season: SeasonConfig) -> np.ndarray:
    """Season name for each date (by month); ``None`` for dates outside the season."""
    months = pd.DatetimeIndex(dates).month
    return np.where(months.isin(season.months), season.name, None)


def group_keys(per_lead_day: bool) -> list[str]:
    """Columns that define one threshold group."""
    return ["box_id", "season"] + (["lead_day"] if per_lead_day else [])


def compute_bust_thresholds(
    df: pd.DataFrame, bust: BustConfig, train_years: Sequence[int]
) -> pd.DataFrame:
    """Compute per-group error percentiles from the TRAINING years only.

    Args:
        df: Table with ``year``, ``season``, ``box_id``, ``lead_day`` and ``abs_error``.
        bust: Bust configuration (percentile, grouping, minimum samples).
        train_years: The only years allowed to influence the thresholds. Rows of any other
            year are ignored, even if present in ``df``.

    Returns:
        One row per group: group keys, ``threshold`` and ``n_train``.

    Raises:
        LabelError: If a group has fewer than ``min_samples_per_group`` training samples.
    """
    keys = group_keys(bust.per_lead_day)
    train = df[df["year"].isin(list(train_years)) & df["season"].notna()]
    if train.empty:
        raise LabelError("No training rows available to compute bust thresholds")
    grouped = train.groupby(keys, observed=True, sort=True)["abs_error"]
    q = grouped.quantile(bust.percentile / 100.0)
    if bust.min_error_floor_mm > 0:
        q = np.maximum(q, float(bust.min_error_floor_mm))
    out = q.rename("threshold").to_frame()
    out["n_train"] = grouped.size()
    out = out.reset_index()
    too_small = out[out["n_train"] < bust.min_samples_per_group]
    if not too_small.empty:
        raise LabelError(
            f"{len(too_small)} threshold groups have fewer than "
            f"{bust.min_samples_per_group} training samples (e.g. "
            f"{too_small.iloc[0][keys].to_dict()})"
        )
    return out


def apply_bust_labels(
    df: pd.DataFrame, thresholds: pd.DataFrame, per_lead_day: bool
) -> pd.DataFrame:
    """Add ``bust_threshold`` and ``bust`` (strictly above threshold) to ``df``.

    The thresholds are applied unchanged to every row, whatever its year.

    Raises:
        LabelError: If any row has no threshold (never filled in silently).
    """
    keys = group_keys(per_lead_day)
    out = df.drop(columns=[c for c in ("bust_threshold", "bust") if c in df.columns])
    out = out.merge(
        thresholds[[*keys, "threshold"]].rename(columns={"threshold": "bust_threshold"}),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    n_missing = int(out["bust_threshold"].isna().sum())
    if n_missing:
        raise LabelError(f"{n_missing} rows have no bust threshold for their group {keys}")
    out["bust"] = (out["abs_error"].astype("float64") > out["bust_threshold"]).astype(np.int8)
    return out


def save_thresholds(
    thresholds: pd.DataFrame,
    path: str | Path,
    *,
    bust: BustConfig,
    train_years: Sequence[int],
    season: SeasonConfig,
    data_mode: str,
) -> Path:
    """Write thresholds plus the metadata needed to audit them to a JSON artifact."""
    meta: dict[str, Any] = {
        "percentile": bust.percentile,
        "per_lead_day": bust.per_lead_day,
        "min_error_floor_mm": bust.min_error_floor_mm,
        "group_keys": group_keys(bust.per_lead_day),
        "min_samples_per_group": bust.min_samples_per_group,
        "train_years": list(train_years),
        "season": season.name,
        "season_months": season.months,
        "data_mode": data_mode,
        "n_groups": len(thresholds),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": get_git_hash(),
        "note": "Computed on TRAIN years only; applied unchanged to validation/test years.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = thresholds.to_dict(orient="records")
    path.write_text(json.dumps({"meta": meta, "thresholds": records}, indent=1), encoding="utf-8")
    logger.info("Saved %d bust thresholds to %s", len(thresholds), path)
    return path


def load_thresholds(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the thresholds artifact and its metadata."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(payload["thresholds"]), payload["meta"]
