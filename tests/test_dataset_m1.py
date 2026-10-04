"""Tests for M1 dataset generation, synthetic source, and bust labelling with no leakage."""

from __future__ import annotations

import pandas as pd

from akashnetra.config import AppConfig, load_config
from akashnetra.features.labels import load_thresholds
from akashnetra.ingest.synthetic import SyntheticSource
from akashnetra.processing.dataset import BuildResult


def test_synthetic_source_shapes_and_reproducibility() -> None:
    cfg = load_config()
    source = SyntheticSource(cfg)
    fc1, obs1 = source._year(2015)
    fc2, obs2 = source._year(2015)
    pd.testing.assert_frame_equal(fc1, fc2)
    pd.testing.assert_frame_equal(obs1, obs2)

    assert set(fc1.columns) >= {
        "init_date",
        "lead_day",
        "box_id",
        "valid_date",
        "fcst_rain_mean",
        "fcst_rain_spread",
        "moisture_flux_850",
        "wind_shear_200_850",
    }
    assert (fc1["fcst_rain_mean"] >= 0).all()
    assert (fc1["fcst_rain_spread"] >= 0).all()
    assert (obs1["obs_rain"] >= 0).all()


def test_threshold_leakage_guard(small_build: tuple[AppConfig, BuildResult]) -> None:
    cfg, result = small_build
    thresh_df, meta = load_thresholds(result.thresholds_path)
    split = cfg.years.active_split

    # Thresholds must ONLY be computed on training years
    assert meta["train_years"] == split.train.years
    for val_yr in split.val.years:
        assert val_yr not in meta["train_years"]
    for test_yr in split.test.years:
        assert test_yr not in meta["train_years"]

    # Verify output parquet exists and is populated
    assert result.table_path.exists()
    df = pd.read_parquet(result.table_path)
    assert len(df) > 0
    assert set(df["split"].unique()) == {"train", "val", "test"}

    # Train bust rate should be approximately 10% (~0.10) by definition of 90th percentile
    train_rate = df[df["split"] == "train"]["bust"].mean()
    assert 0.08 <= train_rate <= 0.12, f"Train bust rate {train_rate} expected ~0.10"

    # All thresholds must respect the minimum-error floor
    assert (thresh_df["threshold"] >= cfg.bust.min_error_floor_mm).all()


def test_min_error_floor_elevates_trivial_errors() -> None:
    from akashnetra.config import BustConfig
    from akashnetra.features.labels import compute_bust_thresholds

    dummy_df = pd.DataFrame(
        {
            "year": [2015] * 100,
            "season": ["JJAS"] * 100,
            "box_id": ["B15N_74E"] * 100,
            "lead_day": [1] * 100,
            "abs_error": [0.2] * 100,  # All errors tiny (0.2 mm)
        }
    )
    bust_cfg = BustConfig(
        percentile=90.0,
        min_samples_per_group=30,
        per_lead_day=True,
        min_error_floor_mm=5.0,
    )
    thresh = compute_bust_thresholds(dummy_df, bust_cfg, train_years=[2015])
    assert thresh.loc[0, "threshold"] == 5.0
