import pandas as pd
import pytest

from akashnetra.features.build import (
    check_no_label_features,
    fit_feature_climatology,
)


def test_no_label_columns_in_features():
    check_no_label_features(["fcst_rain_mean", "moisture_flux_850"])

    with pytest.raises(ValueError, match="Label columns used as features"):
        check_no_label_features(["fcst_rain_mean", "bust"])

    with pytest.raises(ValueError, match="Label columns used as features"):
        check_no_label_features(["obs_rain"])


def test_feature_climatology_train_only():
    df = pd.DataFrame(
        {
            "init_date": pd.date_range("2015-01-01", periods=3),
            "lead_day": [1, 1, 1],
            "box_id": ["B1", "B2", "B3"],
            "year": [2015, 2016, 2017],  # mixing years
            "fcst_rain_spread": [0.1, 0.2, 0.3],
            "moisture_flux_850": [1, 2, 3],
            "z500": [10, 20, 30],
            "t2m": [300, 301, 302],
            "region_rain_z": [0, 0, 0],
            "lat": [15, 16, 17],
            "lon": [75, 76, 77],
            "fcst_rain_mean": [1, 2, 3],
        }
    )

    # Fit only on 2015 and 2016
    clim = fit_feature_climatology(df, train_years=[2015, 2016])

    # 2017 values (0.3, 3, 30, 302) should not influence the mean
    # moisture_flux_850 for B1 and B2 is 1 and 2, but wait!
    # They are in different boxes. B1 has 2015, B2 has 2016, B3 has 2017.
    # The groups are by (box_id, lead_day, month).
    # So B3 will just be missing in the climatology since it was 2017.

    assert "B1" in clim.anomalies["box_id"].values
    assert "B2" in clim.anomalies["box_id"].values
    assert "B3" not in clim.anomalies["box_id"].values
