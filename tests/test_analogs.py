import pandas as pd
import pytest

from akashnetra.features.analogs import AnalogIndex, AnalogLeakageError


def test_analog_leakage_on_fit():
    df = pd.DataFrame(
        {
            "init_date": pd.date_range("2015-01-01", periods=3),
            "valid_date": pd.date_range("2015-01-02", periods=3),
            "lead_day": [1, 1, 1],
            "box_id": ["B1", "B2", "B3"],
            "year": [2015, 2016, 2017],  # mixing years
            "fcst_rain_mean": [1.0, 2.0, 3.0],
            "fcst_rain_spread": [0.1, 0.2, 0.3],
            "obs_rain": [1.0, 2.0, 3.0],
            "abs_error": [0.0, 0.0, 0.0],
            "bust": [0, 0, 0],
            "moisture_flux_850": [1, 1, 1],
            "wind_shear_200_850": [1, 1, 1],
            "z500_anom": [0, 0, 0],
            "t2m_anom": [0, 0, 0],
            "lat": [15.0, 16.0, 17.0],
            "lon": [75.0, 76.0, 77.0],
        }
    )

    # Should work if we only ask for 2015 and 2016 as train years
    # Wait, k=1 will fail if len(grp) <= k, we need at least 2 rows per lead for k=1
    df2 = pd.concat([df] * 2, ignore_index=True)
    df2["lead_day"] = [1, 1, 1, 2, 2, 2]  # 3 rows per lead, one for each year
    df3 = pd.concat(
        [df2] * 3, ignore_index=True
    )  # 9 rows per lead: 3 of 2015, 3 of 2016, 3 of 2017

    # This should pass because we explicitly pass train_years=[2015, 2016]
    # And it filters internally. Wait, the fit method filters df by train_years.
    # What if the user tries to query with a train year?
    index = AnalogIndex.fit(df3, train_years=[2015, 2016], k=2)
    assert set(index.library["year"].unique()) == {2015, 2016}

    # query() should raise if given a train year
    with pytest.raises(AnalogLeakageError, match="LOYO for train rows"):
        index.query(df3[df3["year"] == 2015])

    # query_leave_one_year_out() should work for train years
    # And should NOT include the same year in analogs
    ids, dist = index.query_leave_one_year_out(df3[df3["year"] == 2015])
    # ids contains indices into index.library
    analog_years = index.library.loc[ids.flatten(), "year"].values
    assert 2015 not in analog_years


def test_analog_leakage_error_internal():
    # Test assert_library_clean directly
    index = AnalogIndex(k=1, train_years=[2015])
    index.library = pd.DataFrame({"year": [2016]})
    with pytest.raises(AnalogLeakageError, match="Non-training years"):
        index.assert_library_clean()
