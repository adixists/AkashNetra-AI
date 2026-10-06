from akashnetra.explain.reasons import generate_reason
from akashnetra.explain.rules import check_physics


def test_physics_rules():
    # Test dry_heavy_rain
    row_heavy = {"fcst_rain_mean": 60.0, "moisture_flux_850": 2.0}
    res = check_physics(row_heavy)
    assert not res["passed"]
    assert any(r["id"] == "dry_heavy_rain" for r in res["rules_triggered"])

    # Test dry_stable
    row_stable = {"fcst_rain_mean": 1.0, "fcst_rain_spread": 0.2}
    res = check_physics(row_stable)
    assert not res["passed"]
    assert any(r["id"] == "dry_stable" for r in res["rules_triggered"])

    # Test passed
    row_ok = {"fcst_rain_mean": 10.0, "fcst_rain_spread": 5.0, "moisture_flux_850": 15.0}
    res = check_physics(row_ok)
    assert res["passed"]
    assert len(res["rules_triggered"]) == 0


def test_generate_reason():
    drivers = [
        {"feature": "Moisture flux (850 hPa)", "raises_risk": True},
        {"feature": "Ensemble spread", "raises_risk": False},
    ]
    analogs = [{"bust": True}, {"bust": True}, {"bust": False}]

    reason = generate_reason(drivers, analogs)
    assert "moisture flux (850 hpa)" in reason.lower()
    assert "2 out of 3" in reason.lower()
