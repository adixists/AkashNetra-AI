import numpy as np
import pandas as pd

from akashnetra.models.calibrate import Calibrator
from akashnetra.models.evaluate import (
    choose_alert_threshold,
    metrics_by_lead,
)


def test_choose_alert_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.95])

    # Perfect precision for top 3
    res = choose_alert_threshold(y_true, y_prob, precision_target=1.0)
    assert res["threshold"] <= 0.8  # Any value between 0.2 and 0.8 gives 1.0 precision
    assert res["target_met"] is True

    # If target unreachable, should fallback to max F1
    y_prob_bad = np.array([0.9, 0.9, 0.1, 0.1, 0.1])
    res_bad = choose_alert_threshold(y_true, y_prob_bad, precision_target=0.9)
    assert res_bad["target_met"] is False


def test_metrics_structure():
    df = pd.DataFrame(
        {
            "lead_day": [1, 1, 2, 2],
            "bust": [0, 1, 0, 1],
            "p_m1": [0.1, 0.9, 0.2, 0.8],
            "p_m2": [0.2, 0.8, 0.3, 0.7],
        }
    )

    metrics1 = metrics_by_lead(df, y_col="bust", p_col="p_m1", threshold=0.5)
    metrics2 = metrics_by_lead(df, y_col="bust", p_col="p_m2", threshold=0.5)

    metrics = {"p_m1": metrics1, "p_m2": metrics2}

    assert "p_m1" in metrics
    assert "p_m2" in metrics
    assert "overall" in metrics["p_m1"]
    assert "1" in metrics["p_m1"]["by_lead"]
    assert "2" in metrics["p_m1"]["by_lead"]
    assert metrics["p_m1"]["overall"]["pr_auc"] > 0
    assert metrics["p_m2"]["overall"]["pr_auc"] > 0


def test_calibrator():
    # Just a simple sanity check
    cal = Calibrator(method="isotonic")
    rng = np.random.default_rng()
    y_prob = rng.random(100)
    y_true = (y_prob > 0.5).astype(int)

    cal.fit(y_prob, y_true)
    y_cal = cal.transform(y_prob)

    assert len(y_cal) == 100
    assert (y_cal >= 0).all() and (y_cal <= 1).all()
