"""Physics sanity checks on the input features and forecast."""

from __future__ import annotations

from typing import Any


def check_physics(row: dict[str, Any]) -> dict[str, Any]:
    """Apply rule-based sanity checks to a forecast row.

    Returns:
        dict: ``passed`` (bool), ``rules_triggered`` (list of dicts).
    """
    triggered = []

    # Rule 1: High rain, low moisture flux
    fcst_rain = float(row.get("fcst_rain_mean", 0.0))
    moisture = float(row.get("moisture_flux_850", 0.0))
    if fcst_rain > 50.0 and moisture < 5.0:
        triggered.append(
            {
                "id": "dry_heavy_rain",
                "message": (
                    "Heavy rain forecast but 850 hPa moisture flux is very low. "
                    "Physically unlikely."
                ),
            }
        )

    # Rule 2: High probability but low spread and low rain
    # Assuming the alert check is done outside, or we just check if it's a dry stable situation
    spread = float(row.get("fcst_rain_spread", 0.0))
    if fcst_rain < 2.0 and spread < 0.5:
        triggered.append(
            {
                "id": "dry_stable",
                "message": (
                    "Low forecast rainfall and low ensemble spread. "
                    "If a bust occurs, the absolute error may still be small."
                ),
            }
        )

    return {"passed": len(triggered) == 0, "rules_triggered": triggered}
