"""Natural language synthesis for alert reasons."""

from __future__ import annotations

from typing import Any


def generate_reason(
    drivers: list[dict[str, Any]], analogs: list[dict[str, Any]] | None = None
) -> str:
    """Generate a plain-language reason string from SHAP drivers and analogs.

    Args:
        drivers: Top SHAP drivers from Explainer.explain_row().
        analogs: List of analog evidence dictionaries.

    Returns:
        str: Human-readable sentence summarizing why the alert fired.
    """
    if not drivers:
        return "No specific drivers identified."

    # Get the top driver that raises risk
    top_risk = next((d for d in drivers if d["raises_risk"]), None)

    parts = []

    if top_risk:
        feat = top_risk["feature"]
        parts.append(f"The primary driver is elevated risk from {feat.lower()}.")
    else:
        parts.append("Multiple interacting factors contribute to the bust risk.")

    # Add analog context if available
    if analogs:
        num_busts = sum(1 for a in analogs if a["bust"])
        k = len(analogs)
        parts.append(f"{num_busts} out of {k} similar past forecasts resulted in a bust.")

    return " ".join(parts)
