"""Phase 2 placeholder: PyTorch spatial CNN over forecast-field patches.

Not implemented in Phase 1. The planned design is documented here so the interface is
stable when the work starts:

* Input : tensor ``(channels, lat, lon)`` of ensemble-mean / spread / moisture-flux /
  shear / height-anomaly fields around a box for one ``(init_date, lead_day)``.
* Output: probability of a bust for the centre box, comparable (same labels, same split,
  same metrics) to the LightGBM model so the two can be compared fairly.
* Training must reuse the year-based split and train-only thresholds from Phase 1.
"""

from __future__ import annotations

from typing import Any


class SpatialCNNBustModel:
    """Placeholder for the Phase 2 spatial CNN bust model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "SpatialCNNBustModel is a Phase 2 stub (PyTorch spatial CNN). "
            "Phase 1 uses LightGBM; see models/train.py."
        )

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Train the model (Phase 2)."""
        raise NotImplementedError("Phase 2 stub.")

    def predict_proba(self, *args: Any, **kwargs: Any) -> Any:
        """Return bust probabilities (Phase 2)."""
        raise NotImplementedError("Phase 2 stub.")
