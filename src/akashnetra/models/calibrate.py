"""Probability calibration fitted on the VALIDATION years (isotonic default, Platt optional)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


class Calibrator:
    """Map raw model scores to calibrated probabilities."""

    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic") -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        """Fit on validation-set raw scores and labels."""
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration needs both classes in the validation labels")
        if self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(np.asarray(raw, dtype=np.float64), y)
        else:
            self._model = LogisticRegression().fit(self._logit(raw), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Calibrated probabilities in [0, 1]."""
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        if self.method == "isotonic":
            return self._model.predict(np.asarray(raw, dtype=np.float64))
        return self._model.predict_proba(self._logit(raw))[:, 1]
