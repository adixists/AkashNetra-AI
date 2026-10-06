"""Reference predictors every model must be compared against."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ClimatologyBaseline:
    """Constant bust probability per lead day = TRAIN-year bust rate at that lead day.

    With the minimum-error floor the train rate is not exactly 0.10 at every lead, so a
    per-lead constant is a slightly stronger (fairer) reference than a single 0.10.
    """

    name = "climatology"

    def __init__(self) -> None:
        self.rates_: dict[int, float] = {}
        self.overall_: float = float("nan")

    def fit(self, lead_day: pd.Series, y: pd.Series) -> ClimatologyBaseline:
        """Store train bust rates."""
        frame = pd.DataFrame({"lead_day": lead_day.to_numpy(), "y": y.to_numpy()})
        self.rates_ = {int(k): float(v) for k, v in frame.groupby("lead_day")["y"].mean().items()}
        self.overall_ = float(frame["y"].mean())
        return self

    def predict_proba(self, lead_day: pd.Series) -> np.ndarray:
        """Bust probability for each row (overall rate for unseen lead days)."""
        return lead_day.map(self.rates_).fillna(self.overall_).to_numpy(np.float64)


class SpreadLogisticBaseline:
    """Logistic regression on ensemble spread alone.

    Uses ``log1p(spread)``, a monotone transform of spread: ranking metrics (PR-AUC, ROC-AUC)
    are identical to using raw spread; the log only makes the fitted probabilities better
    behaved for a heavy-tailed input.
    """

    name = "spread_logistic"

    def __init__(self, seed: int = 42) -> None:
        self.model = LogisticRegression(random_state=seed)

    @staticmethod
    def _x(spread: pd.Series) -> np.ndarray:
        return np.log1p(spread.to_numpy(np.float64).clip(min=0)).reshape(-1, 1)

    def fit(self, spread: pd.Series, y: pd.Series) -> SpreadLogisticBaseline:
        """Fit on training rows."""
        self.model.fit(self._x(spread), y.to_numpy())
        return self

    def predict_proba(self, spread: pd.Series) -> np.ndarray:
        """Bust probability for each row."""
        return self.model.predict_proba(self._x(spread))[:, 1]
