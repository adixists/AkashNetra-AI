"""Analog search: the K most similar past forecasts at the same lead day, TRAIN years only.

Leakage rules enforced here (and tested in ``tests/test_analogs.py``):

* The library contains TRAIN-year rows only.
* Validation/test rows are matched against the whole library (never against their own or
  any other val/test year).
* Training rows are matched leave-one-year-out (analogs from other train years only), so
  their analog features are distributed like those of unseen years and never contain the
  row's own label.

Similarity is Euclidean distance in standardised feature space; ``similarity = 1/(1+d)``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Standardised inputs that define "similar forecast situation".
ANALOG_INPUTS: list[str] = [
    "fcst_rain_mean",
    "fcst_rain_spread",
    "moisture_flux_850",
    "wind_shear_200_850",
    "z500_anom",
    "t2m_anom",
    "lat",
    "lon",
]
LIBRARY_COLUMNS: list[str] = [
    "init_date",
    "valid_date",
    "lead_day",
    "box_id",
    "year",
    "fcst_rain_mean",
    "fcst_rain_spread",
    "obs_rain",
    "abs_error",
    "bust",
]


class AnalogLeakageError(RuntimeError):
    """Raised if a non-training year would enter the analog library."""


@dataclass
class _LeadModel:
    scaler: StandardScaler
    nn: NearestNeighbors
    lib_ids: np.ndarray  # library ids of the rows the NN was fitted on


@dataclass
class AnalogIndex:
    """Per-lead-day nearest-neighbour index over the TRAIN-year analog library."""

    k: int
    train_years: list[int]
    inputs: list[str] = field(default_factory=lambda: list(ANALOG_INPUTS))
    library: pd.DataFrame = field(default_factory=pd.DataFrame)
    _models: dict[int, _LeadModel] = field(default_factory=dict, repr=False)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        train_years: Sequence[int],
        k: int,
        inputs: Sequence[str] | None = None,
    ) -> AnalogIndex:
        """Build the library from TRAIN-year rows of ``df`` and fit one index per lead day."""
        inputs = list(inputs or ANALOG_INPUTS)
        train_years = sorted(int(y) for y in train_years)
        cols = list(dict.fromkeys(LIBRARY_COLUMNS + inputs))
        lib = df.loc[df["year"].isin(train_years), cols].reset_index(drop=True)
        lib.insert(0, "lib_id", np.arange(len(lib), dtype=np.int64))
        index = cls(k=k, train_years=train_years, inputs=inputs, library=lib)
        index.assert_library_clean()
        for lead, grp in lib.groupby("lead_day"):
            if len(grp) <= k:
                raise ValueError(f"Lead day {lead}: library has {len(grp)} rows, need > k={k}")
            scaler = StandardScaler().fit(grp[inputs].to_numpy(np.float64))
            nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(grp[inputs]))
            index._models[int(lead)] = _LeadModel(scaler, nn, grp["lib_id"].to_numpy())
        logger.info("Analog library: %d train rows, years %s, k=%d", len(lib), train_years, k)
        return index

    def assert_library_clean(self) -> None:
        """Raise :class:`AnalogLeakageError` if any library row is outside the train years."""
        bad = set(self.library["year"].unique()) - set(self.train_years)
        if bad:
            raise AnalogLeakageError(f"Non-training years in the analog library: {sorted(bad)}")

    def query(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Find analogs for rows of NON-training years (val/test/new forecasts).

        Returns:
            ``(lib_ids, distances)``, both shaped ``(len(rows), k)``, nearest first.

        Raises:
            AnalogLeakageError: If ``rows`` contains training-year rows (use
                :meth:`query_leave_one_year_out` for those).
        """
        if rows["year"].isin(self.train_years).any():
            raise AnalogLeakageError("query() is for non-training rows; use LOYO for train rows")
        ids = np.empty((len(rows), self.k), dtype=np.int64)
        dist = np.empty((len(rows), self.k), dtype=np.float32)
        lead_values = rows["lead_day"].to_numpy()
        for lead in np.unique(lead_values):
            pos = np.flatnonzero(lead_values == lead)
            model = self._models[int(lead)]
            x = model.scaler.transform(rows.iloc[pos][self.inputs].to_numpy(np.float64))
            d, j = model.nn.kneighbors(x, n_neighbors=self.k)
            ids[pos], dist[pos] = model.lib_ids[j], d
        return ids, dist

    def query_leave_one_year_out(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Find analogs for TRAINING rows using only library rows from other train years."""
        if not rows["year"].isin(self.train_years).all():
            raise ValueError("query_leave_one_year_out() expects training-year rows only")
        if len(self.train_years) < 2:
            raise ValueError("Leave-one-year-out analogs need at least two training years")
        ids = np.empty((len(rows), self.k), dtype=np.int64)
        dist = np.empty((len(rows), self.k), dtype=np.float32)
        lib = self.library
        lead_values, year_values = rows["lead_day"].to_numpy(), rows["year"].to_numpy()
        for lead in np.unique(lead_values):
            model = self._models[int(lead)]
            lib_lead = lib[lib["lead_day"] == lead]
            for year in np.unique(year_values[lead_values == lead]):
                pos = np.flatnonzero((lead_values == lead) & (year_values == year))
                other = lib_lead[lib_lead["year"] != year]
                nn = NearestNeighbors(n_neighbors=self.k).fit(
                    model.scaler.transform(other[self.inputs].to_numpy(np.float64))
                )
                x = model.scaler.transform(rows.iloc[pos][self.inputs].to_numpy(np.float64))
                d, j = nn.kneighbors(x, n_neighbors=self.k)
                ids[pos], dist[pos] = other["lib_id"].to_numpy()[j], d
        return ids, dist

    def neighbours_for(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Analogs for any rows: LOYO for training years, full library otherwise."""
        ids = np.empty((len(rows), self.k), dtype=np.int64)
        dist = np.empty((len(rows), self.k), dtype=np.float32)
        is_train = rows["year"].isin(self.train_years).to_numpy()
        if is_train.any():
            ids[is_train], dist[is_train] = self.query_leave_one_year_out(rows[is_train])
        if (~is_train).any():
            ids[~is_train], dist[~is_train] = self.query(rows[~is_train])
        return ids, dist

    def analog_features(self, ids: np.ndarray) -> pd.DataFrame:
        """``analog_bust_rate`` and ``analog_mean_error`` from neighbour ids."""
        bust = self.library["bust"].to_numpy(np.float32)[ids]
        err = self.library["abs_error"].to_numpy(np.float32)[ids]
        return pd.DataFrame(
            {"analog_bust_rate": bust.mean(axis=1), "analog_mean_error": err.mean(axis=1)}
        )

    def evidence(self, ids: Sequence[int], distances: Sequence[float]) -> list[dict]:
        """Human-readable records of analog cases (for the explanation panel)."""
        rows = self.library.set_index("lib_id").loc[list(ids)]
        out = []
        for (lib_id, r), d in zip(rows.iterrows(), distances, strict=True):
            out.append(
                {
                    "lib_id": int(lib_id),
                    "init_date": r["init_date"].date().isoformat(),
                    "valid_date": r["valid_date"].date().isoformat(),
                    "box_id": str(r["box_id"]),
                    "lead_day": int(r["lead_day"]),
                    "similarity": float(1.0 / (1.0 + float(d))),
                    "fcst_rain_mm": float(r["fcst_rain_mean"]),
                    "obs_rain_mm": float(r["obs_rain"]),
                    "abs_error_mm": float(r["abs_error"]),
                    "bust": bool(r["bust"]),
                }
            )
        return out


def neighbours_frame(ids: np.ndarray, dist: np.ndarray, row_ids: np.ndarray) -> pd.DataFrame:
    """Tidy table of neighbour ids/distances per row (stored as an artifact)."""
    data: dict[str, np.ndarray] = {"row_id": row_ids.astype(np.int64)}
    for j in range(ids.shape[1]):
        data[f"analog_id_{j}"] = ids[:, j]
    for j in range(ids.shape[1]):
        data[f"analog_dist_{j}"] = dist[:, j].astype(np.float32)
    return pd.DataFrame(data)
