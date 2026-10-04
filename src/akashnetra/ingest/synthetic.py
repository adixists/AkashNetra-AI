"""DEMO MODE: synthetic forecasts and observations.

**This is not weather data.** It is a transparent random generator whose only purpose is to
exercise the whole pipeline when real data is unavailable. Everything produced here is
labelled ``data_mode = "synthetic"`` ("SYNTHETIC DEMO DATA") in the dataset, the API and the
dashboard. Skill measured on this data says nothing about real forecast skill.

Generative story (all coefficients are in :class:`SyntheticParams`):

1. **Truth.** A smooth, temporally persistent latent field ``L`` on a 0.25° grid
   (AR(1) in time, Gaussian-smoothed noise in space) is turned into daily rainfall with a
   monsoon seasonal cycle and a fixed spatial pattern: ``rain = clim * relu(L + 0.2)**1.5``
   (normalised). Observations are this field averaged onto the 1° boxes with
   :func:`akashnetra.processing.regrid.block_average_to_boxes`.
2. **Predictors available at issue time.** For every (init_date, lead_day, box): moisture
   flux, shear, 500 hPa height and 2 m temperature are noisy views of the truth whose
   correlation with it decays with lead time.
3. **Forecast error.** ``eps ~ scale * t(4)/sqrt(2)`` with
   ``scale = (1.2 + 0.3*obs) * (0.45 + 0.13*lead) * exp(0.55*u + 0.22*mfz + 0.12*shz)``,
   where ``u`` is a hidden predictability state (smooth, persistent in time), ``mfz`` the
   standardised moisture flux and ``shz`` the standardised shear. So errors grow with lead
   time, with rain amount, and with moisture flux / shear: this is the structure a bust model
   can pick up *by construction*.
4. **Ensemble.** Centre ``c = max(0, obs + eps)``; members ``max(0, c + s*z_i)`` with
   ``s`` driven by the same hidden state ``u`` (plus noise), so spread is an informative but
   imperfect warning signal. ``fcst_rain_mean``/``fcst_rain_spread`` are the sample mean/std
   of the members.

Each year is generated from ``default_rng([seed, year])``, so a year's data is identical
regardless of which other years are requested.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from akashnetra.config import AppConfig
from akashnetra.ingest.base import (
    FORECAST_COLUMNS,
    OBSERVATION_COLUMNS,
    ForecastSource,
    ObservationSource,
    SourceError,
)
from akashnetra.processing.boxes import BoxGrid
from akashnetra.processing.regrid import block_average_to_boxes, cell_centres

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticParams:
    """All tunable constants of the synthetic generator."""

    n_members: int = 11
    fine_res_deg: float = 0.25
    seed: int = 42
    # Truth field
    latent_persistence: float = 0.7  # day-to-day AR(1) coefficient of the latent field
    latent_sigma_cells: float = 6.0  # spatial smoothing [fine cells] (~1.5 deg)
    peak_doy: float = 201.0  # 20 July
    season_width_days: float = 32.0
    clim_peak_mm: float = 6.0  # peak-season mean daily rain (above a 0.8 mm base)
    # Hidden predictability state
    uncertainty_persistence: float = 0.85
    box_sigma: float = 1.0  # spatial smoothing of box-level fields [boxes]
    # Predictor skill decays as exp(-lead / tau)
    predictor_tau_days: float = 6.0
    # Error model
    err_base_mm: float = 1.2
    err_per_obs_mm: float = 0.30
    err_lead_base: float = 0.45
    err_lead_slope: float = 0.13
    err_beta_u: float = 0.55
    err_beta_moisture: float = 0.22
    err_beta_shear: float = 0.12
    err_t_df: int = 4
    err_scale_cap_mm: float = 30.0  # keeps tails physically plausible
    err_z_clip: float = 5.0  # clip of the standardised heavy-tailed draw
    # Spread model
    spread_cap_mm: float = 40.0
    spread_base_mm: float = 0.6
    spread_per_rain: float = 0.2
    spread_lead_base: float = 0.5
    spread_lead_slope: float = 0.09
    spread_beta_u: float = 0.5
    spread_beta_moisture: float = 0.1
    spread_noise: float = 0.35


def _rain_normaliser() -> float:
    """E[relu(Z + 0.2)**1.5] for Z ~ N(0, 1), so mean rain equals the climatology."""
    z = np.random.default_rng(0).standard_normal(1_000_000)
    return float(np.mean(np.maximum(z + 0.2, 0.0) ** 1.5))


_RAIN_NORM = _rain_normaliser()


def _axis_norm(n: int, sigma: float) -> float:
    """Std of white noise after 1-D periodic Gaussian smoothing on an axis of length n."""
    impulse = np.zeros(n)
    impulse[0] = 1.0
    kernel = gaussian_filter1d(impulse, sigma, mode="wrap")
    return float(np.sqrt(np.sum(kernel**2)))


def smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma: float) -> np.ndarray:
    """Spatially smooth, unit-variance Gaussian noise over the last two axes (periodic)."""
    white = rng.standard_normal(shape)
    sig = (0.0,) * (len(shape) - 2) + (sigma, sigma)
    out = gaussian_filter(white, sigma=sig, mode="wrap")
    return out / (_axis_norm(shape[-2], sigma) * _axis_norm(shape[-1], sigma))


def ar1_smooth(
    rng: np.random.Generator, n: int, shape_yx: tuple[int, int], phi: float, sigma: float
) -> np.ndarray:
    """Unit-variance field ``(n, ny, nx)`` that is AR(1) in time and smooth in space."""
    eps = smooth_noise(rng, (n, *shape_yx), sigma)
    out = np.empty_like(eps)
    out[0] = eps[0]
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + scale * eps[t]
    return out


class SyntheticSource(ForecastSource, ObservationSource):
    """Synthetic forecasts and matching observations (DEMO MODE)."""

    name = "synthetic"
    data_mode = "synthetic"

    def __init__(self, cfg: AppConfig, params: SyntheticParams | None = None) -> None:
        self._cfg = cfg
        self._grid = BoxGrid.from_config(cfg.region)
        self._params = params or SyntheticParams(
            n_members=cfg.synthetic.n_members,
            fine_res_deg=cfg.synthetic.fine_res_deg,
            seed=cfg.model.seed,
        )
        self._cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    @property
    def params(self) -> SyntheticParams:
        """Generator parameters."""
        return self._params

    def describe(self) -> dict[str, Any]:
        """Provenance description."""
        return {
            "name": self.name,
            "data_mode": self.data_mode,
            "label": "SYNTHETIC DEMO DATA",
            "warning": "Randomly generated; not real weather and not real forecast skill.",
            "n_members": self._params.n_members,
            "seed": self._params.seed,
            "params": self._params.__dict__,
        }

    # ------------------------------------------------------------------ API
    def load_forecasts(self, years: Sequence[int]) -> pd.DataFrame:
        """Forecast table for all in-season initialisation dates of ``years``."""
        frames = [self._year(y)[0] for y in years]
        return pd.concat(frames, ignore_index=True)[list(FORECAST_COLUMNS)]

    def load_observations(self, years: Sequence[int]) -> pd.DataFrame:
        """Observation table for all valid dates needed by forecasts of ``years``."""
        frames = [self._year(y)[1] for y in years]
        obs = pd.concat(frames, ignore_index=True)[list(OBSERVATION_COLUMNS)]
        return obs.drop_duplicates(["valid_date", "box_id"], ignore_index=True)

    # ------------------------------------------------------------- internals
    def _year(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if year not in self._cache:
            logger.info("Generating SYNTHETIC data for %d", year)
            self._cache[year] = self._generate_year(year)
        return self._cache[year]

    def _generate_year(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        p, grid, cfg = self._params, self._grid, self._cfg
        months, max_lead = cfg.season.months, cfg.forecast.max_lead_day
        rng = np.random.default_rng([p.seed, year])

        first = pd.Timestamp(year=year, month=min(months), day=1)
        last = pd.Timestamp(year=year, month=max(months), day=1) + pd.offsets.MonthEnd(0)
        end = last + pd.Timedelta(days=max_lead)
        if end.year != year:
            raise SourceError(
                "Seasons whose forecasts verify in the next calendar year are not supported "
                "by the synthetic source in Phase 1."
            )
        days = pd.date_range(first, end)
        in_season = days.month.isin(months) & (days <= last)
        init_idx = np.flatnonzero(in_season)
        leads = np.arange(1, max_lead + 1)
        n_days, n_i, n_d = len(days), len(init_idx), len(leads)
        ny, nx = grid.n_lat, grid.n_lon

        doy = days.dayofyear.to_numpy().astype(float)
        seas = np.exp(-0.5 * ((doy - p.peak_doy) / p.season_width_days) ** 2)

        # 1. Truth on the fine grid, then averaged to boxes -------------------------
        lat_f = cell_centres(grid.lat_min, grid.lat_max, p.fine_res_deg)
        lon_f = cell_centres(grid.lon_min, grid.lon_max, p.fine_res_deg)
        latent = ar1_smooth(
            rng, n_days, (len(lat_f), len(lon_f)), p.latent_persistence, p.latent_sigma_cells
        )
        fx = (lon_f - grid.lon_min) / (grid.lon_max - grid.lon_min)
        fy = (lat_f - grid.lat_min) / (grid.lat_max - grid.lat_min)
        spatial = 0.7 + 0.5 * fx[None, :] + 0.4 * np.exp(-(((fy[:, None] - 0.5) / 0.3) ** 2))
        clim = (0.8 + p.clim_peak_mm * seas)[:, None, None] * spatial[None]
        rain_fine = clim * np.maximum(latent + 0.2, 0.0) ** 1.5 / _RAIN_NORM
        obs_box = block_average_to_boxes(rain_fine, lat_f, lon_f, grid, min_valid_fraction=1.0)
        latent_box = block_average_to_boxes(latent, lat_f, lon_f, grid, min_valid_fraction=1.0)

        # 2. Predictors at issue time -------------------------------------------------
        vidx = init_idx[:, None] + leads[None, :]  # valid-day index, (n_i, n_d)
        shape4 = (n_i, n_d, ny, nx)
        obs_v = obs_box[vidx]
        latent_v = latent_box[vidx]
        seas_v = seas[vidx][..., None, None]
        rho = np.exp(-leads / p.predictor_tau_days)[None, :, None, None]

        mfz = rho * latent_v + np.sqrt(1 - rho**2) * smooth_noise(rng, shape4, p.box_sigma)
        shz = smooth_noise(rng, shape4, p.box_sigma)
        zz = smooth_noise(rng, shape4, p.box_sigma)
        tn = smooth_noise(rng, shape4, p.box_sigma)
        lat_c = grid.lat_centres[None, None, :, None]

        moisture_flux = np.clip(55.0 + 60.0 * seas_v + 26.0 * mfz, 5.0, None)
        shear = np.clip(24.0 + 6.0 * shz - 3.0 * seas_v, 2.0, None)
        z500 = 5885.0 - 12.0 * seas_v - 1.2 * (lat_c - 20.0) + 8.0 * zz - 3.0 * rho * latent_v
        t2m = 306.0 - 5.0 * seas_v - 0.1 * (lat_c - 20.0) - 1.5 * mfz + 0.8 * tn

        # 3. Hidden predictability state and forecast error ---------------------------
        u_t = ar1_smooth(rng, n_i, (ny, nx), p.uncertainty_persistence, p.box_sigma)
        u_n = smooth_noise(rng, shape4, p.box_sigma)
        u = np.sqrt(0.7) * u_t[:, None] + np.sqrt(0.3) * u_n

        lead_arr = leads[None, :, None, None].astype(float)
        scale = (
            (p.err_base_mm + p.err_per_obs_mm * obs_v)
            * (p.err_lead_base + p.err_lead_slope * lead_arr)
            * np.exp(p.err_beta_u * u + p.err_beta_moisture * mfz + p.err_beta_shear * shz)
        )
        scale = np.minimum(scale, p.err_scale_cap_mm)
        z_t = rng.standard_t(p.err_t_df, size=shape4) / np.sqrt(p.err_t_df / (p.err_t_df - 2))
        eps = scale * np.clip(z_t, -p.err_z_clip, p.err_z_clip)
        centre = np.maximum(obs_v + eps, 0.0)

        # 4. Ensemble ------------------------------------------------------------------
        spread_scale = (
            (p.spread_base_mm + p.spread_per_rain * centre)
            * (p.spread_lead_base + p.spread_lead_slope * lead_arr)
            * np.exp(
                p.spread_beta_u * u
                + p.spread_beta_moisture * mfz
                + p.spread_noise * rng.standard_normal(shape4)
            )
        )
        spread_scale = np.minimum(spread_scale, p.spread_cap_mm)
        noise_m = rng.standard_normal(shape4 + (p.n_members,))
        members = np.maximum(
            centre[..., None] + spread_scale[..., None] * noise_m,
            0.0,
        )
        fcst_mean = members.mean(axis=-1)
        fcst_spread = members.std(axis=-1, ddof=1)

        # Tables ---------------------------------------------------------------------
        ii, dd, yy, xx = (a.ravel() for a in np.meshgrid(*map(np.arange, shape4), indexing="ij"))
        box_ids = np.asarray(grid.box_ids)[yy * nx + xx]
        init_dates = days.to_numpy()[init_idx][ii]
        valid_dates = days.to_numpy()[vidx[ii, dd]]
        fc = pd.DataFrame(
            {
                "init_date": init_dates,
                "lead_day": leads[dd].astype(np.int8),
                "box_id": box_ids,
                "valid_date": valid_dates,
                "fcst_rain_mean": fcst_mean.ravel().astype(np.float32),
                "fcst_rain_spread": fcst_spread.ravel().astype(np.float32),
                "n_members": np.full(len(ii), p.n_members, dtype=np.int16),
                "moisture_flux_850": moisture_flux.ravel().astype(np.float32),
                "wind_shear_200_850": shear.ravel().astype(np.float32),
                "z500": z500.ravel().astype(np.float32),
                "t2m": t2m.ravel().astype(np.float32),
            }
        )

        needed = np.unique(vidx)
        mesh = np.meshgrid(needed, np.arange(ny), np.arange(nx), indexing="ij")
        oi, oy, ox = (a.ravel() for a in mesh)
        obs = pd.DataFrame(
            {
                "valid_date": days.to_numpy()[oi],
                "box_id": np.asarray(grid.box_ids)[oy * nx + ox],
                "obs_rain": obs_box[oi, oy, ox].astype(np.float32),
            }
        )
        return fc, obs
