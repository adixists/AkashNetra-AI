"""Regridding of fine-resolution fields (e.g. IMD 0.25°) onto the 1° boxes."""

from __future__ import annotations

import numpy as np

from akashnetra.processing.boxes import BoxGrid


def cell_centres(lo: float, hi: float, res: float) -> np.ndarray:
    """Centres of regular cells of size ``res`` covering ``[lo, hi)``."""
    n = round((hi - lo) / res)
    return lo + (np.arange(n) + 0.5) * res


def _membership(coords: np.ndarray, lo: float, size: float, n_boxes: int) -> np.ndarray:
    """0/1 matrix ``(n_boxes, len(coords))``; entry 1 if the coordinate is inside the box."""
    idx = np.floor((np.asarray(coords, dtype=float) - lo) / size + 1e-9).astype(int)
    valid = (idx >= 0) & (idx < n_boxes)
    mat = np.zeros((n_boxes, len(idx)), dtype=float)
    mat[idx[valid], np.flatnonzero(valid)] = 1.0
    return mat


def block_average_to_boxes(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    grid: BoxGrid,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """Average fine-grid values within each box, ignoring NaN cells.

    Args:
        values: Array ``(..., n_lat_fine, n_lon_fine)``; leading dims (e.g. time) are kept.
        lats: Latitudes of the fine cell centres, length ``n_lat_fine``.
        lons: Longitudes of the fine cell centres, length ``n_lon_fine``.
        grid: Target box grid. Cells outside the grid are ignored.
        min_valid_fraction: A box becomes NaN if fewer than this fraction of its fine cells
            are valid (non-NaN). Boxes with no fine cells at all are always NaN.

    Returns:
        Array ``(..., grid.n_lat, grid.n_lon)``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (len(lats), len(lons)):
        raise ValueError(
            f"values trailing shape {values.shape[-2:]} does not match "
            f"(len(lats), len(lons)) = ({len(lats)}, {len(lons)})"
        )
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")

    w_lat = _membership(lats, grid.lat_min, grid.size, grid.n_lat)
    w_lon = _membership(lons, grid.lon_min, grid.size, grid.n_lon)

    finite = np.isfinite(values)
    total = w_lat @ np.where(finite, values, 0.0) @ w_lon.T
    count = w_lat @ finite.astype(float) @ w_lon.T
    n_cells = (w_lat.sum(axis=1)[:, None]) * (w_lon.sum(axis=1)[None, :])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        frac = count / n_cells
    ok = (count > 0) & (frac >= min_valid_fraction) & (n_cells > 0)
    return np.where(ok, mean, np.nan)
