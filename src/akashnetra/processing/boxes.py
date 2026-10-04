"""The 1°×1° box grid over the configured region.

Boxes tile the region exactly (edges are multiples of ``box_size_deg``). A box is
identified by its south-west corner, e.g. ``B15N_74E``. Latitude/longitude intervals are
closed at the lower edge and open at the upper edge, so every point belongs to at most
one box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akashnetra.config import RegionConfig


def format_box_id(lat_lo: float, lon_lo: float) -> str:
    """Build a box id from the south-west corner, e.g. ``B15N_74E``."""

    def _fmt(value: float, pos: str, neg: str) -> str:
        return f"{abs(round(value, 6)):g}{pos if value >= 0 else neg}"

    return f"B{_fmt(lat_lo, 'N', 'S')}_{_fmt(lon_lo, 'E', 'W')}"


@dataclass(frozen=True)
class BoxGrid:
    """Rectangular grid of square boxes, indexed row-major (south to north, west to east)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    size: float

    @classmethod
    def from_config(cls, region: RegionConfig) -> BoxGrid:
        """Create the grid from the validated region config."""
        return cls(
            lat_min=region.lat_min,
            lat_max=region.lat_max,
            lon_min=region.lon_min,
            lon_max=region.lon_max,
            size=region.box_size_deg,
        )

    @property
    def n_lat(self) -> int:
        """Number of box rows."""
        return round((self.lat_max - self.lat_min) / self.size)

    @property
    def n_lon(self) -> int:
        """Number of box columns."""
        return round((self.lon_max - self.lon_min) / self.size)

    @property
    def n_boxes(self) -> int:
        """Total number of boxes."""
        return self.n_lat * self.n_lon

    @property
    def lat_centres(self) -> np.ndarray:
        """Latitude of box-row centres, shape ``(n_lat,)``."""
        return self.lat_min + (np.arange(self.n_lat) + 0.5) * self.size

    @property
    def lon_centres(self) -> np.ndarray:
        """Longitude of box-column centres, shape ``(n_lon,)``."""
        return self.lon_min + (np.arange(self.n_lon) + 0.5) * self.size

    @property
    def box_ids(self) -> list[str]:
        """All box ids in row-major order."""
        return [
            format_box_id(self.lat_min + r * self.size, self.lon_min + c * self.size)
            for r in range(self.n_lat)
            for c in range(self.n_lon)
        ]

    def table(self) -> pd.DataFrame:
        """One row per box with ids, bounds, centres and grid indices (row-major)."""
        rows, cols = np.divmod(np.arange(self.n_boxes), self.n_lon)
        lat_lo = self.lat_min + rows * self.size
        lon_lo = self.lon_min + cols * self.size
        return pd.DataFrame(
            {
                "box_id": self.box_ids,
                "row": rows,
                "col": cols,
                "lat_lo": lat_lo,
                "lat_hi": lat_lo + self.size,
                "lon_lo": lon_lo,
                "lon_hi": lon_lo + self.size,
                "lat": lat_lo + self.size / 2,
                "lon": lon_lo + self.size / 2,
            }
        )

    def locate(self, lat: float, lon: float) -> tuple[int, int] | None:
        """Return ``(row, col)`` of the box containing the point, or ``None`` if outside."""
        if not (self.lat_min <= lat < self.lat_max and self.lon_min <= lon < self.lon_max):
            return None
        row = int(np.floor((lat - self.lat_min) / self.size + 1e-9))
        col = int(np.floor((lon - self.lon_min) / self.size + 1e-9))
        return min(row, self.n_lat - 1), min(col, self.n_lon - 1)

    def box_id_at(self, lat: float, lon: float) -> str | None:
        """Return the id of the box containing the point, or ``None`` if outside."""
        loc = self.locate(lat, lon)
        if loc is None:
            return None
        row, col = loc
        return format_box_id(self.lat_min + row * self.size, self.lon_min + col * self.size)
