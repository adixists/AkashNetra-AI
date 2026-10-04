"""Tests for the box grid and regridding."""

from __future__ import annotations

import numpy as np
import pytest

from akashnetra.config import load_config
from akashnetra.processing.boxes import BoxGrid, format_box_id
from akashnetra.processing.regrid import block_average_to_boxes, cell_centres


@pytest.fixture()
def grid() -> BoxGrid:
    return BoxGrid.from_config(load_config().region)


def test_default_grid_has_140_unique_boxes(grid: BoxGrid) -> None:
    ids = grid.box_ids
    assert len(ids) == 140 == grid.n_boxes
    assert len(set(ids)) == 140
    assert ids[0] == "B15N_74E" and ids[-1] == "B24N_87E"


def test_table_geometry(grid: BoxGrid) -> None:
    t = grid.table()
    assert len(t) == 140
    first = t.iloc[0]
    assert (first.lat, first.lon, first.lat_lo, first.lat_hi) == (15.5, 74.5, 15.0, 16.0)
    assert (t.lat_hi - t.lat_lo == 1.0).all() and (t.lon_hi - t.lon_lo == 1.0).all()
    assert t.lat.between(15, 25).all() and t.lon.between(74, 88).all()


def test_locate_edges(grid: BoxGrid) -> None:
    assert grid.locate(15.0, 74.0) == (0, 0)
    assert grid.locate(24.99, 87.99) == (9, 13)
    assert grid.locate(16.0, 75.0) == (1, 1)  # lower edges belong to the upper box
    assert grid.locate(25.0, 80.0) is None  # upper edge is outside
    assert grid.locate(14.99, 80.0) is None
    assert grid.locate(20.0, 88.0) is None
    assert grid.box_id_at(20.5, 80.5) == "B20N_80E"
    assert grid.box_id_at(0.0, 0.0) is None


def test_fractional_and_negative_coordinates() -> None:
    g = BoxGrid(lat_min=-1.0, lat_max=1.0, lon_min=-1.0, lon_max=0.0, size=0.5)
    assert g.n_boxes == 8
    assert format_box_id(-1.0, -1.0) == "BS1_W1".replace("BS1_W1", "B1S_1W")
    assert "B0.5N_1W" in g.box_ids and "B1S_0.5W" in g.box_ids


def test_block_average_known_values() -> None:
    grid = BoxGrid(lat_min=0, lat_max=2, lon_min=0, lon_max=2, size=1.0)
    lats = cell_centres(0, 2, 0.5)
    lons = cell_centres(0, 2, 0.5)
    values = np.arange(16, dtype=float).reshape(4, 4)
    out = block_average_to_boxes(values, lats, lons, grid)
    expected = np.array(
        [
            [values[0:2, 0:2].mean(), values[0:2, 2:4].mean()],
            [values[2:4, 0:2].mean(), values[2:4, 2:4].mean()],
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_block_average_nan_handling_and_leading_dims() -> None:
    grid = BoxGrid(lat_min=0, lat_max=1, lon_min=0, lon_max=1, size=1.0)
    c = cell_centres(0, 1, 0.5)
    v = np.array([[1.0, np.nan], [3.0, 5.0]])
    assert block_average_to_boxes(v, c, c, grid, min_valid_fraction=0.5)[0, 0] == 3.0
    assert np.isnan(block_average_to_boxes(v, c, c, grid, min_valid_fraction=1.0)[0, 0])
    stack = np.stack([v, v * 2])
    out = block_average_to_boxes(stack, c, c, grid, min_valid_fraction=0.5)
    assert out.shape == (2, 1, 1) and out[1, 0, 0] == 6.0


def test_block_average_ignores_cells_outside_grid_and_validates_shape() -> None:
    grid = BoxGrid(lat_min=0, lat_max=1, lon_min=0, lon_max=1, size=1.0)
    lats = np.array([0.25, 0.75, 1.25])  # last row lies outside the grid
    lons = np.array([0.25, 0.75])
    values = np.array([[1.0, 1.0], [3.0, 3.0], [1000.0, 1000.0]])
    assert block_average_to_boxes(values, lats, lons, grid)[0, 0] == 2.0
    with pytest.raises(ValueError, match="does not match"):
        block_average_to_boxes(values, lats[:2], lons, grid)
