"""Shared fixtures. Integration tests use a small 4x4-box region so they run quickly."""

from __future__ import annotations

from pathlib import Path

import pytest

from akashnetra.config import AppConfig, RegionConfig, load_config
from akashnetra.processing.dataset import BuildResult, build_dataset

ROOT = Path(__file__).resolve().parents[1]


def make_small_cfg(root: Path, monkeypatch: pytest.MonkeyPatch | None = None) -> AppConfig:
    """Default config on a 4x4 region, writing outputs under ``root``."""
    if monkeypatch is not None:
        monkeypatch.delenv("AKASHNETRA_DATA_MODE", raising=False)
    cfg = load_config(ROOT / "config.yaml")
    region = RegionConfig(lat_min=15.0, lat_max=19.0, lon_min=74.0, lon_max=78.0)
    return cfg.model_copy(update={"region": region, "root": root})


@pytest.fixture(scope="session")
def small_build(tmp_path_factory: pytest.TempPathFactory) -> tuple[AppConfig, BuildResult]:
    """Run the whole M1 pipeline once on the small region and share the result."""
    root = tmp_path_factory.mktemp("akashnetra_small")
    cfg = make_small_cfg(root)
    return cfg, build_dataset(cfg)
