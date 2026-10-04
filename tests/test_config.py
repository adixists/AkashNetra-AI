"""Tests for config loading/validation, logging and run-info helpers (M0)."""

from __future__ import annotations

import copy
import io
import logging
from pathlib import Path

import pytest
import yaml

from akashnetra.config import (
    AppConfig,
    ConfigError,
    find_project_root,
    load_config,
    secret_status,
)
from akashnetra.logging_setup import log_run_header, setup_logging
from akashnetra.models.spatial_cnn import SpatialCNNBustModel
from akashnetra.runinfo import get_git_hash, library_versions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture()
def cfg(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.delenv("AKASHNETRA_DATA_MODE", raising=False)
    monkeypatch.delenv("AKASHNETRA_API_URL", raising=False)
    return load_config(ROOT / "config.yaml")


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads(cfg: AppConfig) -> None:
    assert cfg.data_mode == "synthetic"  # no real data until M6
    assert cfg.season.months == [6, 7, 8, 9]
    assert cfg.bust.percentile == 90
    assert cfg.forecast.lead_days == list(range(1, 11))
    assert cfg.features.analogs.k == 10
    assert cfg.alerts.precision_target == 0.5
    assert cfg.root == find_project_root()


def test_default_region_is_central_india_140_boxes(cfg: AppConfig) -> None:
    r = cfg.region
    assert (r.lat_min, r.lat_max, r.lon_min, r.lon_max) == (15.0, 25.0, 74.0, 88.0)
    assert (r.n_lat, r.n_lon, r.n_boxes) == (10, 14, 140)


def test_split_defaults_match_spec(cfg: AppConfig) -> None:
    full = cfg.years.split.full
    assert full.train.years == list(range(2000, 2015))
    assert full.val.years == [2015, 2016]
    assert full.test.years == [2017, 2018, 2019]
    assert cfg.years.dev_years.years == list(range(2015, 2020))
    assert cfg.years.active == "dev"


@pytest.mark.parametrize("mode", ["dev", "full"])
def test_splits_are_disjoint_and_chronological(cfg: AppConfig, mode: str) -> None:
    split = getattr(cfg.years.split, mode)
    train, val, test = (set(split.years_of(n)) for n in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert max(train) < min(val) and max(val) < min(test)


def test_split_of(cfg: AppConfig) -> None:
    split = cfg.years.split.full
    assert split.split_of(2010) == "train"
    assert split.split_of(2016) == "val"
    assert split.split_of(2019) == "test"
    assert split.split_of(1999) is None


def test_overlapping_split_is_rejected(raw_config: dict, tmp_path: Path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["years"]["split"]["full"]["val"] = {"start": 2014, "end": 2016}  # overlaps train
    with pytest.raises(ConfigError, match="chronological"):
        load_config(_write(tmp_path, raw))


def test_split_outside_year_range_is_rejected(raw_config: dict, tmp_path: Path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["years"]["split"]["dev"]["test"] = {"start": 2019, "end": 2021}
    with pytest.raises(ConfigError, match="outside"):
        load_config(_write(tmp_path, raw))


def test_region_not_multiple_of_box_is_rejected(raw_config: dict, tmp_path: Path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["region"]["lat_max"] = 24.5
    with pytest.raises(ConfigError, match="multiple"):
        load_config(_write(tmp_path, raw))


def test_unknown_key_is_rejected(raw_config: dict, tmp_path: Path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["bust"]["percentille"] = 95  # typo must not be silently ignored
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, raw))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_data_mode_env_override_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHNETRA_DATA_MODE", "real")
    assert load_config(ROOT / "config.yaml").data_mode == "real"
    monkeypatch.setenv("AKASHNETRA_DATA_MODE", "auto")  # no implicit fallback mode exists
    with pytest.raises(ConfigError):
        load_config(ROOT / "config.yaml")


def test_paths_resolve_under_root(cfg: AppConfig) -> None:
    assert cfg.data_dir == (cfg.root / "data").resolve()
    assert cfg.artifacts_dir == (cfg.root / "artifacts").resolve()


def test_run_header_logs_config_but_never_secrets(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CDS_API_KEY", "super-secret-key-123")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test_header")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_run_header(logger, cfg, title="test")
    finally:
        logger.removeHandler(handler)
    text = stream.getvalue()
    assert "super-secret-key-123" not in text
    assert "SYNTHETIC DEMO DATA" in text
    assert "data_mode=synthetic" in text
    assert secret_status()["CDS_API_KEY"] is True


def test_setup_logging_is_idempotent() -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    setup_logging("INFO")
    n1 = len(root.handlers)
    setup_logging("INFO")
    assert len(root.handlers) == n1
    assert n1 >= before


def test_runinfo_helpers() -> None:
    assert isinstance(get_git_hash(ROOT), str)
    versions = library_versions()
    assert "numpy" in versions and versions["numpy"] != "not installed"


def test_spatial_cnn_is_a_documented_stub() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        SpatialCNNBustModel()
