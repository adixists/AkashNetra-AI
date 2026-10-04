"""Typed, validated configuration for AkashNetra AI.

The YAML file (``config.yaml``) is parsed into frozen Pydantic models. Unknown keys are
rejected so typos cannot silently change behaviour. Secrets are read from the
environment / ``.env`` and are never part of the config object or of any log output.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_FILENAME = "config.yaml"
ENV_CONFIG_PATH = "AKASHNETRA_CONFIG"
ENV_HOME = "AKASHNETRA_HOME"
ENV_DATA_MODE = "AKASHNETRA_DATA_MODE"
ENV_API_URL = "AKASHNETRA_API_URL"

DataMode = Literal["synthetic", "real"]
SplitName = Literal["train", "val", "test"]


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


class _Strict(BaseModel):
    """Base model: reject unknown keys, immutable after load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class ProjectConfig(_Strict):
    name: str
    model_version: str


class PathsConfig(_Strict):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class RegionConfig(_Strict):
    """Rectangular pilot region tiled by square boxes aligned to ``box_size_deg``."""

    lat_min: float = Field(ge=-90, le=90)
    lat_max: float = Field(ge=-90, le=90)
    lon_min: float = Field(ge=-180, le=360)
    lon_max: float = Field(ge=-180, le=360)
    box_size_deg: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_geometry(self) -> RegionConfig:
        if self.lat_min >= self.lat_max:
            raise ValueError("region.lat_min must be < region.lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("region.lon_min must be < region.lon_max")
        for name, span in (
            ("lat", self.lat_max - self.lat_min),
            ("lon", self.lon_max - self.lon_min),
        ):
            n = span / self.box_size_deg
            if abs(n - round(n)) > 1e-9:
                raise ValueError(
                    f"region {name} extent ({span} deg) must be a whole multiple of "
                    f"box_size_deg ({self.box_size_deg})"
                )
        return self

    @property
    def n_lat(self) -> int:
        """Number of box rows (south to north)."""
        return round((self.lat_max - self.lat_min) / self.box_size_deg)

    @property
    def n_lon(self) -> int:
        """Number of box columns (west to east)."""
        return round((self.lon_max - self.lon_min) / self.box_size_deg)

    @property
    def n_boxes(self) -> int:
        """Total number of boxes in the region."""
        return self.n_lat * self.n_lon


class SeasonConfig(_Strict):
    name: str
    months: list[int]
    assign_by: Literal["init_date"] = "init_date"

    @field_validator("months")
    @classmethod
    def _check_months(cls, v: list[int]) -> list[int]:
        if not v or any(m < 1 or m > 12 for m in v) or len(set(v)) != len(v):
            raise ValueError("season.months must be unique integers in 1..12")
        return sorted(v)


class YearRange(_Strict):
    """Inclusive range of calendar years."""

    start: int
    end: int

    @model_validator(mode="after")
    def _check_order(self) -> YearRange:
        if self.start > self.end:
            raise ValueError(f"year range start ({self.start}) > end ({self.end})")
        return self

    @property
    def years(self) -> list[int]:
        """All years in the range, ascending."""
        return list(range(self.start, self.end + 1))

    def contains(self, other: YearRange) -> bool:
        """True if ``other`` lies fully inside this range."""
        return self.start <= other.start and other.end <= self.end


class SplitConfig(_Strict):
    """Chronological train/validation/test split by year."""

    train: YearRange
    val: YearRange
    test: YearRange

    @model_validator(mode="after")
    def _check_chronology(self) -> SplitConfig:
        if not (self.train.end < self.val.start and self.val.end < self.test.start):
            raise ValueError(
                "split must be chronological and non-overlapping: "
                "train < val < test (leakage guard)"
            )
        return self

    def split_of(self, year: int) -> SplitName | None:
        """Return the split a year belongs to, or ``None`` if it is in no split."""
        for name in ("train", "val", "test"):
            rng: YearRange = getattr(self, name)
            if rng.start <= year <= rng.end:
                return name  # type: ignore[return-value]
        return None

    def years_of(self, name: SplitName) -> list[int]:
        """Years of one split."""
        rng: YearRange = getattr(self, name)
        return rng.years


class SplitsByMode(_Strict):
    full: SplitConfig
    dev: SplitConfig


class YearsConfig(_Strict):
    active: Literal["dev", "full"] = "dev"
    dev_years: YearRange
    full_years: YearRange
    split: SplitsByMode

    @model_validator(mode="after")
    def _check_splits_inside_ranges(self) -> YearsConfig:
        for mode, rng, spl in (
            ("dev", self.dev_years, self.split.dev),
            ("full", self.full_years, self.split.full),
        ):
            for part in ("train", "val", "test"):
                if not rng.contains(getattr(spl, part)):
                    raise ValueError(f"years.split.{mode}.{part} lies outside years.{mode}_years")
        return self

    @property
    def active_years(self) -> YearRange:
        """Year range of the active mode."""
        return self.dev_years if self.active == "dev" else self.full_years

    @property
    def active_split(self) -> SplitConfig:
        """Split of the active mode."""
        return self.split.dev if self.active == "dev" else self.split.full


class ForecastConfig(_Strict):
    max_lead_day: int = Field(default=10, ge=1, le=16)

    @property
    def lead_days(self) -> list[int]:
        """Lead days 1..max_lead_day."""
        return list(range(1, self.max_lead_day + 1))


class BustConfig(_Strict):
    percentile: float = Field(default=90.0, gt=0, lt=100)
    min_samples_per_group: int = Field(default=30, ge=2)
    per_lead_day: bool = True


class GEFSConfig(_Strict):
    bucket: str
    anonymous: bool = True
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0)
    variables: list[str]


class AnalogConfig(_Strict):
    k: int = Field(default=10, ge=1)


class FeaturesConfig(_Strict):
    analogs: AnalogConfig = AnalogConfig()


class ModelConfig(_Strict):
    seed: int = 42
    calibration: Literal["isotonic", "platt"] = "isotonic"
    tuning_trials: int = Field(default=0, ge=0)
    early_stopping_rounds: int = Field(default=50, ge=1)
    lightgbm: dict[str, Any] = Field(default_factory=dict)
    xgboost: dict[str, Any] = Field(default_factory=dict)


class AlertsConfig(_Strict):
    precision_target: float = Field(default=0.5, gt=0, le=1)


class ApiConfig(_Strict):
    host: str = "0.0.0.0"  # noqa: S104  (container-friendly default, overridable)
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)


class DashboardConfig(_Strict):
    api_url: str = "http://localhost:8000"
    port: int = Field(default=8501, ge=1, le=65535)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class AppConfig(_Strict):
    """Root configuration object."""

    project: ProjectConfig
    data_mode: DataMode
    paths: PathsConfig = PathsConfig()
    region: RegionConfig
    season: SeasonConfig
    years: YearsConfig
    forecast: ForecastConfig = ForecastConfig()
    bust: BustConfig = BustConfig()
    gefs: GEFSConfig
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    alerts: AlertsConfig = AlertsConfig()
    api: ApiConfig = ApiConfig()
    dashboard: DashboardConfig = DashboardConfig()

    # Set by load_config(); not part of the YAML schema.
    root: Path = Field(default=Path("."), exclude=True)

    @property
    def data_dir(self) -> Path:
        """Absolute data directory."""
        return (self.root / self.paths.data_dir).resolve()

    @property
    def artifacts_dir(self) -> Path:
        """Absolute artifacts directory."""
        return (self.root / self.paths.artifacts_dir).resolve()

    def to_log_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the config (contains no secrets by construction)."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root (the directory holding ``config.yaml``).

    Order: ``$AKASHNETRA_HOME``; then walk up from ``start`` (default: cwd); finally the
    repository layout relative to this file (works for editable installs).
    """
    env_home = os.environ.get(ENV_HOME)
    if env_home:
        return Path(env_home).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONFIG_FILENAME).is_file():
            return candidate
    fallback = Path(__file__).resolve().parents[2]
    if (fallback / CONFIG_FILENAME).is_file():
        return fallback
    raise ConfigError(
        f"Could not find {CONFIG_FILENAME}. Run from the project directory or set {ENV_HOME}."
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the configuration.

    Args:
        path: Explicit YAML path. Defaults to ``$AKASHNETRA_CONFIG`` or
            ``<project root>/config.yaml``.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or fails validation.
    """
    root = find_project_root()
    load_dotenv(root / ".env", override=False)

    cfg_path = Path(path or os.environ.get(ENV_CONFIG_PATH) or root / CONFIG_FILENAME)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at top level")

    # Explicit, documented environment overrides (the only ones supported).
    if os.environ.get(ENV_DATA_MODE):
        raw["data_mode"] = os.environ[ENV_DATA_MODE]
    if os.environ.get(ENV_API_URL):
        raw.setdefault("dashboard", {})["api_url"] = os.environ[ENV_API_URL]

    try:
        cfg = AppConfig.model_validate(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid configuration in {cfg_path}:\n{exc}") from exc
    return cfg.model_copy(update={"root": root})


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached default configuration (use :func:`load_config` in tests)."""
    return load_config()


# --------------------------------------------------------------------------- #
# Secrets (environment only)
# --------------------------------------------------------------------------- #
SECRET_ENV_VARS: tuple[str, ...] = ("CDS_API_URL", "CDS_API_KEY", "NCUM_DATA_DIR")


def secret_status() -> dict[str, bool]:
    """Report which secret/env variables are set, WITHOUT revealing their values."""
    return {name: bool(os.environ.get(name)) for name in SECRET_ENV_VARS}
