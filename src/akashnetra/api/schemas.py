"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    data_mode: str
    model_version: str

class InitDatesResponse(BaseModel):
    dates: list[str]

class AlertSummaryTile(BaseModel):
    lead_day: int
    n_alerts: int
    max_prob: float
    mean_confidence: float

class AlertSummaryResponse(BaseModel):
    init_date: str
    tiles: list[AlertSummaryTile]
    data_mode: str
    model_version: str

# GeoJSON Models
class GeoJSONProperties(BaseModel):
    box_id: str
    bust_prob: float
    confidence: float
    alert: bool
    fcst_rain_mm: float
    spread_mm: float
    level: str

class GeoJSONGeometry(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[list[float]]]

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]
    data_mode: str
    model_version: str

# Detailed Box Model
class DriverInfo(BaseModel):
    feature: str
    feature_id: str
    feature_value: float
    shap_value: float
    raises_risk: bool

class AnalogInfo(BaseModel):
    lib_id: int
    init_date: str
    valid_date: str
    box_id: str
    lead_day: int
    similarity: float
    fcst_rain_mm: float
    obs_rain_mm: float
    abs_error_mm: float
    bust: bool

class RuleInfo(BaseModel):
    id: str
    message: str

class PhysicsCheck(BaseModel):
    passed: bool
    rules_triggered: list[RuleInfo]

class BoxDetailResponse(BaseModel):
    box_id: str
    init_date: str
    lead_day: int
    bust_prob: float
    confidence: float
    alert: bool
    drivers: list[DriverInfo]
    analogs: list[AnalogInfo]
    physics: PhysicsCheck
    reason: str
    data_mode: str
    model_version: str

class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
    data_mode: str
    model_version: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
