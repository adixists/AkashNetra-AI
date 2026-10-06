"""FastAPI application for AkashNetra AI."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from akashnetra.api.schemas import (
    AlertSummaryResponse,
    BoxDetailResponse,
    GeoJSONFeatureCollection,
    HealthResponse,
    InitDatesResponse,
    MetricsResponse,
)
from akashnetra.api.services import ForecastService
from akashnetra.config import load_config

logger = logging.getLogger(__name__)

cfg = load_config()

app = FastAPI(
    title="AkashNetra AI API",
    description="Forecast Bust Predictor API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service globally
service: ForecastService | None = None


@app.on_event("startup")
def startup_event() -> None:
    global service
    service = ForecastService(cfg)
    logger.info("API initialized with data_mode=%s", service.data_mode)

@app.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, str]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {
        "status": "ok",
        "data_mode": service.data_mode,
        "model_version": service.model_version
    }

@app.get("/init-dates", response_model=InitDatesResponse)
def get_init_dates() -> dict[str, list[str]]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {"dates": service.get_init_dates()}

@app.get("/alerts", response_model=GeoJSONFeatureCollection)
def get_alerts(
    init_date: str = Query(..., description="YYYY-MM-DD"),
    lead_day: int = Query(..., ge=1, le=10),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
) -> dict[str, Any]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    features = service.get_alerts(init_date, lead_day, threshold)
    return {
        "type": "FeatureCollection",
        "features": features,
        "data_mode": service.data_mode,
        "model_version": service.model_version
    }

@app.get("/alerts/summary", response_model=AlertSummaryResponse)
def get_alerts_summary(
    init_date: str = Query(..., description="YYYY-MM-DD"),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
) -> dict[str, Any]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    tiles = service.get_summary(init_date, threshold)
    return {
        "init_date": init_date,
        "tiles": tiles,
        "data_mode": service.data_mode,
        "model_version": service.model_version
    }

@app.get("/box/{box_id}", response_model=BoxDetailResponse)
def get_box_detail(
    box_id: str,
    init_date: str = Query(...),
    lead_day: int = Query(..., ge=1, le=10),
    threshold: float = Query(0.5)
) -> dict[str, Any]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return service.get_box_detail(box_id, init_date, lead_day, threshold)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> dict[str, Any]:
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "metrics": service.get_metrics(),
        "data_mode": service.data_mode,
        "model_version": service.model_version
    }
