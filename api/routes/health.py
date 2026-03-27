"""XG3 Speedway GP — Health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    return {
        "status": "ok",
        "service": "speedway",
        "version": "1.0.0",
        "model_loaded": predictor.is_loaded if predictor else False,
    }


@router.get("/health/ready")
async def health_ready(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    model_ready = predictor is not None and predictor.is_loaded
    return {
        "status": "ok" if model_ready else "degraded",
        "ready": model_ready,
        "model_schema": predictor.schema_version if predictor else "not_loaded",
    }


@router.get("/health/live")
async def health_live() -> dict:
    return {"status": "ok"}
