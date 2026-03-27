"""
XG3 Speedway GP — Admin endpoints.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/speedway/admin", tags=["admin"])


@router.get("/status")
async def admin_status(request: Request) -> dict:
    """Return model and service status."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    elo_count = len(predictor._elo.snapshot())
    history_count = len(predictor._rider_history)

    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "schema_version": predictor.schema_version,
        "elo_riders_tracked": elo_count,
        "rider_histories_tracked": history_count,
    }


@router.get("/elo-ratings")
async def elo_ratings(request: Request) -> dict:
    """Return all ELO ratings."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    snap = predictor._elo.snapshot()
    sorted_ratings = sorted(snap.items(), key=lambda x: x[1], reverse=True)
    return {
        "status": "ok",
        "count": len(sorted_ratings),
        "ratings": [
            {"slug": s, "elo": round(r, 2)}
            for s, r in sorted_ratings
        ],
    }
