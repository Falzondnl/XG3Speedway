"""
XG3 Speedway GP — Race prediction and pricing endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pricing.markets import (
    price_h2h,
    price_heat_winner,
    price_round_winner,
    price_top3_finish,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/speedway", tags=["speedway"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RiderInput(BaseModel):
    slug: str = Field(..., description="Rider identifier slug (e.g. 'bartosz-zmarzlik')")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country_code: Optional[str] = None
    is_wildcard: bool = False
    is_substitute: bool = False
    sgp_wins: Optional[int] = None
    appearances: Optional[int] = None
    heats_won: Optional[int] = None
    heats_raced: Optional[int] = None
    finals: Optional[int] = None
    wins: Optional[int] = None
    fim_rank: Optional[int] = None
    born_at: Optional[str] = None
    round_date: Optional[str] = None


class PriceRoundRequest(BaseModel):
    riders: list[RiderInput] = Field(..., min_length=2, max_length=24,
                                     description="List of riders in this GP round (2-24)")
    venue_slug: Optional[str] = None
    venue_country: Optional[str] = None
    track_length: Optional[float] = None
    season_id: int = 0
    win_margin: float = Field(0.05, ge=0.0, le=0.30)
    top3_margin: float = Field(0.05, ge=0.0, le=0.30)


class HeatRiderInput(BaseModel):
    slug: str
    gate_position: Optional[int] = None


class PriceHeatRequest(BaseModel):
    heat_riders: list[HeatRiderInput] = Field(..., min_length=2, max_length=6)
    all_riders: list[RiderInput] = Field(..., min_length=2,
                                         description="All round riders (for full probability context)")
    venue_slug: Optional[str] = None
    venue_country: Optional[str] = None
    track_length: Optional[float] = None
    season_id: int = 0
    margin: float = Field(0.05, ge=0.0, le=0.30)


class H2HRequest(BaseModel):
    rider_a_slug: str
    rider_b_slug: str
    all_riders: list[RiderInput] = Field(..., min_length=2)
    venue_slug: Optional[str] = None
    venue_country: Optional[str] = None
    track_length: Optional[float] = None
    season_id: int = 0
    margin: float = Field(0.05, ge=0.0, le=0.30)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/price-round")
async def price_round(body: PriceRoundRequest, request: Request) -> dict:
    """
    Price a Speedway GP round.

    Returns:
    - round_winner: decimal odds + win_prob per rider
    - top3_finish:  decimal odds + p_1st/p_2nd/p_3rd per rider
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    riders_dicts = [r.model_dump() for r in body.riders]

    try:
        rider_probs = predictor.predict_round(
            riders=riders_dicts,
            venue_slug=body.venue_slug,
            venue_country=body.venue_country,
            track_length=body.track_length,
            season_id=body.season_id,
        )
    except Exception as e:
        logger.exception("predict_round failed")
        raise HTTPException(500, detail=f"Prediction error: {e}")

    round_winner = price_round_winner(rider_probs, margin=body.win_margin)
    top3 = price_top3_finish(rider_probs, margin=body.top3_margin)

    return {
        "status": "ok",
        "venue_slug": body.venue_slug,
        "venue_country": body.venue_country,
        "model_schema": predictor.schema_version,
        "round_winner": round_winner,
        "top3_finish": top3,
        "n_riders": len(rider_probs),
    }


@router.post("/price-heat")
async def price_heat(body: PriceHeatRequest, request: Request) -> dict:
    """Price winner of a specific heat from 2-6 riders."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    all_dicts = [r.model_dump() for r in body.all_riders]
    heat_dicts = [r.model_dump() for r in body.heat_riders]

    try:
        all_probs = predictor.predict_round(
            riders=all_dicts,
            venue_slug=body.venue_slug,
            venue_country=body.venue_country,
            track_length=body.track_length,
            season_id=body.season_id,
        )
    except Exception as e:
        logger.exception("predict_round failed for heat pricing")
        raise HTTPException(500, detail=f"Prediction error: {e}")

    heat_priced = price_heat_winner(heat_dicts, all_probs, margin=body.margin)
    if not heat_priced:
        raise HTTPException(400, detail="Heat riders not found in all_riders list")

    return {
        "status": "ok",
        "heat_winner": heat_priced,
    }


@router.post("/price-h2h")
async def price_h2h_endpoint(body: H2HRequest, request: Request) -> dict:
    """Price a head-to-head between two riders to finish higher."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    all_dicts = [r.model_dump() for r in body.all_riders]

    try:
        all_probs = predictor.predict_round(
            riders=all_dicts,
            venue_slug=body.venue_slug,
            venue_country=body.venue_country,
            track_length=body.track_length,
            season_id=body.season_id,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction error: {e}")

    result = price_h2h(
        rider_a_slug=body.rider_a_slug,
        rider_b_slug=body.rider_b_slug,
        rider_probs=all_probs,
        margin=body.margin,
    )
    if result is None:
        raise HTTPException(400, detail="One or both riders not found in rider list")

    return {"status": "ok", "h2h": result}


@router.get("/fixtures")
@router.get("/events")
async def list_fixtures() -> list:
    """
    Fixture discovery endpoint for GenericSportHub.
    Speedway GP is not available via Optic Odds — returns honest empty list.
    Fixtures are added manually via the scheduler or trading console.
    """
    return []


@router.get("/riders/top")
async def top_riders(request: Request, limit: int = 20) -> dict:
    """Return top riders by current ELO rating."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(503, detail="Predictor not initialized")

    elo_dict = predictor._elo.snapshot()
    sorted_riders = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "status": "ok",
        "riders": [
            {"slug": slug, "elo_rating": round(rating, 1)}
            for slug, rating in sorted_riders
        ],
    }
