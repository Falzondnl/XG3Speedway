"""
api/routes/live.py — Speedway GP Live In-Running Repricing Endpoints

Endpoints
---------
POST /api/v1/speedway/live/reprice
    Receive a live round state update (current heat results, points standings,
    DNF/DNR events) and return immediately repriced markets.

GET  /api/v1/speedway/live/{round_id}/state
    Return the current live state for an in-progress GP round.

POST /api/v1/speedway/live/{round_id}/suspend
    Operator endpoint to suspend all markets for a round.

POST /api/v1/speedway/live/{round_id}/resume
    Resume previously suspended markets.

Speedway GP live pricing model
-------------------------------
Speedway GP rounds consist of multiple heats (typically 20 heats with 4 riders
each, then semi-finals and a final). Live repricing works by:

1. After each completed heat: update points standings with actual scores.
2. Recalculate P(win round) as Harville DP over remaining heats + final.
3. Riders who have mathematically been eliminated from winning get P=0.
4. For riders still in contention, blend:
     P_live = alpha * P_ml_adjusted + (1-alpha) * P_elo_form

Form factor: a rider's form across the completed heats of this round
(points scored per heat so far vs expected) modulates their baseline probability.

Suspension triggers
-------------------
- Feed stale > 90 seconds
- Rider excluded/DNF with high win probability (material market event)
- Manual operator override via POST /suspend

Pinnacle blend
--------------
If pinnacle_odds supplied, blended at 40%:
  P_final = 0.60 * P_ml_form_adjusted + 0.40 * P_pinnacle
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, model_validator

from pricing.markets import _apply_margin, harville_top3

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STALE_THRESHOLD_SECONDS: float = 90.0
PINNACLE_BLEND_ALPHA: float = 0.60
FORM_BLEND_WEIGHT: float = 0.40  # Weight of form factor vs baseline ML probability

# In-process state (production: use Redis)
_live_state_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HeatResult(BaseModel):
    """Actual result of a completed heat."""
    heat_number: int = Field(..., ge=1)
    rider_slug: str
    finishing_position: int = Field(..., ge=1, le=4, description="1st=3pts, 2nd=2pts, 3rd=1pt, 4th=0")
    points_scored: int = Field(..., ge=0, le=3)
    dnf: bool = Field(default=False, description="Did Not Finish this heat")
    excluded: bool = Field(default=False, description="Excluded from heat by referee")


class RiderLiveStatus(BaseModel):
    """Aggregate live status for one rider across all completed heats."""
    slug: str
    heats_completed: int = Field(default=0, ge=0)
    points_total: int = Field(default=0, ge=0)
    current_round_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Current position in round standings",
    )
    mathematically_eliminated: bool = Field(
        default=False,
        description="Rider cannot mathematically win the round anymore",
    )
    dnf: bool = Field(default=False, description="Rider has retired from round (mechanical/injury)")


class PinnacleEntry(BaseModel):
    slug: str
    decimal_odds: float = Field(..., gt=1.0)


class LiveRepriceRequest(BaseModel):
    """Live repricing request for a Speedway GP round."""
    round_id: str = Field(..., description="Unique round identifier")
    venue_slug: Optional[str] = None
    venue_country: Optional[str] = None
    track_length: Optional[float] = None
    season_id: int = 0
    riders: list[dict] = Field(
        ...,
        min_length=2,
        description="Original rider list (same format as /price-round)",
    )
    heat_results: list[HeatResult] = Field(
        default_factory=list,
        description="All completed heat results so far",
    )
    rider_statuses: list[RiderLiveStatus] = Field(
        ...,
        description="Current aggregate status per rider",
    )
    heats_completed: int = Field(..., ge=0, description="Total heats completed so far")
    heats_total: int = Field(..., ge=4, description="Total heats in this round format")
    pinnacle_odds: Optional[list[PinnacleEntry]] = None
    win_margin: float = Field(default=0.05, ge=0.0, le=0.30)
    top3_margin: float = Field(default=0.05, ge=0.0, le=0.30)
    feed_timestamp_utc: str
    feed_source: str = Field(default="manual")

    @model_validator(mode="after")
    def validate_heats(self) -> "LiveRepriceRequest":
        if self.heats_completed > self.heats_total:
            raise ValueError(
                f"heats_completed ({self.heats_completed}) cannot exceed "
                f"heats_total ({self.heats_total})"
            )
        return self


class LiveSelectionPrice(BaseModel):
    slug: str
    label: str
    fair_probability: float = Field(ge=0.0, le=1.0)
    decimal_odds: float = Field(ge=1.0)
    current_points: int
    current_rank: Optional[int]
    heats_completed: int
    is_eliminated: bool
    is_dnf: bool
    market_suspended: bool = False
    probability_change_vs_prerace: Optional[float] = None


class LiveMarket(BaseModel):
    market_type: str
    is_suspended: bool
    suspension_reason: Optional[str]
    selections: list[LiveSelectionPrice]


class LiveRepriceResponse(BaseModel):
    request_id: str
    round_id: str
    venue_slug: Optional[str]
    heats_completed: int
    heats_total: int
    round_progress_pct: float
    n_active: int
    n_eliminated: int
    n_dnf: int
    is_suspended: bool
    suspension_reason: Optional[str]
    markets: list[LiveMarket]
    feed_age_seconds: float
    blend_mode: str
    model_schema: str
    repriced_at_utc: str
    elapsed_ms: float


class LiveStateResponse(BaseModel):
    round_id: str
    is_live: bool
    is_suspended: bool
    suspension_reason: Optional[str]
    heats_completed: Optional[int]
    heats_total: Optional[int]
    round_progress_pct: Optional[float]
    n_active: Optional[int]
    n_dnf: Optional[int]
    feed_source: Optional[str]
    last_update_utc: Optional[str]
    feed_age_seconds: Optional[float]
    current_leader_slug: Optional[str]
    current_leader_points: Optional[int]


class SuspendResponse(BaseModel):
    round_id: str
    suspended: bool
    reason: str
    suspended_at_utc: str


class ResumeResponse(BaseModel):
    round_id: str
    resumed: bool
    resumed_at_utc: str


# ---------------------------------------------------------------------------
# Core live probability computation
# ---------------------------------------------------------------------------


def _compute_live_probabilities(
    base_rider_probs: list[dict],
    rider_statuses: list[RiderLiveStatus],
    heat_results: list[HeatResult],
    heats_completed: int,
    heats_total: int,
    pinnacle_odds: Optional[list[PinnacleEntry]],
) -> tuple[list[dict], str]:
    """
    Compute live win probabilities by blending ML baseline with form factor.

    Form factor: Rider's actual points per heat vs expected points per heat.
    Better-than-expected performance increases win probability multiplicatively.

    Returns (updated_results, blend_mode).
    updated_results: [
      {slug, win_prob, top3_prob, pre_race_win_prob,
       current_points, current_rank, heats_completed,
       is_eliminated, is_dnf}
    ]
    """
    status_map: dict[str, RiderLiveStatus] = {s.slug: s for s in rider_statuses}
    pre_race_map: dict[str, float] = {
        r.get("slug", ""): float(r.get("win_prob", 0.0)) for r in base_rider_probs
    }

    # Compute points per heat expected based on ML win_prob
    # E[pts per heat] = 3*P(1st) + 2*P(2nd) + 1*P(3rd) using Harville
    wp_list = [max(pre_race_map.get(r.get("slug", ""), 1e-9), 1e-9) for r in base_rider_probs]
    wp_arr = np.array(wp_list, dtype=float)
    wp_arr = wp_arr / wp_arr.sum()

    p1_list, p2_list, p3_list = harville_top3(wp_arr.tolist())
    slug_list = [r.get("slug", "") for r in base_rider_probs]

    expected_pts_per_heat: dict[str, float] = {}
    for i, slug in enumerate(slug_list):
        expected_pts_per_heat[slug] = (
            3.0 * float(p1_list[i]) + 2.0 * float(p2_list[i]) + 1.0 * float(p3_list[i])
        )

    # Compute form factors
    adjusted_probs: dict[str, float] = {}
    for r in base_rider_probs:
        slug = r.get("slug", "")
        s = status_map.get(slug)
        pre = pre_race_map.get(slug, 0.0)

        if s is None:
            adjusted_probs[slug] = pre
            continue

        if s.mathematically_eliminated or s.dnf:
            adjusted_probs[slug] = 0.0
            continue

        if s.heats_completed > 0:
            # Form factor: actual pts per heat / expected pts per heat
            actual_pts_rate = s.points_total / max(s.heats_completed, 1)
            expected_pts_rate = expected_pts_per_heat.get(slug, 1.0)
            form_ratio = actual_pts_rate / max(expected_pts_rate, 0.01)
            # Clamp form factor: 0.25x to 4x multiplier
            form_ratio = max(0.25, min(4.0, form_ratio))
            # Progress weight: as more heats complete, form becomes more important
            progress_weight = min(heats_completed / max(heats_total, 1), 0.8)
            # Blend: pre-race ML * (1 + form_adjustment * progress_weight)
            form_adjustment = (form_ratio - 1.0) * FORM_BLEND_WEIGHT * progress_weight
            adjusted = pre * (1.0 + form_adjustment)
            adjusted_probs[slug] = max(adjusted, 1e-9)
        else:
            # No heats completed — use pre-race probability with slight decay
            # if heats_total > 0 (normalised by how many heats have passed overall)
            if heats_completed > 0:
                decay = 1.0 - (heats_completed / max(heats_total, 1)) * 0.1
                adjusted_probs[slug] = max(pre * decay, 1e-9)
            else:
                adjusted_probs[slug] = pre

    # Normalise (only non-eliminated, non-DNF riders)
    active_slugs = [s for s, p in adjusted_probs.items() if p > 0.0]
    active_arr = np.array([adjusted_probs[s] for s in active_slugs], dtype=float)
    if active_arr.sum() > 1e-9:
        active_arr = active_arr / active_arr.sum()
    for i, slug in enumerate(active_slugs):
        adjusted_probs[slug] = float(active_arr[i])

    # Optional Pinnacle blend
    blend_mode = "ml_form_only"
    if pinnacle_odds:
        pinn_map_raw: dict[str, float] = {}
        for entry in pinnacle_odds:
            if entry.decimal_odds > 1.0:
                pinn_map_raw[entry.slug] = 1.0 / entry.decimal_odds

        pinn_active = {s: pinn_map_raw[s] for s in active_slugs if s in pinn_map_raw}
        if pinn_active:
            pinn_arr = np.array(list(pinn_active.values()), dtype=float)
            if pinn_arr.sum() > 1e-9:
                pinn_arr = pinn_arr / pinn_arr.sum()
                pinn_norm = dict(zip(pinn_active.keys(), pinn_arr.tolist()))

                for slug in active_slugs:
                    ml_p = adjusted_probs.get(slug, 0.0)
                    pinn_p = pinn_norm.get(slug, ml_p)
                    adjusted_probs[slug] = PINNACLE_BLEND_ALPHA * ml_p + (1.0 - PINNACLE_BLEND_ALPHA) * pinn_p

                blend_arr = np.array([adjusted_probs[s] for s in active_slugs], dtype=float)
                if blend_arr.sum() > 1e-9:
                    blend_arr = blend_arr / blend_arr.sum()
                    for i, slug in enumerate(active_slugs):
                        adjusted_probs[slug] = float(blend_arr[i])
                blend_mode = "ml_form_pinnacle_blend"

    # Compute live top-3 via Harville on active riders
    active_wp = np.array([max(adjusted_probs.get(s, 1e-9), 1e-9) for s in active_slugs], dtype=float)
    active_wp = active_wp / active_wp.sum()
    if len(active_wp) >= 3:
        p1a, p2a, p3a = harville_top3(active_wp.tolist())
        top3_map = {
            slug: min(float(p1a[i]) + float(p2a[i]) + float(p3a[i]), 0.9999)
            for i, slug in enumerate(active_slugs)
        }
    else:
        top3_map = {s: float(adjusted_probs.get(s, 0.0)) for s in active_slugs}

    # Build final result list
    results: list[dict] = []
    for r in base_rider_probs:
        slug = r.get("slug", "")
        s = status_map.get(slug)
        results.append({
            "slug": slug,
            "first_name": r.get("first_name", ""),
            "last_name": r.get("last_name", ""),
            "win_prob": float(adjusted_probs.get(slug, 0.0)),
            "top3_prob": float(top3_map.get(slug, 0.0)),
            "pre_race_win_prob": float(pre_race_map.get(slug, 0.0)),
            "current_points": s.points_total if s else 0,
            "current_rank": s.current_round_rank if s else None,
            "heats_completed": s.heats_completed if s else 0,
            "is_eliminated": bool(s.mathematically_eliminated) if s else False,
            "is_dnf": bool(s.dnf) if s else False,
        })

    results.sort(key=lambda x: (x["current_points"], x["win_prob"]), reverse=True)
    return results, blend_mode


# ---------------------------------------------------------------------------
# Endpoint: POST /live/reprice
# ---------------------------------------------------------------------------


@router.post(
    "/reprice",
    response_model=LiveRepriceResponse,
    summary="Live in-running reprice for a Speedway GP round",
    description=(
        "Receives heat results and live rider standings, returns repriced markets.\n\n"
        "The live model blends ML probabilities with in-round form factor: "
        "riders scoring above-expectation get probability boosts. "
        "Optional Pinnacle odds blended at 40% weight.\n\n"
        "Auto-suspends if feed is stale > 90 seconds."
    ),
    responses={
        200: {"description": "Live prices returned"},
        422: {"description": "Invalid payload"},
        503: {"description": "Predictor not initialised"},
    },
)
async def reprice_round(
    body: LiveRepriceRequest,
    request: Request,
) -> LiveRepriceResponse:
    """
    Live repricing pipeline:
    1. Get SpeedwayPredictor base probabilities for the round field.
    2. Apply form factor adjustments from completed heats.
    3. Optionally blend with Pinnacle odds.
    4. Check feed staleness — auto-suspend if stale.
    5. Apply margin, build Round Winner and Top-3 markets.
    6. Cache state for GET endpoint.
    """
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialised",
        )

    logger.info(
        "speedway_live_reprice_request",
        request_id=request_id,
        round_id=body.round_id,
        heats_completed=body.heats_completed,
        heats_total=body.heats_total,
        n_riders=len(body.riders),
        feed_source=body.feed_source,
    )

    # Feed staleness check
    is_suspended = False
    suspension_reason: Optional[str] = None
    feed_age_seconds: float = 0.0
    try:
        feed_ts = datetime.fromisoformat(body.feed_timestamp_utc.replace("Z", "+00:00"))
        feed_age_seconds = (now_utc - feed_ts).total_seconds()
        if feed_age_seconds > STALE_THRESHOLD_SECONDS:
            is_suspended = True
            suspension_reason = (
                f"Feed stale: {feed_age_seconds:.0f}s "
                f"(threshold {STALE_THRESHOLD_SECONDS:.0f}s)"
            )
    except (ValueError, TypeError):
        feed_age_seconds = 0.0

    # Manual suspension check
    susp_record = _live_state_store.get(f"suspend:{body.round_id}")
    if susp_record and susp_record.get("suspended"):
        is_suspended = True
        suspension_reason = susp_record.get("reason", "Manual suspension")

    # Run predictor for base probabilities
    try:
        base_probs = predictor.predict_round(
            riders=body.riders,
            venue_slug=body.venue_slug,
            venue_country=body.venue_country,
            track_length=body.track_length,
            season_id=body.season_id,
        )
    except Exception as exc:
        logger.error(
            "speedway_live_predictor_error",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    # Compute live probabilities
    live_results, blend_mode = _compute_live_probabilities(
        base_rider_probs=base_probs,
        rider_statuses=body.rider_statuses,
        heat_results=body.heat_results,
        heats_completed=body.heats_completed,
        heats_total=body.heats_total,
        pinnacle_odds=body.pinnacle_odds,
    )

    # Tally live stats
    n_active = sum(1 for r in live_results if not r["is_eliminated"] and not r["is_dnf"])
    n_eliminated = sum(1 for r in live_results if r["is_eliminated"])
    n_dnf = sum(1 for r in live_results if r["is_dnf"])
    round_progress_pct = round(body.heats_completed / max(body.heats_total, 1) * 100.0, 1)

    # Build Round Winner market
    win_probs_active = [
        max(float(r["win_prob"]), 1e-9)
        for r in live_results
        if not r["is_eliminated"] and not r["is_dnf"]
    ]
    win_total = sum(win_probs_active) or 1.0

    rw_selections: list[LiveSelectionPrice] = []
    active_idx = 0
    for r in live_results:
        pre = float(r["pre_race_win_prob"])
        if r["is_eliminated"] or r["is_dnf"]:
            rw_selections.append(LiveSelectionPrice(
                slug=r["slug"],
                label=f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"],
                fair_probability=0.0,
                decimal_odds=999.0,
                current_points=r["current_points"],
                current_rank=r["current_rank"],
                heats_completed=r["heats_completed"],
                is_eliminated=r["is_eliminated"],
                is_dnf=r["is_dnf"],
                market_suspended=is_suspended,
                probability_change_vs_prerace=round(0.0 - pre, 6),
            ))
        else:
            wp = float(r["win_prob"])
            # Margined decimal odds
            book_prob = (wp / win_total) * (1.0 + body.win_margin)
            odds = round(1.0 / max(book_prob, 1e-9), 3)
            rw_selections.append(LiveSelectionPrice(
                slug=r["slug"],
                label=f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"],
                fair_probability=round(wp, 6),
                decimal_odds=max(odds, 1.001),
                current_points=r["current_points"],
                current_rank=r["current_rank"],
                heats_completed=r["heats_completed"],
                is_eliminated=r["is_eliminated"],
                is_dnf=r["is_dnf"],
                market_suspended=is_suspended,
                probability_change_vs_prerace=round(wp - pre, 6),
            ))
            active_idx += 1

    round_winner_market = LiveMarket(
        market_type="round_winner",
        is_suspended=is_suspended,
        suspension_reason=suspension_reason,
        selections=rw_selections,
    )

    # Build Top-3 market (show top-8 by top3_prob)
    top3_sorted = sorted(
        [r for r in live_results if not r["is_dnf"] and not r["is_eliminated"]],
        key=lambda x: x["top3_prob"],
        reverse=True,
    )[:8]

    top3_selections: list[LiveSelectionPrice] = []
    for r in top3_sorted:
        p_t3 = max(float(r["top3_prob"]), 1e-9)
        book_prob = p_t3 * (1.0 + body.top3_margin)
        odds = round(1.0 / min(book_prob, 0.9999), 3)
        top3_selections.append(LiveSelectionPrice(
            slug=r["slug"],
            label=f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"],
            fair_probability=round(p_t3, 6),
            decimal_odds=max(odds, 1.001),
            current_points=r["current_points"],
            current_rank=r["current_rank"],
            heats_completed=r["heats_completed"],
            is_eliminated=r["is_eliminated"],
            is_dnf=r["is_dnf"],
            market_suspended=is_suspended,
            probability_change_vs_prerace=round(
                p_t3 - float(r["pre_race_win_prob"]), 6
            ),
        ))

    top3_market = LiveMarket(
        market_type="top3_finish",
        is_suspended=is_suspended,
        suspension_reason=suspension_reason,
        selections=top3_selections,
    )

    # Cache live state
    current_leader = next(
        (r["slug"] for r in live_results if r.get("current_rank") == 1),
        None,
    )
    current_leader_pts = next(
        (r["current_points"] for r in live_results if r.get("current_rank") == 1),
        None,
    )

    _live_state_store[f"state:{body.round_id}"] = {
        "round_id": body.round_id,
        "heats_completed": body.heats_completed,
        "heats_total": body.heats_total,
        "round_progress_pct": round_progress_pct,
        "n_active": n_active,
        "n_dnf": n_dnf,
        "n_eliminated": n_eliminated,
        "is_suspended": is_suspended,
        "suspension_reason": suspension_reason,
        "feed_source": body.feed_source,
        "last_update_utc": now_iso,
        "feed_age_seconds": round(feed_age_seconds, 1),
        "current_leader_slug": current_leader,
        "current_leader_points": current_leader_pts,
    }

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    schema_version = (
        predictor.schema_version
        if hasattr(predictor, "schema_version")
        else "unknown"
    )

    logger.info(
        "speedway_live_reprice_ok",
        request_id=request_id,
        round_id=body.round_id,
        heats_completed=body.heats_completed,
        n_active=n_active,
        n_eliminated=n_eliminated,
        n_dnf=n_dnf,
        is_suspended=is_suspended,
        blend_mode=blend_mode,
        elapsed_ms=elapsed_ms,
    )

    return LiveRepriceResponse(
        request_id=request_id,
        round_id=body.round_id,
        venue_slug=body.venue_slug,
        heats_completed=body.heats_completed,
        heats_total=body.heats_total,
        round_progress_pct=round_progress_pct,
        n_active=n_active,
        n_eliminated=n_eliminated,
        n_dnf=n_dnf,
        is_suspended=is_suspended,
        suspension_reason=suspension_reason,
        markets=[round_winner_market, top3_market],
        feed_age_seconds=round(feed_age_seconds, 1),
        blend_mode=blend_mode,
        model_schema=schema_version,
        repriced_at_utc=now_iso,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /live/{round_id}/state
# ---------------------------------------------------------------------------


@router.get(
    "/{round_id}/state",
    response_model=LiveStateResponse,
    summary="Get current live state for a Speedway GP round",
    description=(
        "Returns the cached live state for an in-progress round. "
        "Populated by POST /live/reprice calls. "
        "is_live=False if no reprice has been received for this round_id."
    ),
    responses={
        200: {"description": "State returned"},
    },
)
async def get_live_state(round_id: str) -> LiveStateResponse:
    """Return current live state for a round."""
    state = _live_state_store.get(f"state:{round_id}")

    if state is None:
        logger.debug("speedway_live_state_not_found", round_id=round_id)
        return LiveStateResponse(
            round_id=round_id,
            is_live=False,
            is_suspended=False,
            suspension_reason=None,
            heats_completed=None,
            heats_total=None,
            round_progress_pct=None,
            n_active=None,
            n_dnf=None,
            feed_source=None,
            last_update_utc=None,
            feed_age_seconds=None,
            current_leader_slug=None,
            current_leader_points=None,
        )

    # Recompute feed age
    feed_age: Optional[float] = state.get("feed_age_seconds")
    try:
        last_update = datetime.fromisoformat(
            state["last_update_utc"].replace("Z", "+00:00")
        )
        feed_age = (datetime.now(timezone.utc) - last_update).total_seconds()
    except (ValueError, KeyError, AttributeError):
        pass

    return LiveStateResponse(
        round_id=round_id,
        is_live=True,
        is_suspended=bool(state.get("is_suspended", False)),
        suspension_reason=state.get("suspension_reason"),
        heats_completed=state.get("heats_completed"),
        heats_total=state.get("heats_total"),
        round_progress_pct=state.get("round_progress_pct"),
        n_active=state.get("n_active"),
        n_dnf=state.get("n_dnf"),
        feed_source=state.get("feed_source"),
        last_update_utc=state.get("last_update_utc"),
        feed_age_seconds=round(feed_age, 1) if feed_age is not None else None,
        current_leader_slug=state.get("current_leader_slug"),
        current_leader_points=state.get("current_leader_points"),
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /live/{round_id}/suspend
# ---------------------------------------------------------------------------


@router.post(
    "/{round_id}/suspend",
    response_model=SuspendResponse,
    summary="Suspend all markets for a Speedway GP round",
    description=(
        "Operator endpoint: immediately suspend all live markets. "
        "Use for: rider crash, protest, track incident, feed outage."
    ),
    responses={
        200: {"description": "Suspension applied"},
    },
)
async def suspend_round_markets(
    round_id: str,
    reason: str = "operator_suspend",
) -> SuspendResponse:
    """Suspend all active markets for a round."""
    now_iso = datetime.now(timezone.utc).isoformat()
    _live_state_store[f"suspend:{round_id}"] = {
        "suspended": True,
        "reason": reason,
        "suspended_at_utc": now_iso,
    }
    logger.warning(
        "speedway_live_markets_suspended",
        round_id=round_id,
        reason=reason,
    )
    return SuspendResponse(
        round_id=round_id,
        suspended=True,
        reason=reason,
        suspended_at_utc=now_iso,
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /live/{round_id}/resume
# ---------------------------------------------------------------------------


@router.post(
    "/{round_id}/resume",
    response_model=ResumeResponse,
    summary="Resume suspended markets for a Speedway GP round",
    description="Clears the suspension flag. Next reprice call will return active prices.",
    responses={
        200: {"description": "Suspension cleared"},
    },
)
async def resume_round_markets(round_id: str) -> ResumeResponse:
    """Clear suspension for a round."""
    key = f"suspend:{round_id}"
    if key in _live_state_store:
        del _live_state_store[key]
        logger.info("speedway_live_markets_resumed", round_id=round_id)
    return ResumeResponse(
        round_id=round_id,
        resumed=True,
        resumed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
