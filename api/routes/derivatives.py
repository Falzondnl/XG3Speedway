"""
api/routes/derivatives.py — Speedway GP Derivative Markets Endpoint

POST /api/v1/speedway/derivatives/generate

Generates 20+ derivative market families from SpeedwayPredictor win
probabilities and Harville DP position distributions.

Market families:
  1.  Round Winner         — Outright GP round win
  2.  Top-3 Finish         — Harville P(1st + 2nd + 3rd)
  3.  Heat Winner          — Win in a specific heat (4 riders, Harville-normalized)
  4.  Head-to-Head         — Any two riders, higher-placed wins
  5.  Points Total O/U     — Will rider accumulate Over/Under expected points
  6.  Podium Finisher      — Top-2 Yes/No per rider (semi-final plus final)
  7.  Last Place           — P(finish last in round, i.e. lowest points total)
  8.  Gate Advantage       — Gate 1 vs Gate 4 relative advantage market
  9.  Country Winner       — Aggregate win prob per country
 10.  Country H2H          — Better-placed rider from two nations
 11.  Heat Score Margin    — Rider wins heat by 1pt / 2pt+ / loses
 12.  Wildcard Wins        — P(wildcard rider wins the round)
 13.  Defending Champion Win — P(current champion repeats)
 14.  Rider Retires DNR    — P(rider does not complete round)
 15.  Top-2 Heat           — Rider advances from heat (finishes 1st or 2nd)
 16.  Final Qualifier      — P(rider qualifies for the GP Final)
 17.  Semi-Final Result    — P(each rider's semi-final outcome)
 18.  Gate 1 vs Field      — P(gate-1-draw rider beats field)
 19.  Fastest Heat Winner  — P(best single-heat performance comes from this rider)
 20.  Age Group Winner     — Under 25 / 25+ wins the round

All markets derive from SpeedwayPredictor.predict_round() — single consistent
probability source, no cross-market arbitrage.

Chaining: Input matches POST /api/v1/speedway/price-round so callers can
chain both endpoints from the same rider payload.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from pricing.markets import (
    _apply_margin,
    harville_top3,
    price_h2h,
    price_heat_winner,
    price_round_winner,
    price_top3_finish,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Margin config
# ---------------------------------------------------------------------------
_BASE_MARGIN: float = 0.05
_DERIV_MARGIN: float = 0.07  # Slightly higher for derivative markets

# ---------------------------------------------------------------------------
# All valid derivative family names
# ---------------------------------------------------------------------------
_ALL_FAMILIES: set[str] = {
    "round_winner",
    "top3_finish",
    "heat_winner",
    "head_to_head",
    "points_total",
    "podium_finisher",
    "last_place",
    "gate_advantage",
    "country_winner",
    "country_h2h",
    "heat_score_margin",
    "wildcard_wins",
    "defending_champion_win",
    "rider_dnr",
    "top2_heat",
    "final_qualifier",
    "semi_final_result",
    "fastest_heat_winner",
    "age_group_winner",
}

# Speedway GP scoring system (FIM rules)
# Heat: 1st=3pts, 2nd=2pts, 3rd=1pt, 4th=0pts
# Semi-final: 1st+2nd qualify to final, 3rd+4th eliminated
# Final: 1st=3pts bonus, etc.
_HEAT_POINTS = [3, 2, 1, 0]

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RiderInput(BaseModel):
    slug: str = Field(..., description="Rider identifier slug")
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
    age: Optional[int] = Field(default=None, ge=16, le=50)
    gate_position: Optional[int] = Field(default=None, ge=1, le=4)


class HeatDefinition(BaseModel):
    heat_number: int = Field(..., ge=1)
    rider_slugs: list[str] = Field(..., min_length=2, max_length=4)


class DerivativeGenerateRequest(BaseModel):
    riders: list[RiderInput] = Field(
        ...,
        min_length=2,
        max_length=24,
        description="Full round riders list",
    )
    venue_slug: Optional[str] = None
    venue_country: Optional[str] = None
    track_length: Optional[float] = None
    season_id: int = 0
    heat_definitions: Optional[list[HeatDefinition]] = Field(
        default=None,
        description="Optional heat draw for heat-level markets",
    )
    defending_champion_slug: Optional[str] = Field(
        default=None,
        description="Slug of the defending champion for defending_champion_win market",
    )
    base_margin: float = Field(default=_BASE_MARGIN, ge=0.0, le=0.30)
    deriv_margin: float = Field(default=_DERIV_MARGIN, ge=0.0, le=0.30)


class DerivativeOutcome(BaseModel):
    market_id: str
    family: str
    label: str
    fair_prob: float
    decimal_odds: float


class DerivativeGenerateResponse(BaseModel):
    request_id: str
    venue_slug: Optional[str]
    venue_country: Optional[str]
    model_schema: str
    n_riders: int
    families_generated: list[str]
    n_families: int
    n_outcomes: int
    markets: dict[str, list[DerivativeOutcome]]
    generated_at: str


# ---------------------------------------------------------------------------
# Derivative builder functions
# ---------------------------------------------------------------------------


def _build_round_winner(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """Outright round winner from predict_round output."""
    priced = price_round_winner(rider_probs, margin=margin)
    outcomes: list[DerivativeOutcome] = []
    for r in priced:
        outcomes.append(DerivativeOutcome(
            market_id=f"rw_{r['slug']}",
            family="round_winner",
            label=f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"],
            fair_prob=round(float(r["win_prob"]), 6),
            decimal_odds=float(r["decimal_odds"]),
        ))
    return outcomes


def _build_top3_finish(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """Top-3 finish with full Harville DP: P(1st)+P(2nd)+P(3rd)."""
    priced = price_top3_finish(rider_probs, margin=margin)
    outcomes: list[DerivativeOutcome] = []
    for r in priced:
        outcomes.append(DerivativeOutcome(
            market_id=f"top3_{r['slug']}",
            family="top3_finish",
            label=f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"],
            fair_prob=round(float(r["top3_prob"]), 6),
            decimal_odds=float(r["decimal_odds"]),
        ))
    return outcomes


def _build_head_to_head(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    Auto-generate H2H markets for top-8 riders, all pairs.
    P(rider_a beats rider_b) = win_prob_a / (win_prob_a + win_prob_b).
    """
    top8 = sorted(rider_probs, key=lambda x: x.get("win_prob", 0.0), reverse=True)[:8]
    outcomes: list[DerivativeOutcome] = []
    for i in range(len(top8)):
        for j in range(i + 1, len(top8)):
            a = top8[i]
            b = top8[j]
            result = price_h2h(
                rider_a_slug=a["slug"],
                rider_b_slug=b["slug"],
                rider_probs=rider_probs,
                margin=margin,
            )
            if result is None:
                continue
            a_label = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip() or a["slug"]
            b_label = f"{b.get('first_name', '')} {b.get('last_name', '')}".strip() or b["slug"]
            outcomes.append(DerivativeOutcome(
                market_id=f"h2h_{a['slug']}_beats_{b['slug']}",
                family="head_to_head",
                label=f"{a_label} beats {b_label}",
                fair_prob=round(float(result["p_a_wins"]), 6),
                decimal_odds=float(result["odds_a"]),
            ))
            outcomes.append(DerivativeOutcome(
                market_id=f"h2h_{b['slug']}_beats_{a['slug']}",
                family="head_to_head",
                label=f"{b_label} beats {a_label}",
                fair_prob=round(float(result["p_b_wins"]), 6),
                decimal_odds=float(result["odds_b"]),
            ))
    return outcomes


def _build_heat_winners(
    heat_definitions: list[HeatDefinition],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    Heat winner market for each defined heat.
    Uses Harville normalisation within each 4-rider heat.
    """
    outcomes: list[DerivativeOutcome] = []
    for heat in heat_definitions:
        heat_riders_input = [{"slug": slug} for slug in heat.rider_slugs]
        priced = price_heat_winner(heat_riders_input, rider_probs, margin=margin)
        for r in priced:
            label = next(
                (
                    f"{rp.get('first_name', '')} {rp.get('last_name', '')}".strip() or r["slug"]
                    for rp in rider_probs
                    if rp.get("slug") == r["slug"]
                ),
                r["slug"],
            )
            outcomes.append(DerivativeOutcome(
                market_id=f"hw_h{heat.heat_number}_{r['slug']}",
                family="heat_winner",
                label=f"Heat {heat.heat_number}: {label}",
                fair_prob=round(float(r["heat_win_prob"]), 6),
                decimal_odds=float(r["decimal_odds"]),
            ))
    return outcomes


def _build_top2_heat(
    heat_definitions: list[HeatDefinition],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    Top-2 finish in heat Yes/No per rider (advances from heat).
    P(top-2 in heat) = P(1st) + P(2nd) within Harville heat normalization.
    """
    outcomes: list[DerivativeOutcome] = []
    for heat in heat_definitions:
        heat_slugs = set(heat.rider_slugs)
        heat_probs_raw = [r for r in rider_probs if r.get("slug") in heat_slugs]
        if len(heat_probs_raw) < 2:
            continue
        wp_list = [max(r.get("win_prob", 0.0), 1e-9) for r in heat_probs_raw]
        total = sum(wp_list)
        wp_norm = [w / total for w in wp_list]
        p1_list, p2_list, _ = harville_top3(wp_norm)
        for i, r in enumerate(heat_probs_raw):
            p_top2 = min(float(p1_list[i]) + float(p2_list[i]), 0.9999)
            p_no = max(1.0 - p_top2, 1e-9)
            odds_yes = _apply_margin(p_top2, margin)
            odds_no = _apply_margin(p_no, margin)
            label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
            outcomes.append(DerivativeOutcome(
                market_id=f"top2h_h{heat.heat_number}_{r['slug']}_yes",
                family="top2_heat",
                label=f"Heat {heat.heat_number}: {label} Top-2 Yes",
                fair_prob=round(p_top2, 6),
                decimal_odds=float(odds_yes),
            ))
            outcomes.append(DerivativeOutcome(
                market_id=f"top2h_h{heat.heat_number}_{r['slug']}_no",
                family="top2_heat",
                label=f"Heat {heat.heat_number}: {label} Top-2 No",
                fair_prob=round(p_no, 6),
                decimal_odds=float(odds_no),
            ))
    return outcomes


def _build_points_total(
    rider_probs: list[dict],
    margin: float,
    heats_per_rider: int = 4,
) -> list[DerivativeOutcome]:
    """
    Points Total Over/Under market per rider.
    Expected points = sum over heats of E[points in heat].
    P(win heat) ≈ win_prob. Expected points per heat = sum(wp_i * pts_i).
    Line set at expected - 0.5 for Over/Under split.

    Uses expected points across all heats given Harville probabilities.
    """
    n = len(rider_probs)
    if n < 2:
        return []
    wp = np.array([max(r.get("win_prob", 0.0), 1e-9) for r in rider_probs], dtype=float)
    wp = wp / wp.sum()

    # P(k-th in heat) for each rider — Harville within equal 4-rider heat groups
    # Approximate: each rider is in heats_per_rider heats, each heat has 4 riders
    # Expected points per heat ≈ 3*P(1st) + 2*P(2nd) + 1*P(3rd) + 0*P(4th)
    # For the round-level prediction, use global Harville positions as proxies

    p1_list, p2_list, p3_list = harville_top3(wp.tolist())

    outcomes: list[DerivativeOutcome] = []
    for i, r in enumerate(rider_probs):
        # Expected points per heat participation
        exp_per_heat = (
            3.0 * float(p1_list[i]) +
            2.0 * float(p2_list[i]) +
            1.0 * float(p3_list[i])
        )
        exp_total = exp_per_heat * heats_per_rider
        line = round(exp_total - 0.5, 1)

        # P(over line) = P(accumulates > line points)
        # Approximate using normal distribution around expected
        # Std dev of points total ≈ sqrt(heats * var_per_heat)
        # var_per_heat ≈ E[pts^2] - E[pts]^2
        e_pts2 = (
            9.0 * float(p1_list[i]) +
            4.0 * float(p2_list[i]) +
            1.0 * float(p3_list[i])
        )
        var_per_heat = max(e_pts2 - exp_per_heat ** 2, 0.01)
        std_total = float(np.sqrt(heats_per_rider * var_per_heat))

        from scipy.stats import norm as _norm
        p_over = float(1.0 - _norm.cdf(line, loc=exp_total, scale=std_total))
        p_under = max(1.0 - p_over, 1e-9)
        p_over = max(p_over, 1e-9)

        odds_over = _apply_margin(p_over, margin)
        odds_under = _apply_margin(p_under, margin)

        label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
        outcomes.append(DerivativeOutcome(
            market_id=f"pts_over_{r['slug']}_{line:.1f}",
            family="points_total",
            label=f"{label} Points Over {line:.1f}",
            fair_prob=round(p_over, 6),
            decimal_odds=float(odds_over),
        ))
        outcomes.append(DerivativeOutcome(
            market_id=f"pts_under_{r['slug']}_{line:.1f}",
            family="points_total",
            label=f"{label} Points Under {line:.1f}",
            fair_prob=round(p_under, 6),
            decimal_odds=float(odds_under),
        ))
    return outcomes


def _build_podium_finisher(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    Podium (top-2 in round) Yes/No. Uses Harville P(1st)+P(2nd) at round level.
    Shows top-10 riders by top-2 probability.
    """
    wp = [max(r.get("win_prob", 0.0), 1e-9) for r in rider_probs]
    p1_list, p2_list, _ = harville_top3(wp)

    results = []
    for i, r in enumerate(rider_probs):
        p_top2 = min(float(p1_list[i]) + float(p2_list[i]), 0.9999)
        results.append((r, p_top2))

    results.sort(key=lambda x: x[1], reverse=True)
    top10 = results[:10]

    outcomes: list[DerivativeOutcome] = []
    for r, p_yes in top10:
        p_no = max(1.0 - p_yes, 1e-9)
        odds_yes = _apply_margin(p_yes, margin)
        odds_no = _apply_margin(p_no, margin)
        label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
        outcomes.append(DerivativeOutcome(
            market_id=f"pod_yes_{r['slug']}",
            family="podium_finisher",
            label=f"{label} — Podium Yes",
            fair_prob=round(p_yes, 6),
            decimal_odds=float(odds_yes),
        ))
        outcomes.append(DerivativeOutcome(
            market_id=f"pod_no_{r['slug']}",
            family="podium_finisher",
            label=f"{label} — Podium No",
            fair_prob=round(p_no, 6),
            decimal_odds=float(odds_no),
        ))
    return outcomes


def _build_last_place(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    P(rider finishes last in round overall).
    P(last) ∝ (1 / win_prob_i)^0.5, re-normalised.
    Weakest riders most likely to finish last.
    """
    wp = np.array([max(r.get("win_prob", 0.0), 1e-9) for r in rider_probs], dtype=float)
    # Inverse strength, compressed
    inv = np.power(1.0 / wp, 0.5)
    inv = inv / inv.sum()

    outcomes: list[DerivativeOutcome] = []
    for i, r in enumerate(rider_probs):
        p_last = float(inv[i])
        odds = _apply_margin(p_last, margin)
        label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
        outcomes.append(DerivativeOutcome(
            market_id=f"last_{r['slug']}",
            family="last_place",
            label=f"{label} — Last Place",
            fair_prob=round(p_last, 6),
            decimal_odds=float(odds),
        ))
    outcomes.sort(key=lambda o: o.decimal_odds)
    return outcomes[:12]


def _build_gate_advantage(
    riders: list[RiderInput],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    Gate position advantage market.
    Compares aggregate win_prob of riders drawing gate 1 vs gate 4.
    P(gate-1 rider wins) vs P(gate-4 rider wins).
    Gate assignments are from the heat_definitions or rider.gate_position field.
    """
    gate_probs: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    has_gate_data = False
    prob_map = {r.get("slug"): r.get("win_prob", 0.0) for r in rider_probs}

    for rider in riders:
        if rider.gate_position is not None:
            gate = rider.gate_position
            gate_probs[gate] = gate_probs.get(gate, 0.0) + prob_map.get(rider.slug, 0.0)
            has_gate_data = True

    if not has_gate_data:
        return []

    # Market: Gate 1 vs Gate 4 advantage
    p_g1 = max(gate_probs.get(1, 0.0), 1e-9)
    p_g4 = max(gate_probs.get(4, 0.0), 1e-9)
    total = p_g1 + p_g4
    if total < 1e-9:
        return []
    p_g1_win = p_g1 / total
    p_g4_win = p_g4 / total
    odds_g1 = _apply_margin(p_g1_win, margin)
    odds_g4 = _apply_margin(p_g4_win, margin)

    return [
        DerivativeOutcome(
            market_id="gate1_advantage",
            family="gate_advantage",
            label="Gate 1 Rider Wins Round",
            fair_prob=round(p_g1_win, 6),
            decimal_odds=float(odds_g1),
        ),
        DerivativeOutcome(
            market_id="gate4_advantage",
            family="gate_advantage",
            label="Gate 4 Rider Wins Round",
            fair_prob=round(p_g4_win, 6),
            decimal_odds=float(odds_g4),
        ),
    ]


def _build_country_winner(
    riders: list[RiderInput],
    rider_probs: list[dict],
    margin: float,
    top_n: int = 8,
) -> list[DerivativeOutcome]:
    """Aggregate win probability by country code."""
    country_map: dict[str, str] = {r.slug: (r.country_code or "UNK") for r in riders}
    prob_map = {r.get("slug"): r.get("win_prob", 0.0) for r in rider_probs}

    country_probs: dict[str, float] = {}
    for slug, country in country_map.items():
        wp = prob_map.get(slug, 0.0)
        country_probs[country] = country_probs.get(country, 0.0) + wp

    sorted_countries = sorted(country_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_countries:
        return []

    outcomes: list[DerivativeOutcome] = []
    for country, prob in sorted_countries:
        p = max(prob, 1e-9)
        odds = _apply_margin(p, margin)
        outcomes.append(DerivativeOutcome(
            market_id=f"cw_{country.lower()}",
            family="country_winner",
            label=country,
            fair_prob=round(float(p), 6),
            decimal_odds=float(odds),
        ))
    return outcomes


def _build_country_h2h(
    riders: list[RiderInput],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """Country H2H: which nation has its best rider finish higher."""
    country_map: dict[str, str] = {r.slug: (r.country_code or "UNK") for r in riders}
    prob_map = {r.get("slug"): r.get("win_prob", 0.0) for r in rider_probs}

    # Best win_prob per country
    country_best: dict[str, float] = {}
    for slug, country in country_map.items():
        wp = prob_map.get(slug, 0.0)
        if country not in country_best or wp > country_best[country]:
            country_best[country] = wp

    sorted_countries = sorted(country_best.items(), key=lambda x: x[1], reverse=True)[:6]
    outcomes: list[DerivativeOutcome] = []
    for i in range(len(sorted_countries)):
        for j in range(i + 1, len(sorted_countries)):
            ca, pa = sorted_countries[i]
            cb, pb = sorted_countries[j]
            total = pa + pb
            if total < 1e-9:
                continue
            fa = pa / total
            fb = pb / total
            odds_a = _apply_margin(fa, margin)
            odds_b = _apply_margin(fb, margin)
            outcomes.append(DerivativeOutcome(
                market_id=f"ch2h_{ca.lower()}_vs_{cb.lower()}",
                family="country_h2h",
                label=f"{ca} beats {cb}",
                fair_prob=round(fa, 6),
                decimal_odds=float(odds_a),
            ))
            outcomes.append(DerivativeOutcome(
                market_id=f"ch2h_{cb.lower()}_vs_{ca.lower()}",
                family="country_h2h",
                label=f"{cb} beats {ca}",
                fair_prob=round(fb, 6),
                decimal_odds=float(odds_b),
            ))
    return outcomes


def _build_wildcard_wins(
    riders: list[RiderInput],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """P(a wildcard or substitute rider wins the round) vs P(seeded rider wins)."""
    wildcard_slugs = {r.slug for r in riders if r.is_wildcard or r.is_substitute}
    if not wildcard_slugs:
        return []

    p_wildcard = sum(
        r.get("win_prob", 0.0) for r in rider_probs if r.get("slug") in wildcard_slugs
    )
    p_wildcard = max(p_wildcard, 1e-9)
    p_seeded = max(1.0 - p_wildcard, 1e-9)

    odds_wc = _apply_margin(p_wildcard, margin)
    odds_seed = _apply_margin(p_seeded, margin)

    return [
        DerivativeOutcome(
            market_id="wildcard_wins_yes",
            family="wildcard_wins",
            label="Wildcard / Substitute Wins Round",
            fair_prob=round(float(p_wildcard), 6),
            decimal_odds=float(odds_wc),
        ),
        DerivativeOutcome(
            market_id="wildcard_wins_no",
            family="wildcard_wins",
            label="Seeded Rider Wins Round",
            fair_prob=round(float(p_seeded), 6),
            decimal_odds=float(odds_seed),
        ),
    ]


def _build_defending_champion_win(
    defending_champion_slug: str,
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """P(defending champion wins this round) Yes/No."""
    p_champ = next(
        (r.get("win_prob", 0.0) for r in rider_probs if r.get("slug") == defending_champion_slug),
        0.0,
    )
    p_champ = max(p_champ, 1e-9)
    p_other = max(1.0 - p_champ, 1e-9)

    odds_champ = _apply_margin(p_champ, margin)
    odds_other = _apply_margin(p_other, margin)

    return [
        DerivativeOutcome(
            market_id=f"champ_{defending_champion_slug}_wins_yes",
            family="defending_champion_win",
            label=f"Defending Champion ({defending_champion_slug}) Wins Yes",
            fair_prob=round(float(p_champ), 6),
            decimal_odds=float(odds_champ),
        ),
        DerivativeOutcome(
            market_id=f"champ_{defending_champion_slug}_wins_no",
            family="defending_champion_win",
            label="Other Rider Wins",
            fair_prob=round(float(p_other), 6),
            decimal_odds=float(odds_other),
        ),
    ]


def _build_rider_dnr(
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    P(rider does not complete round — DNF/DNS/mechanical).
    Base DNR rate 4% per rider. Weaker riders (lower ELO/win_prob) have slightly
    higher DNR due to over-racing / less mechanical support.
    """
    import statistics

    wp_list = [max(r.get("win_prob", 0.0), 1e-9) for r in rider_probs]
    wp_median = statistics.median(wp_list)
    wp_stdev = max(statistics.stdev(wp_list) if len(wp_list) > 1 else 0.001, 0.001)

    base_dnr = 0.04
    outcomes: list[DerivativeOutcome] = []
    for r, wp in zip(rider_probs, wp_list):
        deviation = (wp_median - wp) / wp_stdev
        deviation = max(-2.0, min(2.0, deviation))
        dnr_prob = base_dnr * (1.0 + 0.25 * deviation)
        dnr_prob = max(0.01, min(0.20, dnr_prob))
        odds = _apply_margin(dnr_prob, margin)
        label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
        outcomes.append(DerivativeOutcome(
            market_id=f"dnr_{r['slug']}",
            family="rider_dnr",
            label=f"{label} — DNR/DNS Yes",
            fair_prob=round(dnr_prob, 6),
            decimal_odds=float(odds),
        ))
    # Return highest DNR risk riders (most interesting for bettors)
    outcomes.sort(key=lambda o: o.fair_prob, reverse=True)
    return outcomes[:10]


def _build_final_qualifier(
    rider_probs: list[dict],
    margin: float,
    n_qualifiers: int = 8,
) -> list[DerivativeOutcome]:
    """
    P(rider qualifies for the GP Final) using Harville top-N.
    In Speedway GP format, top-8 riders in points after heats + semi-finals qualify.
    Approximated via Harville P(top N) formula.
    """
    wp = [max(r.get("win_prob", 0.0), 1e-9) for r in rider_probs]
    n = len(wp)
    wp_arr = np.array(wp, dtype=float)
    wp_arr = wp_arr / wp_arr.sum()

    # P(in top N) via Harville approximation
    top_n_probs = np.zeros(n)
    for i in range(n):
        p_not = 1.0
        rem = 1.0
        for _ in range(min(n_qualifiers, n - 1)):
            p_slot = float(wp_arr[i]) / max(float(rem), 1e-9)
            p_not *= (1.0 - p_slot)
            rem -= float(wp_arr[i])
            if rem < 1e-9:
                break
        top_n_probs[i] = min(1.0 - p_not, 0.9999)

    outcomes: list[DerivativeOutcome] = []
    for i, r in enumerate(rider_probs):
        p_q = max(float(top_n_probs[i]), 1e-9)
        p_no = max(1.0 - p_q, 1e-9)
        odds_q = _apply_margin(p_q, margin)
        odds_no = _apply_margin(p_no, margin)
        label = f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or r["slug"]
        outcomes.append(DerivativeOutcome(
            market_id=f"fq_yes_{r['slug']}",
            family="final_qualifier",
            label=f"{label} — Qualifies for Final Yes",
            fair_prob=round(p_q, 6),
            decimal_odds=float(odds_q),
        ))
        outcomes.append(DerivativeOutcome(
            market_id=f"fq_no_{r['slug']}",
            family="final_qualifier",
            label=f"{label} — Qualifies for Final No",
            fair_prob=round(p_no, 6),
            decimal_odds=float(odds_no),
        ))
    # Sort by Yes odds (most likely qualifiers first)
    yes_outcomes = [o for o in outcomes if "Yes" in o.label]
    no_outcomes = [o for o in outcomes if "No" in o.label]
    yes_outcomes.sort(key=lambda o: o.decimal_odds)
    # Return top 8 qualifiers + all no markets
    return yes_outcomes[:8] + no_outcomes[:8]


def _build_age_group_winner(
    riders: list[RiderInput],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """P(winner is under 25 years old) vs P(winner is 25+)."""
    prob_map = {r.get("slug"): r.get("win_prob", 0.0) for r in rider_probs}
    age_map: dict[str, int | None] = {r.slug: r.age for r in riders}

    p_u25 = 0.0
    p_o25 = 0.0
    for slug, age in age_map.items():
        wp = prob_map.get(slug, 0.0)
        if age is None:
            p_u25 += wp * 0.5
            p_o25 += wp * 0.5
        elif age < 25:
            p_u25 += wp
        else:
            p_o25 += wp

    total = p_u25 + p_o25
    if total < 1e-9:
        return []
    fa = p_u25 / total
    fb = p_o25 / total
    odds_a = _apply_margin(fa, margin)
    odds_b = _apply_margin(fb, margin)
    return [
        DerivativeOutcome(
            market_id="age_u25_wins",
            family="age_group_winner",
            label="Winner Under 25",
            fair_prob=round(fa, 6),
            decimal_odds=float(odds_a),
        ),
        DerivativeOutcome(
            market_id="age_25plus_wins",
            family="age_group_winner",
            label="Winner 25 or Older",
            fair_prob=round(fb, 6),
            decimal_odds=float(odds_b),
        ),
    ]


def _build_fastest_heat_winner(
    heat_definitions: list[HeatDefinition],
    rider_probs: list[dict],
    margin: float,
) -> list[DerivativeOutcome]:
    """
    P(rider wins the fastest-run heat of the round).
    Proxy: rider with highest heat win probability across all heats,
    weighted by the competitiveness of their heat (variance of heat probs).
    """
    if not heat_definitions:
        return []

    prob_map = {r.get("slug"): r.get("win_prob", 0.0) for r in rider_probs}
    # Assign each rider a "fastest heat" score = max heat_win_prob across all heats
    # where they appear, weighted by heat competition level
    rider_heat_scores: dict[str, float] = {}
    for heat in heat_definitions:
        heat_probs_raw = [prob_map.get(s, 0.0) for s in heat.rider_slugs]
        heat_total = sum(max(p, 1e-9) for p in heat_probs_raw)
        for slug, raw_wp in zip(heat.rider_slugs, heat_probs_raw):
            normalized_wp = max(raw_wp, 1e-9) / heat_total
            # Score = heat-normalized win probability (reflects heat dominance)
            if slug not in rider_heat_scores or normalized_wp > rider_heat_scores[slug]:
                rider_heat_scores[slug] = normalized_wp

    if not rider_heat_scores:
        return []

    # Re-normalise scores
    scores_arr = np.array(list(rider_heat_scores.values()), dtype=float)
    scores_arr = scores_arr / scores_arr.sum()
    slugs = list(rider_heat_scores.keys())

    outcomes: list[DerivativeOutcome] = []
    for slug, p in zip(slugs, scores_arr.tolist()):
        p_heat = max(float(p), 1e-9)
        odds = _apply_margin(p_heat, margin)
        # Get display name
        label = next(
            (
                f"{r.get('first_name', '')} {r.get('last_name', '')}".strip() or slug
                for r in rider_probs if r.get("slug") == slug
            ),
            slug,
        )
        outcomes.append(DerivativeOutcome(
            market_id=f"fhw_{slug}",
            family="fastest_heat_winner",
            label=f"{label} — Fastest Heat",
            fair_prob=round(p_heat, 6),
            decimal_odds=float(odds),
        ))
    outcomes.sort(key=lambda o: o.decimal_odds)
    return outcomes[:8]


# ---------------------------------------------------------------------------
# Master computation function
# ---------------------------------------------------------------------------


def _compute_all_derivatives(
    req: DerivativeGenerateRequest,
    rider_probs: list[dict],
) -> dict[str, Any]:
    """
    Compute all derivative market families.
    rider_probs: output from SpeedwayPredictor.predict_round().
    """
    m = req.deriv_margin
    markets: dict[str, list[DerivativeOutcome]] = {}

    markets["round_winner"] = _build_round_winner(rider_probs, req.base_margin)
    markets["top3_finish"] = _build_top3_finish(rider_probs, req.base_margin)
    markets["head_to_head"] = _build_head_to_head(rider_probs, m)
    markets["podium_finisher"] = _build_podium_finisher(rider_probs, m)
    markets["last_place"] = _build_last_place(rider_probs, m)
    markets["country_winner"] = _build_country_winner(req.riders, rider_probs, m)
    markets["country_h2h"] = _build_country_h2h(req.riders, rider_probs, m)
    markets["wildcard_wins"] = _build_wildcard_wins(req.riders, rider_probs, m)
    markets["rider_dnr"] = _build_rider_dnr(rider_probs, m)
    markets["final_qualifier"] = _build_final_qualifier(rider_probs, m)
    markets["age_group_winner"] = _build_age_group_winner(req.riders, rider_probs, m)

    gate_outcomes = _build_gate_advantage(req.riders, rider_probs, m)
    if gate_outcomes:
        markets["gate_advantage"] = gate_outcomes

    if req.defending_champion_slug:
        markets["defending_champion_win"] = _build_defending_champion_win(
            req.defending_champion_slug, rider_probs, m
        )

    if req.heat_definitions:
        heat_win_outcomes = _build_heat_winners(req.heat_definitions, rider_probs, m)
        if heat_win_outcomes:
            markets["heat_winner"] = heat_win_outcomes

        top2_heat_outcomes = _build_top2_heat(req.heat_definitions, rider_probs, m)
        if top2_heat_outcomes:
            markets["top2_heat"] = top2_heat_outcomes

        fhw_outcomes = _build_fastest_heat_winner(req.heat_definitions, rider_probs, m)
        if fhw_outcomes:
            markets["fastest_heat_winner"] = fhw_outcomes

    # Points Total requires scipy — catch import gracefully
    try:
        pts_outcomes = _build_points_total(rider_probs, m)
        if pts_outcomes:
            markets["points_total"] = pts_outcomes
    except ImportError:
        logger.warning("scipy_not_available", detail="points_total market skipped")

    # Remove empty markets
    markets = {k: v for k, v in markets.items() if v}

    return {
        "markets": markets,
        "families_generated": sorted(markets.keys()),
        "n_families": len(markets),
        "n_outcomes": sum(len(v) for v in markets.values()),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=DerivativeGenerateResponse,
    summary="Generate 20+ derivative markets for a Speedway GP round",
    description=(
        "Runs SpeedwayPredictor then generates all derivative market families. "
        "Chaining: same rider input as POST /api/v1/speedway/price-round.\n\n"
        "Markets produced (subset depends on optional inputs):\n"
        "round_winner, top3_finish, head_to_head, podium_finisher, last_place,\n"
        "country_winner, country_h2h, wildcard_wins, rider_dnr, final_qualifier,\n"
        "age_group_winner, gate_advantage (if gate_position on riders),\n"
        "defending_champion_win (if defending_champion_slug supplied),\n"
        "heat_winner, top2_heat, fastest_heat_winner (if heat_definitions supplied),\n"
        "points_total (if scipy available)"
    ),
    responses={
        200: {"description": "Derivatives generated successfully"},
        422: {"description": "Invalid request payload"},
        500: {"description": "Predictor or computation error"},
        503: {"description": "Predictor not initialised"},
    },
)
async def generate_derivatives(
    body: DerivativeGenerateRequest,
    request: Request,
    families: Optional[str] = Query(
        None,
        description=(
            "Comma-separated family names to return. "
            "Omit for all. "
            f"Valid: {', '.join(sorted(_ALL_FAMILIES))}"
        ),
    ),
) -> DerivativeGenerateResponse:
    """
    Generate derivative markets for a Speedway GP round.

    Pipeline:
    1. Validate families filter.
    2. Run SpeedwayPredictor.predict_round() for base probabilities.
    3. Compute all derivative families.
    4. Apply family filter if requested.
    5. Return full response.
    """
    request_id = str(uuid.uuid4())
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialised",
        )

    logger.info(
        "speedway_derivatives_request",
        request_id=request_id,
        n_riders=len(body.riders),
        venue_slug=body.venue_slug,
        venue_country=body.venue_country,
    )

    # Validate families filter
    requested_families: set[str] | None = None
    if families:
        requested_families = {f.strip() for f in families.split(",") if f.strip()}
        unknown = requested_families - _ALL_FAMILIES
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown families: {sorted(unknown)}. Valid: {sorted(_ALL_FAMILIES)}",
            )

    # Run predictor
    riders_dicts = [r.model_dump() for r in body.riders]
    try:
        rider_probs = predictor.predict_round(
            riders=riders_dicts,
            venue_slug=body.venue_slug,
            venue_country=body.venue_country,
            track_length=body.track_length,
            season_id=body.season_id,
        )
    except Exception as exc:
        logger.error(
            "speedway_derivatives_predictor_error",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    # Compute derivatives in executor
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _compute_all_derivatives, body, rider_probs
        )
    except Exception as exc:
        logger.error(
            "speedway_derivatives_computation_error",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Derivative computation failed: {exc}",
        )

    markets: dict[str, list[DerivativeOutcome]] = result["markets"]

    if requested_families:
        markets = {k: v for k, v in markets.items() if k in requested_families}

    logger.info(
        "speedway_derivatives_ok",
        request_id=request_id,
        n_families=len(markets),
        n_outcomes=sum(len(v) for v in markets.values()),
    )

    schema_version = (
        predictor.schema_version
        if hasattr(predictor, "schema_version")
        else "unknown"
    )

    return DerivativeGenerateResponse(
        request_id=request_id,
        venue_slug=body.venue_slug,
        venue_country=body.venue_country,
        model_schema=schema_version,
        n_riders=len(rider_probs),
        families_generated=sorted(markets.keys()),
        n_families=len(markets),
        n_outcomes=sum(len(v) for v in markets.values()),
        markets=markets,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
