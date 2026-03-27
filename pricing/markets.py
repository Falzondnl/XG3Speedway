"""
XG3 Speedway GP — Market Pricing.

Markets generated:
1. Round Winner (outright) — Harville win_prob + 5% margin
2. Top-3 Finish            — Harville DP finish probabilities + 5% margin
3. Head-to-Head (any two riders)

Harville DP formula:
  P(i finishes k-th | riders S) = sum over all (k-1)-subsets T of S without {i}:
      prod_{j in T} P(j wins S minus T_prev) * P(i wins S minus T)

For efficiency we compute via recursive DP up to top-3.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Optional


def _apply_margin(prob: float, margin: float) -> float:
    """Convert fair probability to book price with overround margin."""
    if prob <= 0.0:
        return 9999.0
    book_prob = prob * (1.0 + margin)
    book_prob = min(book_prob, 0.9999)
    return round(1.0 / book_prob, 3)


def harville_top3(
    win_probs: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute P(1st), P(2nd), P(3rd) for each position using Harville DP.

    win_probs: list of winning probabilities (already normalized to sum 1.0)
    Returns three lists of same length: p1, p2, p3 per rider.
    """
    n = len(win_probs)
    wp = [max(p, 1e-9) for p in win_probs]
    s = sum(wp)
    wp = [p / s for p in wp]  # re-normalize

    p1 = list(wp)

    # P(i 2nd) = sum_{j != i} P(j 1st) * wp[i] / (1 - wp[j])
    p2 = []
    for i in range(n):
        val = 0.0
        for j in range(n):
            if j == i:
                continue
            denom = 1.0 - wp[j]
            if denom > 1e-9:
                val += wp[j] * (wp[i] / denom)
        p2.append(val)

    # P(i 3rd) = sum_{j,k distinct, j!=i, k!=i, j!=k}
    #              P(j 1st) * P(k 2nd | j 1st) * wp[i] / (1 - wp[j] - wp[k])
    p3 = []
    for i in range(n):
        val = 0.0
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                denom_k = 1.0 - wp[j]
                if denom_k <= 1e-9:
                    continue
                p_k_given_j = wp[k] / denom_k
                denom_i = 1.0 - wp[j] - wp[k]
                if denom_i <= 1e-9:
                    continue
                p_i = wp[i] / denom_i
                val += wp[j] * p_k_given_j * p_i
        p3.append(val)

    return p1, p2, p3


def price_round_winner(
    rider_probs: list[dict],
    margin: float = 0.05,
) -> list[dict]:
    """
    Price the Round Winner market.

    rider_probs: output from SpeedwayPredictor.predict_round()
    Returns list of {slug, first_name, last_name, win_prob, decimal_odds}
    """
    results = []
    for r in rider_probs:
        win_prob = r.get("win_prob", 0.0)
        odds = _apply_margin(win_prob, margin)
        results.append({
            "slug": r["slug"],
            "first_name": r.get("first_name", ""),
            "last_name": r.get("last_name", ""),
            "win_prob": round(win_prob, 6),
            "decimal_odds": odds,
            "market": "round_winner",
        })
    results.sort(key=lambda x: x["decimal_odds"])
    return results


def price_top3_finish(
    rider_probs: list[dict],
    margin: float = 0.05,
) -> list[dict]:
    """
    Price Top-3 Finish market via Harville DP.
    """
    win_probs = [r.get("win_prob", 0.0) for r in rider_probs]
    p1_list, p2_list, p3_list = harville_top3(win_probs)

    results = []
    for i, r in enumerate(rider_probs):
        top3_prob = p1_list[i] + p2_list[i] + p3_list[i]
        top3_prob = min(top3_prob, 0.9999)
        odds = _apply_margin(top3_prob, margin)
        results.append({
            "slug": r["slug"],
            "first_name": r.get("first_name", ""),
            "last_name": r.get("last_name", ""),
            "p_1st": round(p1_list[i], 6),
            "p_2nd": round(p2_list[i], 6),
            "p_3rd": round(p3_list[i], 6),
            "top3_prob": round(top3_prob, 6),
            "decimal_odds": odds,
            "market": "top3_finish",
        })
    results.sort(key=lambda x: x["decimal_odds"])
    return results


def price_h2h(
    rider_a_slug: str,
    rider_b_slug: str,
    rider_probs: list[dict],
    margin: float = 0.05,
) -> Optional[dict]:
    """
    Price head-to-head between two riders to finish higher in the round.
    Uses normalized win probabilities as a proxy for H2H.
    """
    prob_a = None
    prob_b = None
    for r in rider_probs:
        if r["slug"] == rider_a_slug:
            prob_a = r.get("win_prob", 0.0)
        if r["slug"] == rider_b_slug:
            prob_b = r.get("win_prob", 0.0)

    if prob_a is None or prob_b is None:
        return None

    total = prob_a + prob_b
    if total <= 0:
        return None

    p_a_wins = prob_a / total
    p_b_wins = prob_b / total

    return {
        "rider_a": rider_a_slug,
        "rider_b": rider_b_slug,
        "p_a_wins": round(p_a_wins, 6),
        "p_b_wins": round(p_b_wins, 6),
        "odds_a": _apply_margin(p_a_wins, margin),
        "odds_b": _apply_margin(p_b_wins, margin),
        "market": "h2h",
    }


def price_heat_winner(
    heat_riders: list[dict],
    all_round_probs: list[dict],
    margin: float = 0.05,
) -> list[dict]:
    """
    Price winner of a single heat (4 riders) using round-level win probs
    as strength indicators, then Harville-normalize within the heat.
    """
    heat_slugs = {r.get("slug") for r in heat_riders}
    heat_probs = [r for r in all_round_probs if r.get("slug") in heat_slugs]

    total = sum(r.get("win_prob", 0.0) for r in heat_probs)
    if total <= 0:
        return []

    results = []
    for r in heat_probs:
        p = r.get("win_prob", 0.0) / total
        odds = _apply_margin(p, margin)
        results.append({
            "slug": r["slug"],
            "first_name": r.get("first_name", ""),
            "last_name": r.get("last_name", ""),
            "heat_win_prob": round(p, 6),
            "decimal_odds": odds,
            "market": "heat_winner",
        })
    results.sort(key=lambda x: x["decimal_odds"])
    return results
