"""
XG3 Speedway GP — ELO Slug Alias & Seed Patcher.

Root cause (confirmed 2026-05-01):
  The FIM Speedway GP API returns object.slug = "-1" for Tai Woffinden across
  ALL of his heat and round-result rows in the source CSVs.  The SpeedwayELO
  registry therefore stores his ELO under key "-1" (1570.4 after alias fix,
  was unreachable entirely before this patch).

  When callers POST slug="tai-woffinden" to /price-round, self._elo.get() falls
  through to the 1500.0 default, giving Woffinden zero ELO advantage — producing
  near-zero win probability for a 3× world champion.

Fix layers implemented here:
  1. SLUG_ALIAS_MAP  — canonical "bad-slug" → "correct-slug" mapping applied at
     startup to patch the loaded extractor's ELO registry.
  2. patch_predictor_elo() — called from main.py lifespan after predictor.load()
     to rename bad keys and inject the corrected values into the live registry.
  3. CANONICAL_SLUGS — list used by features.py normalization at training time
     so future retrains also produce clean keys.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.predictor import SpeedwayPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slug alias map: bad-slug → canonical-slug
# Source: verified 2026-05-01 from riders.csv and round_result_rankings.csv
# "-1" appears 133× in round_result_rankings and 153× in heat_rankings,
# always with firstName="Tai", lastName="Woffinden".
# ---------------------------------------------------------------------------
SLUG_ALIAS_MAP: dict[str, str] = {
    "-1": "tai-woffinden",
}

# ---------------------------------------------------------------------------
# Reverse map for feature extraction: canonical-slug → set of source aliases
# Used in features.py _clean_* methods to normalise before ELO update.
# ---------------------------------------------------------------------------
SLUG_REVERSE_MAP: dict[str, list[str]] = {
    canonical: [bad for bad, c in SLUG_ALIAS_MAP.items() if c == canonical]
    for canonical in set(SLUG_ALIAS_MAP.values())
}

# ---------------------------------------------------------------------------
# Known top-rider slugs for validation / audit logging.
# All should resolve to non-1500 ELO after patch is applied.
# ---------------------------------------------------------------------------
TOP_RIDERS_CANONICAL: list[str] = [
    "tai-woffinden",
    "bartosz-zmarzlik",
    "jason-doyle",
    "leon-madsen",
    "fredrik-lindgren",
    "martin-vaculik",
    "patryk-dudek",
    "mikkel-michelsen",
    "maciej-janowski",
    "kai-huckenbeck",
]


# ---------------------------------------------------------------------------
# Patch function — applied at startup
# ---------------------------------------------------------------------------

def patch_predictor_elo(predictor: "SpeedwayPredictor") -> None:
    """
    Rename bad ELO registry keys to their canonical slugs in-place.

    Called from main.py lifespan immediately after predictor.load().
    Safe to call multiple times (idempotent).

    Side effects:
    - Removes the bad-slug entry from predictor._elo.ratings
    - Adds (or updates) the canonical-slug entry
    - Updates predictor._rider_history keys to canonical slugs
    - Logs the before/after ELO for every aliased rider
    """
    elo = predictor._elo
    history = predictor._rider_history

    for bad_slug, canonical_slug in SLUG_ALIAS_MAP.items():
        if bad_slug not in elo.ratings:
            # Already cleaned or never present — check if canonical exists
            if canonical_slug not in elo.ratings:
                logger.warning(
                    "elo_patch: neither %r nor %r found in ELO registry; "
                    "rider will default to %.0f at inference",
                    bad_slug, canonical_slug, elo.initial,
                )
            else:
                logger.info(
                    "elo_patch: %r already canonical (ELO=%.1f)",
                    canonical_slug, elo.ratings[canonical_slug],
                )
            continue

        bad_elo = elo.ratings.pop(bad_slug)
        # If canonical already exists (shouldn't happen), take the higher rating
        existing = elo.ratings.get(canonical_slug)
        if existing is not None:
            merged = max(existing, bad_elo)
            logger.info(
                "elo_patch: merged %r (%.1f) + %r (%.1f) → %r = %.1f",
                bad_slug, bad_elo, canonical_slug, existing, canonical_slug, merged,
            )
            elo.ratings[canonical_slug] = merged
        else:
            elo.ratings[canonical_slug] = bad_elo
            logger.info(
                "elo_patch: renamed ELO key %r → %r (%.1f)",
                bad_slug, canonical_slug, bad_elo,
            )

        # Rename rider_history key too (rolling form / venues)
        if bad_slug in history:
            hist_data = history.pop(bad_slug)
            if canonical_slug not in history:
                history[canonical_slug] = hist_data
                logger.info("elo_patch: renamed history key %r → %r", bad_slug, canonical_slug)
            else:
                logger.info(
                    "elo_patch: history for %r already present under canonical key; "
                    "keeping existing (longer history wins)",
                    canonical_slug,
                )

    # Audit log: verify top riders all have non-default ELO
    _audit_top_riders(elo)


def _audit_top_riders(elo: object) -> None:
    """Log ELO for all known top riders so mismatches are visible at startup."""
    ratings = elo.ratings  # type: ignore[attr-defined]
    initial = elo.initial  # type: ignore[attr-defined]
    issues = []
    for slug in TOP_RIDERS_CANONICAL:
        rating = ratings.get(slug, initial)
        status = "OK" if rating != initial else "DEFAULT_1500 — MISMATCH"
        logger.info("elo_audit: %-35s  ELO=%.1f  [%s]", slug, rating, status)
        if rating == initial:
            issues.append(slug)
    if issues:
        logger.error(
            "elo_patch: %d top riders still at default ELO after patch: %s. "
            "These riders will be undervalued at inference.",
            len(issues), issues,
        )
    else:
        logger.info("elo_patch: all %d top riders have real ELO ratings.", len(TOP_RIDERS_CANONICAL))


# ---------------------------------------------------------------------------
# Utility: normalise a slug at inference time (belt-and-suspenders)
# ---------------------------------------------------------------------------

def normalise_slug(slug: str) -> str:
    """
    Map a bad API slug to its canonical form.

    Called in predictor.predict_round() before ELO lookup so that if a caller
    accidentally passes the raw API slug ("-1") they still get the correct ELO.
    """
    return SLUG_ALIAS_MAP.get(slug, slug)
