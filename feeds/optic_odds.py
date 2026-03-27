"""
XG3 Speedway GP — Optic Odds feed adapter.

Fetches live Speedway fixtures from Optic Odds API
(sport: motorsports, category: speedway or speedway_gp).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPTIC_ODDS_BASE = "https://api.opticodds.com/api/v3"
SPEEDWAY_SPORT = "motorsports"
SPEEDWAY_LEAGUE_CANDIDATES = ["speedway_gp", "speedway", "fim_speedway_gp"]


class OpticOddsSpeedwayFeed:
    """Adapter for fetching Speedway GP fixtures and odds from Optic Odds."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("OPTIC_ODDS_API_KEY is required")
        self._api_key = api_key
        self._headers = {"x-api-key": api_key}

    async def fetch_fixtures(
        self,
        league_slug: Optional[str] = None,
    ) -> list[dict]:
        """Fetch upcoming Speedway GP fixtures."""
        league = league_slug or SPEEDWAY_LEAGUE_CANDIDATES[0]
        url = f"{OPTIC_ODDS_BASE}/fixtures"
        params = {
            "sport": SPEEDWAY_SPORT,
            "league": league,
            "is_live": False,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(url, headers=self._headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                fixtures = data.get("data", [])
                logger.info("Fetched %d Speedway fixtures from Optic Odds", len(fixtures))
                return fixtures
            except httpx.HTTPStatusError as e:
                logger.error("Optic Odds HTTP error: %s", e)
                raise
            except Exception as e:
                logger.error("Optic Odds fetch error: %s", e)
                raise

    async def fetch_odds(self, fixture_id: str) -> dict:
        """Fetch odds for a specific fixture."""
        url = f"{OPTIC_ODDS_BASE}/fixtures/{fixture_id}/odds"
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(url, headers=self._headers)
                resp.raise_for_status()
                return resp.json().get("data", {})
            except Exception as e:
                logger.error("Optic Odds odds fetch error for %s: %s", fixture_id, e)
                raise

    async def fetch_leagues(self) -> list[dict]:
        """Fetch all available leagues for motorsports sport."""
        url = f"{OPTIC_ODDS_BASE}/leagues"
        params = {"sport": SPEEDWAY_SPORT}
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(url, headers=self._headers, params=params)
                resp.raise_for_status()
                return resp.json().get("data", [])
            except Exception as e:
                logger.error("Optic Odds leagues fetch error: %s", e)
                raise
