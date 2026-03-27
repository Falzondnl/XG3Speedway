"""XG3 Speedway GP Microservice — Configuration."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "xg3-speedway"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    models_dir: str = "models"
    optic_odds_api_key: str = os.getenv("OPTIC_ODDS_API_KEY", "")
    optic_odds_base_url: str = "https://api.opticodds.com/api/v3"

    # ELO settings
    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1500.0

    # Market margin
    win_margin_pct: float = 0.05
    top3_margin_pct: float = 0.05
    heat_win_margin_pct: float = 0.05

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
