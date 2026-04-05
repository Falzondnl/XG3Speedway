"""
XG3 Speedway GP Microservice — Main entry point.

Port: 8036 (local dev) / $PORT (Railway)
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---- logging setup --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup; release resources on shutdown."""
    from config import get_settings
    from ml.predictor import SpeedwayPredictor

    settings = get_settings()
    models_dir = settings.models_dir

    logger.info("=== XG3 Speedway GP starting up ===")
    logger.info("Models dir: %s", models_dir)

    predictor = SpeedwayPredictor()
    ensemble_pkl = Path(models_dir) / "r0" / "ensemble.pkl"

    if ensemble_pkl.exists():
        try:
            predictor.load(str(Path(models_dir) / "r0"))
            logger.info("Models loaded successfully — schema: %s", predictor.schema_version)
        except Exception as exc:
            logger.error("Model load failed: %s — service will use ELO fallback", exc)
    else:
        logger.warning(
            "No ensemble.pkl found at %s — serving in ELO-only mode. "
            "Run: python train.py to train the models.",
            ensemble_pkl,
        )

    app.state.predictor = predictor
    logger.info("=== XG3 Speedway GP ready ===")

    yield

    logger.info("=== XG3 Speedway GP shutting down ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    from config import get_settings

    settings = get_settings()

    app = FastAPI(
        title="XG3 Speedway GP",
        description="FIM Speedway GP round prediction & market pricing — Tier 1",
        version=settings.service_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Register routes --------------------------------------------------
    from api.routes.health import router as health_router
    from api.routes.races import router as races_router
    from api.routes.admin import router as admin_router
    from api.routes.settlement import router as settlement_router

    app.include_router(health_router)
    app.include_router(races_router)
    app.include_router(admin_router)
    app.include_router(settlement_router)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import get_settings

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
