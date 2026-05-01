"""
XG3 Speedway GP — Predictor.

Loads a trained ensemble from disk and provides round-winner probability
estimates per rider.  Applies Harville DP normalization so probabilities
across riders in a round sum to 1.0 (round-winner market).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.calibrator import BetaCalibrator
from ml.elo_seed import normalise_slug, patch_predictor_elo
from ml.features import FEATURE_COLUMNS, SpeedwayELO

logger = logging.getLogger(__name__)


class SpeedwayPredictor:
    """Inference wrapper around the trained stacking ensemble."""

    def __init__(self) -> None:
        self._ensemble: Optional[dict] = None
        self._calibrator: Optional[BetaCalibrator] = None
        self._elo: SpeedwayELO = SpeedwayELO()
        self._rider_history: dict[str, dict] = {}
        self._loaded = False

    # -----------------------------------------------------------------------
    def load(self, models_dir: str = "models/r0") -> "SpeedwayPredictor":
        path = Path(models_dir)

        ens_path = path / "ensemble.pkl"
        if not ens_path.exists():
            raise FileNotFoundError(f"ensemble.pkl not found at {ens_path}")
        with open(ens_path, "rb") as f:
            self._ensemble = pickle.load(f)

        cal_path = path / "calibrator.pkl"
        if cal_path.exists():
            self._calibrator = BetaCalibrator.load(cal_path)
        else:
            logger.warning("calibrator.pkl not found — uncalibrated probabilities will be used")

        # Load feature extractor state (carries ELO + rider history)
        ext_path = path / "extractor.pkl"
        if ext_path.exists():
            with open(ext_path, "rb") as f:
                extractor = pickle.load(f)
            self._elo = getattr(extractor, "_elo", SpeedwayELO())
            self._rider_history = getattr(extractor, "_rider_history", {})
        else:
            logger.warning("extractor.pkl not found — fresh ELO/history state")

        self._loaded = True
        feature_cols = self._ensemble.get("feature_columns", FEATURE_COLUMNS)
        logger.info("SpeedwayPredictor loaded from %s (%d features, schema=%s)",
                    models_dir, len(feature_cols),
                    self._ensemble.get("schema_version", "unknown"))

        # Patch ELO registry: rename corrupted source slugs (e.g. "-1") to
        # their canonical forms (e.g. "tai-woffinden") so inference lookups
        # resolve correctly.  This is idempotent and safe to call every load.
        patch_predictor_elo(self)

        return self

    # -----------------------------------------------------------------------
    def predict_round(
        self,
        riders: list[dict],
        venue_slug: Optional[str] = None,
        venue_country: Optional[str] = None,
        track_length: Optional[float] = None,
        season_id: int = 0,
    ) -> list[dict]:
        """
        Given a list of rider dicts with at minimum {'slug': str},
        return a list of {'slug', 'raw_prob', 'calibrated_prob', 'win_prob'}
        where win_prob is Harville-normalized so all sum to 1.0.

        If models are not loaded, fall back to ELO-based Harville DP.
        """
        if not self._loaded:
            return self._elo_harville_fallback(riders, venue_country)

        feature_cols = self._ensemble.get("feature_columns", FEATURE_COLUMNS)
        rows = []
        for r in riders:
            raw_slug = r.get("slug", "")
            # Normalise slug: maps corrupted source values (e.g. "-1") to their
            # canonical form (e.g. "tai-woffinden") before ELO / history lookup.
            slug = normalise_slug(raw_slug)
            hist = self._rider_history.get(slug, {
                "heat_pts": [], "round_ranks": [], "round_wins": 0,
                "venues": {}, "season_round_pts": {}, "n_rounds": 0,
            })
            elo_rating = self._elo.get(slug)

            # Build feature row matching training schema
            from ml.features import SpeedwayFeatureExtractor
            extractor = SpeedwayFeatureExtractor()
            rider_series = pd.Series({
                "rider_country": r.get("country_code", ""),
                "is_wildcard": r.get("is_wildcard", False),
                "is_substitute": r.get("is_substitute", False),
                "round_starts_at": r.get("round_date"),
            })
            static = {
                "sgpWins": r.get("sgp_wins"),
                "appearances": r.get("appearances"),
                "heatsWon": r.get("heats_won"),
                "heatsRaced": r.get("heats_raced"),
                "finals": r.get("finals"),
                "wins": r.get("wins"),
                "fimRank": r.get("fim_rank"),
                "bornAt": r.get("born_at"),
            }
            feat = extractor._compute_features(
                slug=slug,
                hist=hist,
                elo_rating=elo_rating,
                venue_slug=venue_slug,
                venue_country=venue_country,
                track_length=track_length if track_length else np.nan,
                rider_row=rider_series,
                static=static,
                season_id=season_id,
            )
            rows.append(feat)

        X = pd.DataFrame(rows)[feature_cols].apply(pd.to_numeric, errors="coerce")
        raw_probs = self._run_ensemble(X)

        if self._calibrator:
            cal_probs = self._calibrator.predict(raw_probs)
        else:
            cal_probs = raw_probs

        # Harville normalization
        cal_probs = np.clip(cal_probs, 1e-6, 1.0)
        win_probs = cal_probs / cal_probs.sum()

        results = []
        for i, r in enumerate(riders):
            results.append({
                "slug": normalise_slug(r.get("slug", "")),
                "first_name": r.get("first_name", ""),
                "last_name": r.get("last_name", ""),
                "raw_prob": round(float(raw_probs[i]), 6),
                "calibrated_prob": round(float(cal_probs[i]), 6),
                "win_prob": round(float(win_probs[i]), 6),
            })
        return results

    # -----------------------------------------------------------------------
    def _run_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        cb = self._ensemble["cb"]
        lgb_m = self._ensemble["lgb"]
        xgb_m = self._ensemble["xgb"]
        meta = self._ensemble["meta"]

        p_cb = cb.predict_proba(X)[:, 1]
        p_lgb = lgb_m.predict_proba(X)[:, 1]
        p_xgb = xgb_m.predict_proba(X)[:, 1]
        stack = np.column_stack([p_cb, p_lgb, p_xgb])
        return meta.predict_proba(stack)[:, 1]

    # -----------------------------------------------------------------------
    def _elo_harville_fallback(
        self, riders: list[dict], venue_country: Optional[str]
    ) -> list[dict]:
        """Pure ELO-based fallback when models not loaded."""
        # Normalise slugs before ELO lookup (fixes corrupted source slugs)
        slugs = [normalise_slug(r.get("slug", "")) for r in riders]
        elo_ratings = np.array([self._elo.get(s) for s in slugs], dtype=float)
        # Convert to win probability via Harville formula
        strengths = np.exp(elo_ratings / 400.0)
        win_probs = strengths / strengths.sum()

        results = []
        for i, r in enumerate(riders):
            results.append({
                "slug": slugs[i],  # return canonical slug
                "first_name": r.get("first_name", ""),
                "last_name": r.get("last_name", ""),
                "raw_prob": round(float(win_probs[i]), 6),
                "calibrated_prob": round(float(win_probs[i]), 6),
                "win_prob": round(float(win_probs[i]), 6),
            })
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def schema_version(self) -> str:
        if self._ensemble:
            return self._ensemble.get("schema_version", "unknown")
        return "not_loaded"
