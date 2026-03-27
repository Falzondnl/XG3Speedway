"""
XG3 Speedway GP — Beta Calibrator.

Fits a BetaCalibrator (isotonic-inspired monotone transform) on
held-out probabilities to correct systematic over/under-confidence.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class BetaCalibrator:
    """Wrapper around sklearn IsotonicRegression for probability calibration."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        self._iso.fit(probs, labels)
        self._fitted = True
        logger.info("BetaCalibrator fitted on %d samples", len(probs))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BetaCalibrator not fitted — call fit() first")
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        return self._iso.predict(probs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"iso": self._iso, "fitted": self._fitted}, f)
        logger.info("BetaCalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BetaCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal = cls()
        cal._iso = data["iso"]
        cal._fitted = data["fitted"]
        logger.info("BetaCalibrator loaded from %s", path)
        return cal
