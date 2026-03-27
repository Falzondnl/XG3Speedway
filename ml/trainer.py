"""
XG3 Speedway GP — Model Trainer.

Trains a 3-model stacking ensemble:
  - CatBoost (gradient boosted trees, handles NaN natively)
  - LightGBM  (fast gradient boosting)
  - XGBoost   (gradient boosting with regularization)

Meta-learner: LogisticRegression on OOF predictions.

Temporal split:
  - train:  rounds before 2020
  - val:    2020-2022
  - test:   2023+

Anti-positional-bias: riders are shuffled within each round during training.
GroupKFold by round_id for OOF generation.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError("catboost is required: pip install catboost>=1.2.0") from e

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm is required: pip install lightgbm>=4.3.0") from e

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("xgboost is required: pip install xgboost>=2.0.0") from e

from ml.calibrator import BetaCalibrator
from ml.features import FEATURE_COLUMNS, SpeedwayFeatureExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_BEFORE_YEAR = 2020
VAL_YEARS = (2020, 2022)
TEST_FROM_YEAR = 2023


def _year(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return 0
    return int(pd.Timestamp(ts).year)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_and_save(
    data_dir: str = "D:/codex/Data/motorsports/tier1/curated/speedway",
    models_dir: str = "models/r0",
) -> dict:
    """
    Full training pipeline.  Returns metrics dict.
    """
    logger.info("Loading CSVs from %s ...", data_dir)
    data_dir_path = Path(data_dir)

    round_results = pd.read_csv(data_dir_path / "round_result_rankings.csv", low_memory=False)
    heat_rankings = pd.read_csv(data_dir_path / "heat_rankings.csv", low_memory=False)
    round_details = pd.read_csv(data_dir_path / "round_details.csv", low_memory=False)
    riders = pd.read_csv(data_dir_path / "riders.csv", low_memory=False)

    logger.info("round_results: %d rows, heat_rankings: %d rows", len(round_results), len(heat_rankings))

    # ---- Feature extraction -----------------------------------------------
    extractor = SpeedwayFeatureExtractor()
    df = extractor.fit_transform(round_results, heat_rankings, round_details, riders)

    df["round_year"] = df["round_starts_at"].apply(_year)
    df = df[df["round_year"] > 0].copy()

    # ---- Temporal splits --------------------------------------------------
    train_mask = df["round_year"] < TRAIN_BEFORE_YEAR
    val_mask = (df["round_year"] >= VAL_YEARS[0]) & (df["round_year"] <= VAL_YEARS[1])
    test_mask = df["round_year"] >= TEST_FROM_YEAR

    X_train = df.loc[train_mask, FEATURE_COLUMNS].copy()
    y_train = df.loc[train_mask, "is_winner"].values
    groups_train = df.loc[train_mask, "round_id"].values

    X_val = df.loc[val_mask, FEATURE_COLUMNS].copy()
    y_val = df.loc[val_mask, "is_winner"].values

    X_test = df.loc[test_mask, FEATURE_COLUMNS].copy()
    y_test = df.loc[test_mask, "is_winner"].values

    logger.info("Split sizes — train: %d (pos=%d), val: %d, test: %d",
                len(X_train), int(y_train.sum()),
                len(X_val), len(X_test))

    # Shuffle rows within train to prevent any positional leakage
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    X_train = X_train.iloc[idx].reset_index(drop=True)
    y_train = y_train[idx]
    groups_train = groups_train[idx]

    # ---- Base learners ----------------------------------------------------
    cb_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    # ---- OOF predictions for meta-learner ---------------------------------
    kfold = GroupKFold(n_splits=5)
    oof_cb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))

    logger.info("Generating OOF predictions via GroupKFold ...")
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train, groups_train)):
        logger.info("  Fold %d/%d ...", fold + 1, 5)
        Xtr, Xvl = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train[tr_idx]

        cb_model.fit(Xtr, ytr)
        oof_cb[val_idx] = cb_model.predict_proba(Xvl)[:, 1]

        lgb_model.fit(Xtr, ytr)
        oof_lgb[val_idx] = lgb_model.predict_proba(Xvl)[:, 1]

        xgb_model.fit(Xtr, ytr)
        oof_xgb[val_idx] = xgb_model.predict_proba(Xvl)[:, 1]

    # OOF AUC per model
    oof_auc_cb = roc_auc_score(y_train, oof_cb)
    oof_auc_lgb = roc_auc_score(y_train, oof_lgb)
    oof_auc_xgb = roc_auc_score(y_train, oof_xgb)
    logger.info("OOF AUC — CB: %.4f  LGB: %.4f  XGB: %.4f",
                oof_auc_cb, oof_auc_lgb, oof_auc_xgb)

    # ---- Retrain base models on full train set ----------------------------
    logger.info("Retraining base models on full training set ...")
    cb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # ---- Meta-learner (stacking) -----------------------------------------
    oof_stack = np.column_stack([oof_cb, oof_lgb, oof_xgb])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(oof_stack, y_train)
    logger.info("Meta-learner coefficients: CB=%.3f LGB=%.3f XGB=%.3f",
                meta.coef_[0][0], meta.coef_[0][1], meta.coef_[0][2])

    # ---- Val / test evaluation --------------------------------------------
    def ensemble_proba(X: pd.DataFrame) -> np.ndarray:
        p_cb = cb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        stack = np.column_stack([p_cb, p_lgb, p_xgb])
        return meta.predict_proba(stack)[:, 1]

    val_proba = ensemble_proba(X_val)
    test_proba = ensemble_proba(X_test)

    val_auc = roc_auc_score(y_val, val_proba) if y_val.sum() > 0 else float("nan")
    val_brier = brier_score_loss(y_val, val_proba) if len(y_val) > 0 else float("nan")
    test_auc = roc_auc_score(y_test, test_proba) if y_test.sum() > 0 else float("nan")
    test_brier = brier_score_loss(y_test, test_proba) if len(y_test) > 0 else float("nan")

    logger.info("Val  — AUC: %.4f  Brier: %.4f", val_auc, val_brier)
    logger.info("Test — AUC: %.4f  Brier: %.4f", test_auc, test_brier)

    # ---- Calibrator on val set --------------------------------------------
    calibrator = BetaCalibrator()
    calibrator.fit(val_proba, y_val)

    # ---- Save artefacts ---------------------------------------------------
    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)

    ensemble_payload = {
        "cb": cb_model,
        "lgb": lgb_model,
        "xgb": xgb_model,
        "meta": meta,
        "feature_columns": FEATURE_COLUMNS,
        "schema_version": "speedway_r0_v1_20260327",
    }
    ens_path = out / "ensemble.pkl"
    with open(ens_path, "wb") as f:
        pickle.dump(ensemble_payload, f)

    calibrator.save(out / "calibrator.pkl")

    # Save extractor (with ELO state from training run, used for incremental inference)
    ext_path = out / "extractor.pkl"
    with open(ext_path, "wb") as f:
        pickle.dump(extractor, f)

    ens_size = ens_path.stat().st_size / 1024 / 1024
    logger.info("Saved ensemble.pkl (%.1f MB) to %s", ens_size, out)

    metrics = {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_positives": int(y_train.sum()),
        "oof_auc_catboost": round(oof_auc_cb, 4),
        "oof_auc_lightgbm": round(oof_auc_lgb, 4),
        "oof_auc_xgboost": round(oof_auc_xgb, 4),
        "val_auc": round(val_auc, 4),
        "val_brier": round(val_brier, 4),
        "test_auc": round(test_auc, 4),
        "test_brier": round(test_brier, 4),
        "ensemble_pkl_mb": round(ens_size, 2),
        "models_dir": str(out),
        "schema_version": "speedway_r0_v1_20260327",
    }
    logger.info("Training complete: %s", metrics)
    return metrics
