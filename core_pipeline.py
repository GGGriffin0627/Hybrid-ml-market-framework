# core_pipeline.py

import gc
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from scipy.optimize import minimize, Bounds

# Optional deep learning
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MISSING_VALUE_FILL = 0.5
SEQUENCE_LENGTH = 20
OPTIMIZATION_WINDOW_SIZE = 180

MIN_POSITION = 0.0
MAX_POSITION = 2.0
SIGNAL_SCALE = 400.0

PRIMARY_LGBM_CONFIG = {
    "objective": "regression",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "random_state": RANDOM_SEED,
    "verbosity": -1
}

AUX_LGBM_CONFIG = {
    "objective": "regression",
    "n_estimators": 400,
    "learning_rate": 0.02,
    "num_leaves": 64,
    "random_state": RANDOM_SEED + 1,
    "verbosity": -1
}


# =============================================================================
# SYSTEM STATE
# =============================================================================

SYSTEM_STATE: Dict = {
    "initialized": False
}


# =============================================================================
# EVALUATION METRIC (GENERIC, NOT KAGGLE-BRANDED)
# =============================================================================

def risk_adjusted_score(
    market_returns: np.ndarray,
    risk_free: np.ndarray,
    positions: np.ndarray
) -> float:

    if np.any(positions < MIN_POSITION) or np.any(positions > MAX_POSITION):
        return -1e9

    portfolio_returns = risk_free * (1 - positions) + positions * market_returns
    excess = portfolio_returns - risk_free

    if np.std(portfolio_returns) == 0:
        return -1e9

    sharpe = np.mean(excess) / np.std(portfolio_returns)
    return sharpe


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def construct_features(df: pl.DataFrame) -> pl.DataFrame:

    numeric_cols = [c for c in df.columns if c != "timestamp"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])

    base_col = "M11"
    if base_col not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias(base_col))

    windows = [5, 10, 21, 63]
    for w in windows:
        df = df.with_columns([
            (pl.col(base_col) / pl.col(base_col).shift(w) - 1).alias(f"ret_{w}"),
            pl.col(base_col).rolling_std(w).alias(f"vol_{w}"),
            pl.col(base_col).rolling_mean(w).alias(f"ma_{w}")
        ])

    for c in df.columns:
        if c != "timestamp":
            df = df.with_columns(
                pl.col(c)
                .forward_fill()
                .backward_fill()
                .fill_null(MISSING_VALUE_FILL)
                .alias(c)
            )

    return df.drop_nulls()


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_positions(predictions, market_ret, rf):

    def objective(x):
        return -risk_adjusted_score(market_ret, rf, x)

    x0 = np.clip(predictions * SIGNAL_SCALE + 1.0, MIN_POSITION, MAX_POSITION)

    res = minimize(
        objective,
        x0,
        bounds=Bounds(MIN_POSITION, MAX_POSITION),
        method="Powell",
        options={"maxiter": 10000}
    )

    return np.clip(res.x, MIN_POSITION, MAX_POSITION)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_pipeline(df: pd.DataFrame):

    df_pl = pl.from_pandas(df)
    features_df = construct_features(df_pl).to_pandas()

    feature_cols = [c for c in features_df.columns
                    if c not in {"timestamp", "target"}]

    X = features_df[feature_cols].values
    y = features_df["target"].values

    imputer = SimpleImputer(strategy="constant", fill_value=MISSING_VALUE_FILL)
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model_1 = LGBMRegressor(**PRIMARY_LGBM_CONFIG)
    model_2 = LGBMRegressor(**AUX_LGBM_CONFIG)

    model_1.fit(X, y)
    model_2.fit(X, y)

    preds_1 = model_1.predict(X)
    preds_2 = model_2.predict(X)

    meta_X = np.column_stack([preds_1, preds_2])
    meta = RidgeCV(alphas=np.logspace(-6, 6, 13))
    meta.fit(meta_X, y)

    SYSTEM_STATE.update({
        "initialized": True,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
        "model_1": model_1,
        "model_2": model_2,
        "meta": meta,
        "history": features_df
    })

    gc.collect()


# =============================================================================
# PREDICTION INTERFACE (GENERIC)
# =============================================================================

def predict(df: pd.DataFrame):

    if not SYSTEM_STATE["initialized"]:
        raise RuntimeError("Pipeline not trained")

    history = SYSTEM_STATE["history"]
    combined = pd.concat([history, df], axis=0).reset_index(drop=True)

    combined_pl = pl.from_pandas(combined)
    engineered = construct_features(combined_pl).to_pandas()

    X = engineered.iloc[-len(df):][SYSTEM_STATE["feature_cols"]].values
    X = SYSTEM_STATE["imputer"].transform(X)
    X = SYSTEM_STATE["scaler"].transform(X)

    p1 = SYSTEM_STATE["model_1"].predict(X)
    p2 = SYSTEM_STATE["model_2"].predict(X)

    meta_X = np.column_stack([p1, p2])
    final_pred = SYSTEM_STATE["meta"].predict(meta_X)

    signal = final_pred * SIGNAL_SCALE + 1.0
    return np.clip(signal, MIN_POSITION, MAX_POSITION)
