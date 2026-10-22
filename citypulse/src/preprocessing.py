"""
preprocessing.py - Data Preprocessing and Feature Engineering Pipeline
Constructs leakage-free temporal, kinematic, and spatial features strictly from t <= t0.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "speed_current",
    "speed_mean_15m",
    "speed_mean_30m",
    "speed_mean_60m",
    "speed_trend_15m",
    "speed_trend_30m",
    "speed_std_30m",
    "day_fraction",
    "hour_of_day",
    "is_peak_morning",
    "is_peak_evening",
    "is_weekend",
    "node_id"
]

NUMERICAL_COLS = [
    "speed_current",
    "speed_mean_15m",
    "speed_mean_30m",
    "speed_mean_60m",
    "speed_trend_15m",
    "speed_trend_30m",
    "speed_std_30m",
    "day_fraction",
    "hour_of_day"
]

PASSTHROUGH_COLS = [
    "is_peak_morning",
    "is_peak_evening",
    "is_weekend",
    "node_id"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives kinematic and temporal traffic features strictly from columns at t <= t0.
    Guarantees zero future target leakage.
    """
    feat = pd.DataFrame(index=df.index)
    
    # 1. Kinematic Traffic Features
    feat["speed_current"] = df["x_t+0_d0"]
    
    # Rolling means
    feat["speed_mean_15m"] = df[["x_t-3_d0", "x_t-2_d0", "x_t-1_d0", "x_t+0_d0"]].mean(axis=1)
    past_6 = [f"x_t-{i}_d0" for i in range(1, 7)] + ["x_t+0_d0"]
    feat["speed_mean_30m"] = df[past_6].mean(axis=1)
    past_12 = [f"x_t-{i}_d0" for i in range(1, 12)] + ["x_t+0_d0"]
    feat["speed_mean_60m"] = df[past_12].mean(axis=1)
    
    # Speed trends (acceleration / deceleration)
    feat["speed_trend_15m"] = df["x_t+0_d0"] - df["x_t-3_d0"]
    feat["speed_trend_30m"] = df["x_t+0_d0"] - df["x_t-6_d0"]
    
    # Speed volatility (standard deviation over past 30 min)
    feat["speed_std_30m"] = df[past_6].std(axis=1).fillna(0.0)
    
    # 2. Temporal Features
    feat["day_fraction"] = df["x_t+0_d1"]
    
    # Compute hour of day from timestamp if available, else from normalized day_fraction
    if "t0_timestamp" in df.columns:
        ts = pd.to_datetime(df["t0_timestamp"])
        feat["hour_of_day"] = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek
        feat["is_weekend"] = (dow >= 5).astype(int)
    else:
        feat["hour_of_day"] = (df["x_t+0_d1"] * 24.0).clip(0.0, 23.99)
        feat["is_weekend"] = 0
        
    feat["is_peak_morning"] = ((feat["hour_of_day"] >= 7.0) & (feat["hour_of_day"] <= 10.0)).astype(int)
    feat["is_peak_evening"] = ((feat["hour_of_day"] >= 16.0) & (feat["hour_of_day"] <= 19.0)).astype(int)
    
    feat["node_id"] = df["node_id"].astype(int)
    
    return feat[FEATURE_COLS]

def create_targets(df: pd.DataFrame, horizon_step: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Creates continuous speed target and discrete congestion classification target.
    horizon_step=3 corresponds to t+15min (3 * 5min).
    
    Congestion Classes:
      0: Low Congestion / Free-Flow (Speed >= 55.0 mph)
      1: Moderate Congestion (35.0 <= Speed < 55.0 mph)
      2: High Congestion / Breakdown (Speed < 35.0 mph)
    """
    target_col = f"y_t+{horizon_step}_d0"
    y_reg = df[target_col].copy()
    
    # Classification labels:
    # 0 = Low, 1 = Moderate, 2 = High
    y_clf = pd.Series(0, index=df.index, dtype=int)
    y_clf[y_reg < 55.0] = 1
    y_clf[y_reg < 35.0] = 2
    
    return y_clf, y_reg

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that scales numerical kinematic features
    and passes through binary/categorical node indicators.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS)
        ],
        remainder="drop"
    )
    return transformer
