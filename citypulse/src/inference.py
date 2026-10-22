"""
inference.py - Production Inference Service for CityPulse TrafficPulse AI
Runs model inference, attaches class probabilities, and packages uncertainty metadata.
"""

import os
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import engineer_features

_PREPROCESSOR = None
_CLASSIFIER = None
_FORECASTER = None

LABEL_MAP = {
    0: "Low Congestion (Free-Flow)",
    1: "Moderate Congestion",
    2: "High Congestion / Breakdown"
}

LABEL_SHORT = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

COLOR_MAP = {
    "Low": "#10B981",       # Green
    "Moderate": "#F59E0B",  # Amber / Yellow
    "High": "#EF4444"       # Red
}

from config import resolve_model_path

def load_artifacts():
    global _PREPROCESSOR, _CLASSIFIER, _FORECASTER
    if _PREPROCESSOR is None or _CLASSIFIER is None or _FORECASTER is None:
        try:
            _PREPROCESSOR = joblib.load(resolve_model_path("preprocessing_pipeline.joblib"))
            _CLASSIFIER = joblib.load(resolve_model_path("traffic_classifier.joblib"))
            _FORECASTER = joblib.load(resolve_model_path("traffic_forecaster.joblib"))
        except AttributeError as e:
            if "__pyx_unpickle" in str(e) or "sklearn._loss" in str(e):
                import sklearn
                raise RuntimeError(
                    f"Scikit-learn version mismatch detected! Currently installed scikit-learn is '{sklearn.__version__}', "
                    f"but model artifacts were serialized with scikit-learn '1.3.0'. "
                    f"Please ensure scikit-learn is pinned to 'scikit-learn==1.3.0' and 'numpy<2.0.0' in requirements.txt."
                ) from e
            raise
    return _PREPROCESSOR, _CLASSIFIER, _FORECASTER

def predict_network_state(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a network snapshot dataframe, derives features, and predicts
    congestion classification and continuous speed 15 minutes ahead.
    """
    preprocessor, classifier, forecaster = load_artifacts()
    
    # 1. Feature Engineering
    X_raw = engineer_features(snapshot_df)
    
    # 2. Transform features
    X_transformed = preprocessor.transform(X_raw)
    
    # 3. Predict classification
    pred_classes = classifier.predict(X_transformed)
    pred_probs = classifier.predict_proba(X_transformed)
    
    # 4. Predict continuous speed
    pred_speeds = forecaster.predict(X_transformed)
    
    result = snapshot_df.copy()
    result["pred_class_code"] = pred_classes
    result["pred_condition"] = [LABEL_SHORT[c] for c in pred_classes]
    result["pred_condition_full"] = [LABEL_MAP[c] for c in pred_classes]
    result["pred_speed_15m"] = np.round(pred_speeds, 1)
    
    # Highest class probability as model confidence
    result["confidence_score"] = np.round(np.max(pred_probs, axis=1) * 100.0, 1)
    result["prob_low"] = np.round(pred_probs[:, 0], 3)
    result["prob_mod"] = np.round(pred_probs[:, 1], 3)
    result["prob_high"] = np.round(pred_probs[:, 2], 3)
    
    # Speed drop / delta
    result["current_speed"] = np.round(result["x_t+0_d0"], 1)
    result["speed_change_15m"] = np.round(result["pred_speed_15m"] - result["current_speed"], 1)
    
    # Observed condition at t0
    obs_condition = []
    for s in result["current_speed"]:
        if s >= 55.0:
            obs_condition.append("Low")
        elif s >= 35.0:
            obs_condition.append("Moderate")
        else:
            obs_condition.append("High")
    result["observed_condition"] = obs_condition
    
    return result
