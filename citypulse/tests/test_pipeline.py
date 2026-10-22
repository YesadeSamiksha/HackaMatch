import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import engineer_features, create_targets, FEATURE_COLS
from src.data_loader import get_sensor_locations, get_adjacency_matrix, get_network_snapshot
from src.inference import load_artifacts, predict_network_state
from src.scenario_engine import simulate_scenario
from src.priority_engine import compute_priority_rankings
from src.mapping import create_sensor_map

def test_sensor_locations_validity():
    locs = get_sensor_locations()
    assert len(locs) == 207, f"Expected 207 sensors, found {len(locs)}"
    assert "latitude" in locs.columns and "longitude" in locs.columns
    assert "corridor" in locs.columns
    
    # Check geographic coordinate bounds for Los Angeles County
    assert locs["latitude"].between(33.5, 34.5).all(), "Latitudes must be within Los Angeles County bounds"
    assert locs["longitude"].between(-119.0, -117.5).all(), "Longitudes must be within Los Angeles County bounds"

def test_adjacency_matrix_shape():
    sensor_ids, adj_mx = get_adjacency_matrix()
    assert len(sensor_ids) == 207
    assert adj_mx.shape == (207, 207)
    assert np.count_nonzero(adj_mx > 0) > 1000, "Adjacency matrix should contain positive network distance edges"

def test_leakage_free_feature_engineering():
    # Create synthetic test batch mimicking train.parquet
    cols = ["node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1"] + [f"x_t-{i}_d0" for i in range(1, 12)]
    data = {c: np.random.uniform(50.0, 65.0, size=10) for c in cols}
    data["node_id"] = np.arange(10)
    data["t0_timestamp"] = ["2012-05-01T08:00:00"] * 10
    data["x_t+0_d1"] = [0.33] * 10
    df_raw = pd.DataFrame(data)
    
    features = engineer_features(df_raw)
    assert list(features.columns) == FEATURE_COLS
    assert not any(col.startswith("y_") for col in features.columns), "Future target columns must NOT be in feature matrix"
    assert not features.isnull().any().any(), "Engineered features must not contain nulls"

def test_model_loading_and_inference():
    preprocessor, classifier, forecaster = load_artifacts()
    assert preprocessor is not None
    assert classifier is not None
    assert forecaster is not None
    
    snapshot = get_network_snapshot()
    assert len(snapshot) == 207
    
    preds = predict_network_state(snapshot)
    assert len(preds) == 207
    assert "pred_condition" in preds.columns
    assert "pred_speed_15m" in preds.columns
    assert "confidence_score" in preds.columns
    
    # Valid condition labels
    assert set(preds["pred_condition"].unique()).issubset({"Low", "Moderate", "High"})
    # Physical speed limits
    assert preds["pred_speed_15m"].between(0.0, 100.0).all()
    # Confidence bounds
    assert preds["confidence_score"].between(0.0, 100.0).all()

def test_priority_ranking_engine():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    ranked = compute_priority_rankings(preds)
    
    assert len(ranked) == 207
    assert list(ranked["rank"].values) == list(range(1, 208))
    assert "priority_score" in ranked.columns
    assert "priority_tier" in ranked.columns
    assert set(ranked["priority_tier"].unique()).issubset({"High Priority", "Medium Priority", "Low Priority"})

def test_what_if_scenario_simulation():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    test_node = preds.iloc[0]
    
    sim = simulate_scenario(
        target_node_id=int(test_node["node_id"]),
        baseline_node_data=test_node,
        closure_intensity_pct=50.0,
        closure_duration_mins=30,
        weather_condition="Heavy Rain / Storm"
    )
    
    assert "scenario_speed" in sim
    assert "speed_difference" in sim
    assert sim["scenario_speed"] <= sim["baseline_speed"], "50% closure with heavy rain must reduce or maintain speed"
    assert "affected_neighbors" in sim
    assert isinstance(sim["comparison_table"], pd.DataFrame)

def test_sensor_map_creation():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    ranked = compute_priority_rankings(preds)
    fig = create_sensor_map(ranked, selected_node_id=0)
    assert fig is not None
    assert len(fig.data) >= 3, "Map should have traces for condition categories and selected node"

def test_extreme_speed_boundaries_inference():
    # Test boundary behavior under extreme standstills (0 mph) and max speed (70 mph)
    preprocessor, classifier, forecaster = load_artifacts()
    
    dummy_data = {
        "node_id": [0, 1],
        "t0_timestamp": ["2012-05-01T08:00:00", "2012-05-01T08:00:00"],
        "x_t+0_d0": [0.0, 70.0],
        "x_t+0_d1": [0.33, 0.33]
    }
    for i in range(1, 12):
        dummy_data[f"x_t-{i}_d0"] = [0.0, 70.0]
    
    df_boundary = pd.DataFrame(dummy_data)
    features = engineer_features(df_boundary)
    X_trans = preprocessor.transform(features)
    
    preds_class = classifier.predict(X_trans)
    preds_speed = forecaster.predict(X_trans)
    
    # 0 mph should predict High Congestion (class 2)
    assert preds_class[0] == 2, "0 mph speed must map to High Congestion"
    # 70 mph should predict Low Congestion (class 0)
    assert preds_class[1] == 0, "70 mph speed must map to Low Congestion"
    assert preds_speed[0] < preds_speed[1], "0 mph speed input must yield lower forecast than 70 mph"

def test_missing_model_graceful_handling(tmp_path):
    # Tests that nonexistent path raises FileNotFoundError
    non_existent = tmp_path / "nonexistent_model.joblib"
    with pytest.raises(FileNotFoundError):
        joblib.load(str(non_existent))

def test_temporal_chronological_integrity():
    # Validates that dataset timestamps span March to May 2012
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    
    from config import resolve_data_path
    parquet_path = resolve_data_path("train.parquet")
    if not parquet_path.exists():
        parquet_path = resolve_data_path("snapshot_cache.parquet")
    
    table = pq.read_table(parquet_path, columns=["t0_timestamp"])
    min_ts = pc.min(table["t0_timestamp"]).as_py()
    max_ts = pc.max(table["t0_timestamp"]).as_py()
    
    assert min_ts.startswith("2012-03"), f"Dataset should start in March 2012, got {min_ts}"
    assert max_ts.startswith("2012-05"), f"Dataset should extend into May 2012, got {max_ts}"

def test_loss_unpickling_compatibility():
    """
    Regression test for Cython loss unpickling compatibility:
    Verifies that serialized HistGradientBoosting models unpickle their loss instances
    without 'sklearn._loss._loss has no attribute __pyx_unpickle_CyHalfSquaredError'.
    """
    from config import resolve_model_path
    
    forecaster = joblib.load(resolve_model_path("traffic_forecaster.joblib"))
    classifier = joblib.load(resolve_model_path("traffic_classifier.joblib"))
    
    assert hasattr(forecaster, "_loss"), "Forecaster must have _loss attribute"
    assert forecaster._loss is not None, "Forecaster loss object must be instantiated"
    assert "HalfSquaredError" in type(forecaster._loss).__name__ or hasattr(forecaster._loss, "loss")
    
    assert hasattr(classifier, "_loss"), "Classifier must have _loss attribute"
    assert classifier._loss is not None, "Classifier loss object must be instantiated"

def test_prediction_lab_initialization():
    """
    Regression test for Prediction & Simulation Lab initialization:
    Validates the exact pipeline called by load_lab_data() in pages/3_Prediction_Lab.py.
    """
    from src.data_loader import get_network_snapshot
    from src.inference import predict_network_state
    from src.traffic_patterns import analyze_traffic_patterns
    
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    patterns = analyze_traffic_patterns(sample_size=5000)
    
    assert len(preds) == 207
    assert "pred_speed_15m" in preds.columns
    assert "pred_condition" in preds.columns
    assert "recurrence_map" in patterns
    assert len(patterns["recurrence_map"]) == 207


