"""
test_hotspot_ranking.py - Unit tests for dynamic hotspot ranking engine
Tests formula normalization [0 to 100], ranking stability, tier boundaries, and configurable weights.
"""

import pytest
import pandas as pd
import numpy as np

from src.hotspot_ranking import (
    calculate_hotspot_priority_scores, 
    DEFAULT_WEIGHTS, 
    DEFAULT_TIER_THRESHOLDS
)

def test_hotspot_scores_range_and_normalization():
    # Synthetic batch of 5 sensors
    data = {
        "node_id": [0, 1, 2, 3, 4],
        "sensor_id": [773869, 767541, 767542, 717447, 717446],
        "corridor": ["US-101", "I-405", "SR-134", "I-110", "I-5"],
        "current_speed": [15.0, 30.0, 48.0, 62.0, 68.0],
        "pred_speed_15m": [12.0, 25.0, 46.0, 63.0, 67.0],
        "speed_change_15m": [-3.0, -5.0, -2.0, +1.0, -1.0],
        "confidence_score": [95.0, 90.0, 85.0, 92.0, 98.0]
    }
    df = pd.DataFrame(data)
    
    ranked = calculate_hotspot_priority_scores(df)
    
    assert len(ranked) == 5
    assert (ranked["priority_score"] >= 0.0).all()
    assert (ranked["priority_score"] <= 100.0).all()
    
    # Sensor 0 with 15 mph speed and worsening trend must have higher priority than Sensor 4 with 68 mph
    assert ranked.loc[ranked["node_id"] == 0, "priority_score"].values[0] > \
           ranked.loc[ranked["node_id"] == 4, "priority_score"].values[0]

def test_ranking_stability_under_unchanged_inputs():
    data = {
        "node_id": [10, 20],
        "sensor_id": [1001, 1002],
        "corridor": ["US-101", "I-10"],
        "current_speed": [22.0, 58.0],
        "pred_speed_15m": [20.0, 59.0],
        "speed_change_15m": [-2.0, +1.0],
        "confidence_score": [90.0, 90.0]
    }
    df = pd.DataFrame(data)
    
    run1 = calculate_hotspot_priority_scores(df)
    run2 = calculate_hotspot_priority_scores(df)
    
    pd.testing.assert_series_equal(run1["priority_score"], run2["priority_score"])
    assert list(run1["hotspot_rank"].values) == list(run2["hotspot_rank"].values)

def test_custom_configurable_weights():
    data = {
        "node_id": [1, 2],
        "sensor_id": [101, 102],
        "corridor": ["US-101", "SR-134"],
        "current_speed": [20.0, 50.0],
        "pred_speed_15m": [15.0, 30.0],
        "speed_change_15m": [-5.0, -20.0], # Node 2 has massive deceleration
        "confidence_score": [90.0, 90.0]
    }
    df = pd.DataFrame(data)
    
    # Heavily weight deterioration
    weights_det = {"severity": 0.1, "deterioration": 0.8, "recurrence": 0.05, "persistence": 0.05}
    ranked_det = calculate_hotspot_priority_scores(df, weights=weights_det)
    
    # Node 2 with massive deceleration (-20 mph) should now rank 1st
    assert ranked_det.iloc[0]["node_id"] == 2

def test_severity_tier_assignment():
    data = {
        "node_id": [1, 2, 3, 4],
        "sensor_id": [11, 22, 33, 44],
        "corridor": ["Corridor"] * 4,
        "current_speed": [10.0, 38.0, 52.0, 65.0],
        "pred_speed_15m": [8.0, 35.0, 52.0, 65.0],
        "speed_change_15m": [-2.0, -3.0, 0.0, 0.0],
        "confidence_score": [95.0, 90.0, 85.0, 95.0]
    }
    df = pd.DataFrame(data)
    ranked = calculate_hotspot_priority_scores(df)
    
    assert set(ranked["severity_tier"].unique()).issubset({"Critical", "High", "Medium", "Low"})
    # Slowest sensor should be Critical or High
    assert ranked.loc[ranked["node_id"] == 1, "severity_tier"].values[0] in ["Critical", "High"]
