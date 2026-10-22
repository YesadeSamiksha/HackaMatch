"""
test_traffic_patterns.py - Unit tests for recurring traffic patterns and rush hour intelligence
Tests hour x day heatmap, peak window calculations, and automated insight templates.
"""

import pytest
import pandas as pd
import numpy as np

from src.traffic_patterns import analyze_traffic_patterns

def test_traffic_patterns_structure():
    patterns = analyze_traffic_patterns(sample_size=30000)
    
    assert "heatmap_speed" in patterns
    assert "heatmap_congestion" in patterns
    assert "hourly_summary" in patterns
    assert "insights" in patterns
    assert "persistent_ranking" in patterns
    
    # Validate Heatmap dimensions (24 hours)
    heat_spd = patterns["heatmap_speed"]
    assert len(heat_spd) == 24, "Heatmap must contain all 24 hours (0 to 23)"
    
    # Validate peak speed relationships
    morn = patterns["morn_speed"]
    eve = patterns["eve_speed"]
    midday = patterns["midday_speed"]
    assert 20.0 <= morn <= 70.0
    assert 20.0 <= eve <= 70.0
    assert 20.0 <= midday <= 70.0

def test_three_evidence_backed_insights_format():
    patterns = analyze_traffic_patterns(sample_size=30000)
    insights = patterns["insights"]
    
    # Exactly 3 automated insights
    assert len(insights) == 3
    
    # Verify each insight adheres to the required template:
    # "Sensor {sensor_id} recorded an average speed of {observed_speed} mph during {period}, compared with {baseline_speed} mph during {comparison_period}."
    for ins in insights:
        assert "recorded an average speed of" in ins
        assert "compared with" in ins
        assert "mph" in ins

def test_persistent_ranking_definition():
    patterns = analyze_traffic_patterns(sample_size=30000)
    p_df = patterns["persistent_ranking"]
    
    assert "is_persistent" in p_df.columns
    assert "congestion_rate" in p_df.columns
    # Check that highest congestion rate is ranked 1st
    assert p_df.iloc[0]["congestion_rate"] >= p_df.iloc[-1]["congestion_rate"]
