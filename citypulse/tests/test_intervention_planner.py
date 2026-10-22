"""
test_intervention_planner.py - Unit tests for explainable intervention planner
Tests answers to the 6 core questions, rule-based playbook triggers, and limitation transparency.
"""

import pytest
import pandas as pd

from src.intervention_planner import generate_intervention_plan, ACTION_PLAYBOOK

def test_intervention_plan_answers_six_core_questions():
    sample_node = pd.Series({
        "node_id": 14,
        "sensor_id": 765273,
        "corridor": "US-101 / I-405 Interchange",
        "current_speed": 18.5,
        "pred_speed_15m": 14.0,
        "speed_change_15m": -4.5,
        "priority_score": 82.4,
        "severity_tier": "Critical",
        "score_recurrence": 0.45,
        "confidence_score": 96.0
    })
    
    plan = generate_intervention_plan(sample_node)
    
    # 1. What is happening?
    assert "what_is_happening" in plan
    assert "18.5 mph" in plan["what_is_happening"]
    
    # 2. What is predicted in 15 minutes?
    assert "what_predicted" in plan
    assert "14.0 mph" in plan["what_predicted"]
    
    # 3. Why is the hotspot ranked?
    assert "why_ranked" in plan
    assert len(plan["why_ranked"]) > 0
    
    # 4. Which metrics support it?
    assert "supporting_metrics" in plan
    assert "Current Measured Speed" in plan["supporting_metrics"]
    assert "Priority Urgency Score" in plan["supporting_metrics"]
    
    # 5. Which action should be investigated?
    assert "recommended_action_name" in plan
    assert "recommended_action_details" in plan
    # Severe slowdown with deceleration shockwave should trigger patrol / clearance or ramp metering
    assert plan["recommended_action_name"] in [
        "Patrol Dispatch & Incident Clearance", 
        "On-Ramp Metering Rate Optimization"
    ]
    
    # 6. What limitation applies?
    assert "limitations" in plan
    assert "decision-support" in plan["limitations"].lower()

def test_playbook_triggers():
    assert "Ramp Metering Adjustment" in ACTION_PLAYBOOK
    assert "Patrol Dispatch & Incident Verification" in ACTION_PLAYBOOK
    assert "Dynamic Message Sign (DMS) Advisory" in ACTION_PLAYBOOK
    assert "Variable Speed Advisory" in ACTION_PLAYBOOK
    assert "Continuous Telemetry Monitoring" in ACTION_PLAYBOOK
