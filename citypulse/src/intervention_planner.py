"""
intervention_planner.py - Explainable Intervention Planner for TrafficPulse AI
Combines dynamic priority scores, forecast kinematics, historical patterns,
and what-if simulation to generate actionable, evidence-grounded recommendations.
"""

import pandas as pd
import numpy as np

ACTION_PLAYBOOK = {
    "Ramp Metering Adjustment": {
        "trigger": "Active queue buildup during recurring peak rush hours",
        "description": "Adjust on-ramp metering signal cycle rates upstream to throttle freeway inflow and prevent mainline breakdown.",
        "effectiveness": "Recommended for recurrent arterial bottlenecks with sustained demand."
    },
    "Patrol Dispatch & Incident Verification": {
        "trigger": "Severe unpredicted speed collapse (< 25 mph) or rapid deceleration shockwave",
        "description": "Dispatch Freeway Service Patrol (FSP) or traffic officer to check for stalled vehicles, debris, or lane obstructions.",
        "effectiveness": "Immediate priority for sudden non-recurrent slowdowns."
    },
    "Dynamic Message Sign (DMS) Advisory": {
        "trigger": "Significant speed differential between adjacent highway corridors",
        "description": "Activate roadside electronic advisory boards upstream warning drivers of delays and suggesting alternative bypass corridors.",
        "effectiveness": "Diverts 8-15% of approaching traffic onto parallel arterial routes."
    },
    "Variable Speed Advisory": {
        "trigger": "Moderate congestion with high speed volatility (turbulence)",
        "description": "Harmonize approaching traffic speeds by lowering advisory limits to smooth shockwave propagation.",
        "effectiveness": "Reduces stop-and-go rear-end collision hazards and stabilizes throughput."
    },
    "Continuous Telemetry Monitoring": {
        "trigger": "Normal free-flow or stable baseline traffic",
        "description": "Standard automated 5-minute sensor polling; no active intervention required.",
        "effectiveness": "Baseline operational state."
    }
}

def generate_intervention_plan(node_row: pd.Series) -> dict:
    """
    Synthesizes current state, forecast, historical recurrence, and rule-based playbook
    to answer the 6 core operational questions for any hotspot location.
    """
    node_id = int(node_row.get("node_id", 0))
    sensor_id = str(node_row.get("sensor_id", node_id))
    corridor = str(node_row.get("corridor", "Highway Corridor"))
    curr_spd = float(node_row.get("current_speed", 60.0))
    pred_spd = float(node_row.get("pred_speed_15m", curr_spd))
    speed_change = float(node_row.get("speed_change_15m", pred_spd - curr_spd))
    priority_score = float(node_row.get("priority_score", 50.0))
    tier = str(node_row.get("severity_tier", node_row.get("priority_tier", "Medium")))
    pred_cond = str(node_row.get("pred_condition", "Low"))
    recurrence = float(node_row.get("score_recurrence", 0.20))
    confidence = float(node_row.get("confidence_score", 85.0))
    
    # 1. What is happening?
    if curr_spd < 35.0:
        what_is_happening = f"Severe traffic bottleneck active: Operating speed has collapsed to {curr_spd:.1f} mph ({max(0, 65.0 - curr_spd):.1f} mph below normal free-flow design speed)."
    elif curr_spd < 55.0:
        what_is_happening = f"Moderate flow restriction: Vehicles are traveling at {curr_spd:.1f} mph with noticeable density and reduced spacing."
    else:
        what_is_happening = f"Unrestricted free-flow: Vehicles are moving smoothly at {curr_spd:.1f} mph."
        
    # 2. What is predicted to happen in 15 minutes?
    if speed_change < -2.0:
        what_predicted = f"Congestion is projected to worsen: Model forecasts further deceleration to {pred_spd:.1f} mph ({speed_change:+.1f} mph) within 15 minutes."
    elif speed_change > 2.0:
        what_predicted = f"Congestion is projected to ease: Speeds are predicted to recover to {pred_spd:.1f} mph ({speed_change:+.1f} mph) within 15 minutes."
    else:
        what_predicted = f"Conditions projected to remain steady: Speeds will hover around {pred_spd:.1f} mph ({speed_change:+.1f} mph change) over the next 15 minutes."
        
    # 3. Why is the hotspot highly ranked?
    ranking_factors = []
    if curr_spd < 40.0:
        ranking_factors.append(f"Substantial speed deficit ({65.0 - curr_spd:.1f} mph below free-flow)")
    if speed_change < -2.0:
        ranking_factors.append(f"Active queue buildup ({speed_change:+.1f} mph deceleration)")
    if recurrence >= 0.25:
        ranking_factors.append(f"High historical recurrence ({recurrence*100:.0f}% frequency)")
        
    why_ranked = "; ".join(ranking_factors) if ranking_factors else "Baseline operational ranking under prevailing corridor conditions"
    
    # 4. Which metrics support the explanation?
    supporting_metrics = {
        "Current Measured Speed": f"{curr_spd:.1f} mph",
        "Forecast Speed (t+15m)": f"{pred_spd:.1f} mph",
        "Projected Trend": f"{speed_change:+.1f} mph",
        "Priority Urgency Score": f"{priority_score:.1f} / 100",
        "Historical Recurrence Rate": f"{recurrence*100:.1f}%",
        "Model Confidence Score": f"{confidence:.0f}%"
    }
    
    # 5. Which action should be investigated?
    if curr_spd < 30.0 and speed_change < -2.0:
        primary_action = ACTION_PLAYBOOK["Patrol Dispatch & Incident Verification"]
        action_name = "Patrol Dispatch & Incident Clearance"
    elif curr_spd < 45.0 and recurrence >= 0.20:
        primary_action = ACTION_PLAYBOOK["Ramp Metering Adjustment"]
        action_name = "On-Ramp Metering Rate Optimization"
    elif curr_spd < 50.0:
        primary_action = ACTION_PLAYBOOK["Dynamic Message Sign (DMS) Advisory"]
        action_name = "Corridor Divert Advisory on Overhead DMS"
    elif curr_spd < 55.0:
        primary_action = ACTION_PLAYBOOK["Variable Speed Advisory"]
        action_name = "Harmonized Variable Speed Advisory"
    else:
        primary_action = ACTION_PLAYBOOK["Continuous Telemetry Monitoring"]
        action_name = "Routine Automated Polling"
        
    # 6. What uncertainty or limitation applies?
    limitations = (
        "Investigation suggestions are decision-support aids based on statistical sensor models and historical Caltrans PeMS loop data. "
        "They do not guarantee congestion mitigation without on-site field verification."
    )
    
    return {
        "node_id": node_id,
        "sensor_id": sensor_id,
        "corridor": corridor,
        "priority_score": priority_score,
        "severity_tier": tier,
        "what_is_happening": what_is_happening,
        "what_predicted": what_predicted,
        "why_ranked": why_ranked,
        "supporting_metrics": supporting_metrics,
        "recommended_action_name": action_name,
        "recommended_action_details": primary_action,
        "limitations": limitations
    }
