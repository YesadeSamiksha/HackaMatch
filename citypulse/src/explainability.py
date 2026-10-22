"""
explainability.py - User-Friendly Explainability and Attribution Engine for CityPulse
Translates technical traffic telemetry into clear, everyday language anyone can understand.
"""

import pandas as pd
import numpy as np

def explain_node_prediction(node_data: pd.Series) -> dict:
    """
    Produces clear, jargon-free explanations for any selected traffic location.
    Explains current speeds, predicted changes, and practical reasons in everyday English.
    """
    node_id = int(node_data.get("node_id", 0))
    sensor_id = str(node_data.get("sensor_id", node_id))
    corridor = str(node_data.get("corridor", "Freeway Corridor"))
    current_speed = float(node_data.get("current_speed", node_data.get("x_t+0_d0", 65.0)))
    pred_speed = float(node_data.get("pred_speed_15m", current_speed))
    pred_condition = str(node_data.get("pred_condition", "Low"))
    confidence = float(node_data.get("confidence_score", 85.0))
    speed_trend = float(node_data.get("speed_change_15m", pred_speed - current_speed))
    
    # Normal highway speed baseline
    normal_speed = 65.0
    slowdown_amount = max(0.0, normal_speed - current_speed)
    
    evidence_points = []
    
    # 1. Current Speed Description in Plain English
    if current_speed < 35.0:
        evidence_points.append(
            f"🚗 **Heavy Slowdown:** Traffic is moving at only **{current_speed:.1f} mph** "
            f"— that's **{slowdown_amount:.1f} mph slower** than normal highway flow."
        )
    elif current_speed < 55.0:
        evidence_points.append(
            f"🚙 **Moderate Congestion:** Cars are traveling at **{current_speed:.1f} mph**. "
            f"Traffic is moving, but drivers are experiencing noticeable delays."
        )
    else:
        evidence_points.append(
            f"🟢 **Smooth Highway Flow:** Traffic is cruising at **{current_speed:.1f} mph** with no major delays."
        )
        
    # 2. 15-Minute Forecast Description
    if speed_trend < -2.5:
        evidence_points.append(
            f"📉 **Expect More Delays:** Speeds are predicted to drop by another **{abs(speed_trend):.1f} mph** "
            f"over the next 15 minutes as congestion builds up."
        )
    elif speed_trend > 2.5:
        evidence_points.append(
            f"📈 **Traffic is Clearing:** Speeds are expected to pick up by **+{speed_trend:.1f} mph** "
            f"over the next 15 minutes as the bottleneck eases."
        )
    else:
        evidence_points.append(
            f"⏱️ **Steady Conditions:** Speeds are expected to remain steady around **{pred_speed:.1f} mph** over the next 15 minutes."
        )
        
    # 3. Plain English Recommendations
    if pred_condition == "High":
        status_label = "🔴 Heavy Traffic Jam"
        priority_level = "High Urgency"
        action_summary = "Bottleneck in progress. Recommend notifying drivers via overhead signs and adjusting on-ramp signals."
    elif pred_condition == "Moderate":
        status_label = "🟡 Moderate Delays"
        priority_level = "Medium Watchlist"
        action_summary = "Slowdown detected. Monitor nearby exits and on-ramps to prevent further traffic buildup."
    else:
        status_label = "🟢 Smooth Free-Flow"
        priority_level = "Normal"
        action_summary = "All clear. Normal highway traffic flow."
        
    explanation = {
        "node_id": node_id,
        "sensor_id": sensor_id,
        "corridor": corridor,
        "current_speed": current_speed,
        "pred_speed_15m": pred_speed,
        "pred_condition": pred_condition,
        "status_label": status_label,
        "priority_level": priority_level,
        "confidence": confidence,
        "slowdown_amount": slowdown_amount,
        "speed_trend": speed_trend,
        "evidence_points": evidence_points,
        "action_summary": action_summary,
        "friendly_note": "Predictions are generated from continuous roadside traffic sensors and historical speed trends."
    }
    return explanation
