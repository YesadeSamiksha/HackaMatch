"""
scenario_engine.py - What-If Scenario Simulation & Network Impact Analysis
Simulates hypothetical road closures, capacity reductions, and adverse weather shockwaves.
Leverages highway network graph adjacency for estimating queue propagation to connected segments.
"""

import pandas as pd
import numpy as np
from src.data_loader import get_adjacency_matrix

WEATHER_IMPACT_FACTORS = {
    "Clear / Dry (Baseline)": {
        "speed_factor": 1.00, 
        "capacity_factor": 1.00, 
        "description": "Baseline pavement friction and normal visibility."
    },
    "Light Rain / Wet Pavement": {
        "speed_factor": 0.90, 
        "capacity_factor": 0.85, 
        "description": "10% speed reduction due to increased vehicle headway and wet roadway."
    },
    "Heavy Rain / Storm": {
        "speed_factor": 0.75, 
        "capacity_factor": 0.70, 
        "description": "25% speed reduction due to surface water ponding and windshield visibility impairment."
    },
    "Dense Fog / Poor Visibility": {
        "speed_factor": 0.80, 
        "capacity_factor": 0.75, 
        "description": "20% speed reduction due to cautious driver following distances and reduced sight lines."
    }
}

def simulate_scenario(
    target_node_id: int,
    baseline_node_data: pd.Series,
    closure_intensity_pct: float,
    closure_duration_mins: int,
    weather_condition: str = "Clear / Dry (Baseline)"
) -> dict:
    """
    Simulates the network effect of a hypothetical capacity reduction and/or weather event.
    
    closure_intensity_pct: 0 to 100 (e.g. 50% = 2 of 4 lanes blocked; 100% = full arterial shutdown).
    closure_duration_mins: assumed duration of the incident (15 to 120 mins).
    """
    weather_cfg = WEATHER_IMPACT_FACTORS.get(weather_condition, WEATHER_IMPACT_FACTORS["Clear / Dry (Baseline)"])
    w_speed_factor = weather_cfg["speed_factor"]
    
    current_speed = float(baseline_node_data.get("current_speed", baseline_node_data.get("x_t+0_d0", 60.0)))
    pred_speed = float(baseline_node_data.get("pred_speed_15m", current_speed))
    baseline_condition = str(baseline_node_data.get("pred_condition", "Low"))
    sensor_id = str(baseline_node_data.get("sensor_id", target_node_id))
    corridor = str(baseline_node_data.get("corridor", "Corridor Identifier Unavailable"))
    
    # 1. Direct Node Impact Calculation
    # Capacity reduction and adverse weather attenuate baseline forecast speed
    closure_fraction = max(0.0, min(100.0, float(closure_intensity_pct))) / 100.0
    
    if closure_intensity_pct >= 90.0:
        # Near-total closure collapses speed to crawling queue (~3-5 mph)
        scenario_speed = max(3.0, pred_speed * 0.08 * w_speed_factor)
    elif closure_intensity_pct <= 0.0 and weather_condition == "Clear / Dry (Baseline)":
        # 0% closure with baseline clear weather produces no change from baseline forecast
        scenario_speed = pred_speed
    else:
        # Speed penalty combines capacity reduction and weather factor
        speed_penalty = (closure_fraction * 0.65) + (1.0 - w_speed_factor)
        scenario_speed = max(5.0, pred_speed * (1.0 - speed_penalty))
        
    scenario_speed = round(scenario_speed, 1)
    abs_speed_diff = round(scenario_speed - pred_speed, 1)
    rel_speed_diff_pct = round((abs_speed_diff / max(1.0, pred_speed)) * 100.0, 1)
    
    # Scenario condition classification based on standard highway LOS thresholds
    if scenario_speed < 35.0:
        scenario_condition = "High Congestion (< 35 mph)"
        impact_level = "Critical Impact"
        impact_desc = "Operating speed collapses below 35 mph Level of Service F. Severe queue formation expected."
    elif scenario_speed < 55.0:
        scenario_condition = "Moderate Congestion (35–55 mph)"
        impact_level = "Moderate Impact"
        impact_desc = "Operating speed degraded between 35 and 55 mph. Significant traffic slowdown and reduced throughput."
    else:
        scenario_condition = "Low Congestion (>= 55 mph)"
        impact_level = "Minimal Impact"
        impact_desc = "Operating speed remains at or above 55 mph. Free-flow conditions sustained."
        
    # 2. Network Spatial Connectivity using highway graph
    sensor_ids, adj_mx = get_adjacency_matrix()
    
    node_idx = int(target_node_id)
    affected_neighbors = []
    
    if 0 <= node_idx < adj_mx.shape[0]:
        weights = adj_mx[node_idx, :]
        connected_indices = np.where((weights > 0) & (weights < 1.0))[0] # Non-self connections
        
        # Sort by connection proximity (higher weight = closer distance in exponential kernel)
        sorted_conn = connected_indices[np.argsort(-weights[connected_indices])]
        
        # Incident duration determines shockwave reach limit
        reach_limit = min(len(sorted_conn), max(1, int(closure_duration_mins / 15)))
        
        for neighbor_idx in sorted_conn[:reach_limit]:
            weight = float(weights[neighbor_idx])
            # Queue spillover drop scaled by closure intensity, duration, and connection weight
            spillover_drop = round(abs(abs_speed_diff) * weight * (closure_duration_mins / 60.0) * 0.5, 1)
            neighbor_station_label = str(sensor_ids[neighbor_idx]) if neighbor_idx < len(sensor_ids) else f"Node {neighbor_idx}"
            affected_neighbors.append({
                "neighbor_node_id": int(neighbor_idx),
                "neighbor_sensor_id": neighbor_station_label,
                "network_affinity_weight": round(weight, 3),
                "estimated_speed_reduction_mph": spillover_drop
            })
            
    # 3. User-Friendly Comparison Table (Priority 4.C & 4.D)
    comparison_rows = [
        {
            "Metric": "Corridor Speed (15-min horizon)",
            "Baseline Forecast": f"{pred_speed:.1f} mph",
            "Estimated Scenario Speed": f"{scenario_speed:.1f} mph",
            "Estimated Change from Baseline": f"{abs_speed_diff:+.1f} mph ({rel_speed_diff_pct:+.1f}%)",
            "Modelling Method": "Kinematic Capacity & Weather Sensitivity Analysis",
            "Key Assumptions & Limitations": f"Assumes {closure_intensity_pct:.0f}% capacity throttling for {closure_duration_mins} min; {weather_cfg['description']}"
        },
        {
            "Metric": "Predicted Traffic Condition",
            "Baseline Forecast": baseline_condition,
            "Estimated Scenario Speed": scenario_condition.split(" (")[0],
            "Estimated Change from Baseline": "Condition Transition" if baseline_condition not in scenario_condition else "No Category Shift",
            "Modelling Method": "Level of Service (LOS) Rule Mapping",
            "Key Assumptions & Limitations": "High < 35 mph, Moderate 35-55 mph, Low >= 55 mph"
        },
        {
            "Metric": "Connected Road Segments Affected",
            "Baseline Forecast": "0 Segments Affected",
            "Estimated Scenario Speed": f"{len(affected_neighbors)} Connected Segments Considered",
            "Estimated Change from Baseline": f"+{len(affected_neighbors)} Connected Segments",
            "Modelling Method": "Highway Network Graph Adjacency Propagation",
            "Key Assumptions & Limitations": f"Considers adjacent links within {closure_duration_mins}-min queue shockwave boundary based on network proximity weights."
        }
    ]
    
    # 4. Structured Evidence & Assumptions Audit (Priority 4.E)
    evidence_audit = {
        "observed_data": {
            "current_observed_speed": f"{current_speed:.1f} mph",
            "station_id": sensor_id,
            "corridor": corridor,
            "source_telemetry": "Caltrans PeMS District 7 loop detector historical telemetry archive (5-minute interval).",
            "observation_type": "Historical ground truth observation; not real-time live sensor feed."
        },
        "model_predictions": {
            "baseline_forecast_speed": f"{pred_speed:.1f} mph",
            "forecast_horizon": "15 minutes ahead (t+15m)",
            "baseline_condition": baseline_condition,
            "model_type": "HistGradientBoostingRegressor (Supervised Machine Learning Model)",
            "confidence_metric": f"{float(baseline_node_data.get('confidence_score', 0.0)):.1f}%" if pd.notnull(baseline_node_data.get('confidence_score')) else "Unavailable"
        },
        "scenario_assumptions": {
            "capacity_reduction": f"{closure_intensity_pct:.0f}% of total road capacity unavailable (Hypothetical Assumption)",
            "assumed_duration": f"{closure_duration_mins} minutes continuous disruption",
            "hypothetical_weather": f"{weather_condition} ({weather_cfg['description']})",
            "propagation_model": "Kinematic shockwave attenuation across adjacent network graph nodes"
        },
        "limitations": {
            "nature_of_output": "Results are model estimates and scenario calculations, not confirmed real-world incidents.",
            "spillover_interpretation": "Connected road segments are adjacent links identified by the network graph; actual congestion depends on unobserved ramp volumes and driver detour choices.",
            "data_timeliness": "Based on historical Caltrans PeMS D7 loop telemetry; external live feeds are not active in this offline benchmark."
        }
    }
    
    # Explanatory narrative (Priority 4.D)
    if abs_speed_diff < 0:
        interpretation_narrative = (
            f"Under the selected hypothetical conditions, the estimated speed decreases by {abs(abs_speed_diff):.1f} mph "
            f"(from a baseline forecast of {pred_speed:.1f} mph to {scenario_speed:.1f} mph). "
            f"This suggests a potential deterioration in traffic flow under the scenario assumptions. "
            f"The result is a simulated estimate, not proof that an incident has occurred."
        )
    elif abs_speed_diff > 0:
        interpretation_narrative = (
            f"Under the selected hypothetical conditions, the estimated speed increases by {abs_speed_diff:+.1f} mph. "
            f"The result is a simulated estimate, not proof that an incident has occurred."
        )
    else:
        interpretation_narrative = (
            f"Under the selected conditions, the estimated scenario speed matches the baseline forecast ({pred_speed:.1f} mph). "
            f"Operating conditions remain within baseline projected bounds. "
            f"The result is a simulated estimate, not proof that an incident has occurred."
        )
        
    return {
        "target_node_id": node_idx,
        "baseline_speed": pred_speed,
        "scenario_speed": scenario_speed,
        "speed_difference": abs_speed_diff,
        "baseline_condition": baseline_condition,
        "scenario_condition": scenario_condition,
        "impact_level": impact_level,
        "impact_description": impact_desc,
        "interpretation_narrative": interpretation_narrative,
        "affected_neighbors": affected_neighbors,
        "comparison_table": pd.DataFrame(comparison_rows),
        "evidence_audit": evidence_audit,
        "methodology_disclaimer": "Scenario outcomes represent a deterministic sensitivity simulation utilizing the highway network graph. They are NOT observed real-time traffic incidents."
    }
