"""
scenario_engine.py - What-If Scenario Simulation & Network Impact Analysis
Simulates hypothetical road closures, capacity reductions, and adverse weather shockwaves.
Leverages graph adjacency matrix (adj_mx.pkl) for network spillover to downstream neighbors.
"""

import pandas as pd
import numpy as np
from src.data_loader import get_adjacency_matrix

WEATHER_IMPACT_FACTORS = {
    "Clear / Dry (Baseline)": {"speed_factor": 1.00, "capacity_factor": 1.00, "description": "Normal pavement friction and visibility."},
    "Light Rain / Wet Pavement": {"speed_factor": 0.90, "capacity_factor": 0.85, "description": "10% speed reduction due to increased headway and wet roadway."},
    "Heavy Rain / Storm": {"speed_factor": 0.75, "capacity_factor": 0.70, "description": "25% speed reduction due to ponding and windshield visibility impairment."},
    "Dense Fog / Poor Visibility": {"speed_factor": 0.80, "capacity_factor": 0.75, "description": "20% speed reduction due to cautious driver following distances."}
}

def simulate_scenario(
    target_node_id: int,
    baseline_node_data: pd.Series,
    closure_intensity_pct: float,
    closure_duration_mins: int,
    weather_condition: str = "Clear / Dry (Baseline)"
) -> dict:
    """
    Simulates the network effect of a hypothetical road closure and/or weather event.
    
    closure_intensity_pct: 0 to 100 (e.g. 50% = 2 of 4 lanes blocked; 100% = full arterial shutdown).
    closure_duration_mins: duration of the incident (15 to 120 mins).
    """
    weather_cfg = WEATHER_IMPACT_FACTORS.get(weather_condition, WEATHER_IMPACT_FACTORS["Clear / Dry (Baseline)"])
    w_speed_factor = weather_cfg["speed_factor"]
    
    current_speed = float(baseline_node_data.get("current_speed", baseline_node_data.get("x_t+0_d0", 60.0)))
    pred_speed = float(baseline_node_data.get("pred_speed_15m", current_speed))
    baseline_condition = str(baseline_node_data.get("pred_condition", "Low"))
    
    # 1. Direct Node Impact Calculation
    # Capacity reduction directly collapses operating speed according to Greenshields / BPR fundamental diagram
    closure_fraction = closure_intensity_pct / 100.0
    
    # Speed under scenario is attenuated by closure and weather
    # Full closure drops speed to crawl / queueing speed (~5 mph)
    if closure_intensity_pct >= 90.0:
        scenario_speed = max(3.0, current_speed * 0.08 * w_speed_factor)
    else:
        speed_penalty = (closure_fraction * 0.65) + (1.0 - w_speed_factor)
        scenario_speed = max(5.0, current_speed * (1.0 - speed_penalty))
        
    scenario_speed = round(scenario_speed, 1)
    abs_speed_diff = round(scenario_speed - pred_speed, 1)
    rel_speed_diff_pct = round((abs_speed_diff / max(1.0, pred_speed)) * 100.0, 1)
    
    # Scenario condition classification
    if scenario_speed < 35.0:
        scenario_condition = "High (Congested)"
    elif scenario_speed < 55.0:
        scenario_condition = "Moderate"
    else:
        scenario_condition = "Low (Free-Flow)"
        
    # 2. Network Spatial Spillover using adj_mx.pkl
    sensor_ids, adj_mx = get_adjacency_matrix()
    
    # Find adjacent network neighbors (positive connection weight in adj_mx)
    node_idx = int(target_node_id)
    affected_neighbors = []
    
    if 0 <= node_idx < adj_mx.shape[0]:
        # Outgoing connections (downstream flow) and incoming connections (upstream queue spillback)
        weights = adj_mx[node_idx, :]
        connected_indices = np.where((weights > 0) & (weights < 1.0))[0] # Non-self connections
        
        # Sort by proximity (higher weight = closer distance in exponential kernel)
        sorted_conn = connected_indices[np.argsort(-weights[connected_indices])]
        
        # Duration determines shockwave reach (typical queue growth rate ~10-15 mph backwards)
        reach_limit = min(len(sorted_conn), max(1, int(closure_duration_mins / 15)))
        
        for neighbor_idx in sorted_conn[:reach_limit]:
            weight = float(weights[neighbor_idx])
            spillover_drop = round(abs(abs_speed_diff) * weight * (closure_duration_mins / 60.0) * 0.5, 1)
            affected_neighbors.append({
                "neighbor_node_id": int(neighbor_idx),
                "neighbor_sensor_id": str(sensor_ids[neighbor_idx]) if neighbor_idx < len(sensor_ids) else f"Node {neighbor_idx}",
                "network_affinity_weight": round(weight, 3),
                "estimated_speed_reduction_mph": spillover_drop
            })
            
    # 3. Construct Comparison Table
    comparison_rows = [
        {
            "Metric": "Corridor Speed (15-min horizon)",
            "Baseline Value": f"{pred_speed:.1f} mph",
            "Scenario Value": f"{scenario_speed:.1f} mph",
            "Absolute Difference": f"{abs_speed_diff:+.1f} mph",
            "Relative Change": f"{rel_speed_diff_pct:+.1f}%",
            "Data Category": "Hypothetical Scenario Estimate",
            "Modelling Method": "Kinematic Capacity & Weather Sensitivity Analysis",
            "Key Assumptions": f"Assumes {closure_intensity_pct:.0f}% capacity throttling for {closure_duration_mins} min; {weather_cfg['description']}"
        },
        {
            "Metric": "Traffic Condition Classification",
            "Baseline Value": baseline_condition,
            "Scenario Value": scenario_condition,
            "Absolute Difference": "Level Transition" if baseline_condition != scenario_condition else "No Change",
            "Relative Change": "N/A",
            "Data Category": "Hypothetical Scenario Estimate",
            "Modelling Method": "Level of Service (LOS) Multi-Class Rule Mapping",
            "Key Assumptions": "Free-Flow >= 55 mph, Moderate 35-55 mph, Congested < 35 mph"
        },
        {
            "Metric": "Network Shockwave Reach",
            "Baseline Value": "0 Adjacent Links",
            "Scenario Value": f"{len(affected_neighbors)} Downstream/Upstream Links",
            "Absolute Difference": f"+{len(affected_neighbors)} Links",
            "Relative Change": "N/A",
            "Data Category": "Graph Adjacency Analysis",
            "Modelling Method": "Directed Spatial Adjacency Propagation (`adj_mx.pkl`)",
            "Key Assumptions": f"Queue spillover propagates to links within {closure_duration_mins} min shockwave boundary"
        }
    ]
    
    return {
        "target_node_id": node_idx,
        "baseline_speed": pred_speed,
        "scenario_speed": scenario_speed,
        "speed_difference": abs_speed_diff,
        "baseline_condition": baseline_condition,
        "scenario_condition": scenario_condition,
        "affected_neighbors": affected_neighbors,
        "comparison_table": pd.DataFrame(comparison_rows),
        "methodology_disclaimer": "Scenario outcomes represent a deterministic sensitivity simulation utilizing the highway network graph. They are NOT observed real-time traffic incidents."
    }
