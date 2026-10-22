"""
hotspot_ranking.py - Dynamic Traffic Hotspot Ranking Engine for TrafficPulse AI
Computes transparent, configurable, multi-factor priority scores [0 to 100]
combining severity, predicted deterioration, historical recurrence, and persistence.
"""

import pandas as pd
import numpy as np

# Configurable default component weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "severity": 0.30,
    "deterioration": 0.25,
    "recurrence": 0.25,
    "persistence": 0.20
}

# Configurable severity tier thresholds
DEFAULT_TIER_THRESHOLDS = {
    "Critical": 75.0,
    "High": 50.0,
    "Medium": 25.0
}

def calculate_hotspot_priority_scores(
    network_df: pd.DataFrame,
    historical_recurrence_map: dict = None,
    weights: dict = None,
    tier_thresholds: dict = None,
    reference_free_flow_speed: float = 65.0
) -> pd.DataFrame:
    """
    Computes normalized multi-factor priority scores [0, 100] for each sensor node.
    
    Formula:
      Priority Score = 100 * (
          w_sev * Severity Score +
          w_det * Predicted Deterioration Score +
          w_rec * Historical Recurrence Score +
          w_per * Persistence Score
      )
    
    All component scores are normalized strictly to [0.0, 1.0].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if tier_thresholds is None:
        tier_thresholds = DEFAULT_TIER_THRESHOLDS.copy()
        
    # Validate and renormalize weights if needed
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-3) and weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}
        
    df = network_df.copy()
    
    # 1. Severity Score: Proportional speed deficit below reference free-flow speed (65 mph)
    curr_speeds = df["current_speed"].clip(lower=0.0, upper=reference_free_flow_speed)
    speed_deficits = (reference_free_flow_speed - curr_speeds).clip(lower=0.0)
    score_severity = (speed_deficits / reference_free_flow_speed).clip(0.0, 1.0)
    
    # 2. Predicted Deterioration Score: Rate of projected deceleration in next 15 minutes
    # Negative speed change means worsening traffic
    speed_changes = df["speed_change_15m"]
    # Normalize: a deceleration of 15+ mph gives max deterioration score of 1.0
    score_deterioration = (-speed_changes.clip(upper=0.0) / 15.0).clip(0.0, 1.0)
    # Give a small baseline deterioration score if already severely congested
    score_deterioration = np.where(curr_speeds < 35.0, np.maximum(score_deterioration, 0.4), score_deterioration)
    
    # 3. Historical Recurrence Score: Frequency of congestion in historical observations
    if historical_recurrence_map is not None:
        score_recurrence = df["node_id"].map(historical_recurrence_map).fillna(0.15).clip(0.0, 1.0).values
    else:
        # Default proxy if historical map is not supplied: derived from diurnal peak flag & node severity
        score_recurrence = np.where(curr_speeds < 40.0, 0.65, 0.15)
        
    # 4. Persistence Score: How sustained the low speeds are across past 60 min lookback window
    # Count how many of past 12 steps (5 to 60 min prior) were below 45 mph
    lookback_cols = [f"x_t-{i}_d0" for i in range(1, 12)] + ["x_t+0_d0"]
    available_lb_cols = [c for c in lookback_cols if c in df.columns]
    if available_lb_cols:
        slow_steps = (df[available_lb_cols] < 45.0).sum(axis=1)
        score_persistence = (slow_steps / len(available_lb_cols)).clip(0.0, 1.0)
    else:
        # Fallback persistence based on 30m rolling mean
        score_persistence = (np.maximum(0.0, 50.0 - df.get("speed_mean_30m", curr_speeds)) / 50.0).clip(0.0, 1.0)
        
    # Component Scores Normalized to [0.0, 1.0]
    df["score_severity"] = np.round(score_severity, 3)
    df["score_deterioration"] = np.round(score_deterioration, 3)
    df["score_recurrence"] = np.round(score_recurrence, 3)
    df["score_persistence"] = np.round(score_persistence, 3)
    
    # Final Normalized Priority Score [0 to 100]
    priority_raw = (
        weights["severity"] * score_severity +
        weights["deterioration"] * score_deterioration +
        weights["recurrence"] * score_recurrence +
        weights["persistence"] * score_persistence
    )
    df["priority_score"] = np.round(priority_raw * 100.0, 1)
    
    # Severity Tiers
    tiers = []
    ranking_reasons = []
    recommended_investigations = []
    
    crit_th = tier_thresholds.get("Critical", 75.0)
    high_th = tier_thresholds.get("High", 50.0)
    med_th = tier_thresholds.get("Medium", 25.0)
    
    for _, row in df.iterrows():
        p = row["priority_score"]
        curr_v = row["current_speed"]
        pred_v = row["pred_speed_15m"]
        s_sev = row["score_severity"]
        s_det = row["score_deterioration"]
        s_rec = row["score_recurrence"]
        s_per = row["score_persistence"]
        
        # Primary Ranking Reason
        reasons = []
        if s_sev >= 0.5:
            reasons.append(f"Severe speed drop ({curr_v:.1f} mph vs {reference_free_flow_speed:.0f} mph benchmark)")
        if s_det >= 0.4:
            reasons.append(f"Active queue buildup (predicted {pred_v:.1f} mph)")
        if s_per >= 0.5:
            reasons.append("Sustained queueing over past 60 minutes")
        if s_rec >= 0.5:
            reasons.append(f"Recurring congestion history ({s_rec*100:.0f}% frequency)")
            
        main_reason = "; ".join(reasons) if reasons else "Normal operating conditions within speed thresholds"
        ranking_reasons.append(main_reason)
        
        # Severity Tier & Next Action
        if p >= crit_th:
            tier = "Critical"
            action = "Urgent: Dispatch patrol to confirm incident; check upstream ramp meters and activate dynamic message signs."
        elif p >= high_th:
            tier = "High"
            action = "Prioritize: Review bottleneck queue propagation and adjust signal/metering timing."
        elif p >= med_th:
            tier = "Medium"
            action = "Monitor: Track downstream sensor speeds to detect potential spillback early."
        else:
            tier = "Low"
            action = "Routine: Continuous automated sensor polling; normal free-flow operations."
            
        tiers.append(tier)
        recommended_investigations.append(action)
        
    df["severity_tier"] = tiers
    df["ranking_reason"] = ranking_reasons
    df["recommended_investigation"] = recommended_investigations
    
    # Sort deterministically by priority score descending, then by speed ascending
    df_ranked = df.sort_values(by=["priority_score", "current_speed"], ascending=[False, True]).reset_index(drop=True)
    df_ranked["hotspot_rank"] = df_ranked.index + 1
    
    return df_ranked
