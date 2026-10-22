"""
3_Prediction_Lab.py - Screen 3: Prediction & Simulation Lab
Interactive multi-horizon forecasting, hypothetical scenario modeling with graph spillback,
and dynamic ranking weight sensitivity explorer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores, DEFAULT_WEIGHTS, DEFAULT_TIER_THRESHOLDS
from src.scenario_engine import simulate_scenario, WEATHER_IMPACT_FACTORS
from src.traffic_patterns import analyze_traffic_patterns

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
        Prediction & Simulation Lab
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #94A3B8;">
        Explore multi-horizon forecasts, test hypothetical road-closure scenarios, and adjust ranking policy weights interactively.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Data
@st.cache_data(ttl=600)
def load_lab_data():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    patterns = analyze_traffic_patterns(sample_size=100000)
    return preds, patterns["recurrence_map"]

try:
    with st.spinner("Initializing prediction engine..."):
        predictions_df, rec_map = load_lab_data()
except Exception as e:
    st.error(f"Error initializing lab: {e}")
    st.stop()

# 2. Lab Modes (Tabs)
tab_scen, tab_weights, tab_multi = st.tabs([
    "🧪 What-If Scenario Simulation",
    "⚖️ Ranking Weight Sensitivity Explorer",
    "⏱️ Multi-Horizon Speed Forecasts"
])

# -------------------------------------------------------------
# TAB 1: What-If Scenario Simulator
# -------------------------------------------------------------
with tab_scen:
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
        <span style="font-weight: 700; color: #F59E0B; font-size: 0.82rem;">⚠️ SIMULATION DISCLAIMER:</span>
        <span style="font-size: 0.8rem; color: #FDE68A;">
            Hypothetical scenario outcomes are deterministic sensitivity calculations using network capacity models and graph adjacency.
            They are NOT observed real-time incidents.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        node_opts = {
            f"Station #{row['sensor_id']} ({row['corridor']})": int(row["node_id"])
            for _, row in predictions_df.iterrows()
        }
        chosen_sim_lbl = st.selectbox("Select Target Station", list(node_opts.keys()), index=0)
        target_sim_node = node_opts[chosen_sim_lbl]
        
    with col_w2:
        closure_throttling = st.slider("Capacity Throttling (%)", min_value=0, max_value=100, value=50, step=10, help="50% = 2 of 4 lanes closed; 100% = Full Closure")
        
    with col_w3:
        sim_duration = st.select_slider("Assumed Duration", options=[15, 30, 45, 60, 90, 120], value=30, format_func=lambda x: f"{x} min")
        
    with col_w4:
        sim_weather = st.selectbox("Hypothetical Weather", list(WEATHER_IMPACT_FACTORS.keys()), index=0)
        
    # Execute simulation
    target_row = predictions_df[predictions_df["node_id"] == target_sim_node].iloc[0]
    sim_output = simulate_scenario(
        target_node_id=target_sim_node,
        baseline_node_data=target_row,
        closure_intensity_pct=closure_throttling,
        closure_duration_mins=sim_duration,
        weather_condition=sim_weather
    )
    
    # Quantitative Comparison Metrics
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Baseline Forecast", f"{sim_output['baseline_speed']:.1f} mph", sim_output["baseline_condition"])
    rc2.metric("Scenario Speed", f"{sim_output['scenario_speed']:.1f} mph", f"{sim_output['speed_difference']:+.1f} mph", delta_color="inverse")
    rc3.metric("Spillover Links", f"{len(sim_output['affected_neighbors'])} Connected Links", "adj_mx.pkl Reach")
    rc4.metric("Incident Impact", "CRITICAL" if sim_output["scenario_speed"] < 30 else ("MODERATE" if sim_output["scenario_speed"] < 50 else "MINIMAL"), f"-{closure_throttling}% capacity", delta_color="inverse")
    
    st.markdown("#### Structured Evidence & Assumption Audit")
    st.dataframe(sim_output["comparison_table"], use_container_width=True, hide_index=True)
    
    # Spatial Network Shockwave Reach
    if sim_output["affected_neighbors"]:
        st.markdown("#### Spatial Queue Spillback on Connected Highway Links")
        neigh_df = pd.DataFrame(sim_output["affected_neighbors"])
        neigh_df.columns = ["Neighbor Node", "Sensor Station", "Network Affinity", "Estimated Delay / Speed Drop (mph)"]
        
        cn1, cn2 = st.columns([5, 5])
        with cn1:
            st.dataframe(neigh_df, use_container_width=True, hide_index=True)
        with cn2:
            fig_bar_spill = px.bar(
                neigh_df,
                x="Sensor Station",
                y="Estimated Delay / Speed Drop (mph)",
                color="Network Affinity",
                color_continuous_scale="Reds",
                title="Spillback Speed Deceleration"
            )
            fig_bar_spill.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8")
            )
            st.plotly_chart(fig_bar_spill, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Dynamic Weight Sensitivity Explorer
# -------------------------------------------------------------
with tab_weights:
    st.markdown("#### Interactive Hotspot Policy Weight Tuning")
    st.caption("Adjust the relative importance of severity, deterioration, historical recurrence, and queue persistence to see how priority rankings shift in real time.")
    
    col_wt1, col_wt2, col_wt3, col_wt4 = st.columns(4)
    with col_wt1:
        w_sev = st.slider("Severity Weight", 0.0, 1.0, DEFAULT_WEIGHTS["severity"], 0.05, help="Speed drop below 65 mph benchmark")
    with col_wt2:
        w_det = st.slider("Deterioration Weight", 0.0, 1.0, DEFAULT_WEIGHTS["deterioration"], 0.05, help="Projected 15-minute deceleration")
    with col_wt3:
        w_rec = st.slider("Recurrence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["recurrence"], 0.05, help="Historical frequency of congestion")
    with col_wt4:
        w_per = st.slider("Persistence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["persistence"], 0.05, help="Sustained queueing in lookback window")
        
    custom_weights = {
        "severity": w_sev,
        "deterioration": w_det,
        "recurrence": w_rec,
        "persistence": w_per
    }
    
    total_w = sum(custom_weights.values())
    if total_w == 0:
        st.warning("Total weight cannot be zero. Resetting to defaults.")
        custom_weights = DEFAULT_WEIGHTS.copy()
    else:
        st.info(f"Normalized Weights: Severity {w_sev/total_w*100:.0f}% • Deterioration {w_det/total_w*100:.0f}% • Recurrence {w_rec/total_w*100:.0f}% • Persistence {w_per/total_w*100:.0f}%")
        
    custom_ranked = calculate_hotspot_priority_scores(
        predictions_df,
        historical_recurrence_map=rec_map,
        weights=custom_weights
    )
    
    st.markdown(f"**Top 10 Hotspots with Custom Policy:**")
    st.dataframe(
        custom_ranked.head(10)[["hotspot_rank", "sensor_id", "corridor", "severity_tier", "priority_score", "current_speed", "pred_speed_15m", "ranking_reason"]],
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------------------
# TAB 3: Multi-Horizon Speed Forecasts
# -------------------------------------------------------------
with tab_multi:
    st.markdown("#### Continuous Speed Forecasts (15-min and 30-min Horizons)")
    st.caption("Forecast outputs generated from HistGradientBoosting multi-step regression.")
    
    multi_df = predictions_df[[
        "node_id", "sensor_id", "corridor", "current_speed", 
        "pred_speed_15m", "speed_change_15m", "pred_condition", "confidence_score"
    ]].copy()
    
    # 30-minute estimated horizon (extrapolating 15-min kinematic trend with mean reversion)
    multi_df["est_speed_30m"] = (multi_df["pred_speed_15m"] + (multi_df["speed_change_15m"] * 0.5)).clip(0.0, 70.0).round(1)
    
    multi_df.columns = [
        "Node", "Station", "Corridor", "Current Speed (mph)", 
        "Forecast t+15m (mph)", "Trend 15m", "Predicted State", "Model Confidence", "Forecast t+30m (mph)"
    ]
    st.dataframe(multi_df.head(25), use_container_width=True, hide_index=True)
