"""
1_City_Overview.py - Screen 1: Highway Network City Overview
Displays network-wide KPIs, interactive Mapbox sensor map, risk distributions, and priority segments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_network_snapshot, get_sensor_locations
from src.inference import predict_network_state, COLOR_MAP
from src.priority_engine import compute_priority_rankings
from src.mapping import create_sensor_map

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #FFFFFF;">
        Highway Network Overview
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
        Real-time intelligence and predictive traffic monitoring across the Los Angeles County highway grid (Caltrans PeMS District 7).
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Data & Run Inference (Cached)
@st.cache_data(ttl=600)
def load_and_predict():
    snapshot = get_network_snapshot()
    predictions = predict_network_state(snapshot)
    ranked = compute_priority_rankings(predictions)
    return ranked

try:
    with st.spinner("Analyzing 207 highway sensor stations with HistGradientBoosting models..."):
        network_df = load_and_predict()
except Exception as e:
    st.error(f"Error loading network state: {e}")
    st.stop()

# 2. Executive KPI Summary Cards
total_sensors = len(network_df)
high_priority_count = len(network_df[network_df["priority_tier"] == "High Priority"])
med_priority_count = len(network_df[network_df["priority_tier"] == "Medium Priority"])
low_priority_count = len(network_df[network_df["priority_tier"] == "Low Priority"])
avg_current_speed = network_df["current_speed"].mean()
avg_forecast_speed = network_df["pred_speed_15m"].mean()
free_flow_pct = (len(network_df[network_df["pred_condition"] == "Low"]) / total_sensors) * 100.0

latest_ts = network_df["t0_timestamp"].iloc[0] if "t0_timestamp" in network_df.columns else "2012-05-23T06:40:00"

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Monitored Sensors", f"{total_sensors}", "Caltrans PeMS D7")
kpi2.metric("Critical Bottlenecks", f"{high_priority_count}", f"{high_priority_count/total_sensors*100:.1f}% network", delta_color="inverse")
kpi3.metric("Free-Flow Ratio", f"{free_flow_pct:.1f}%", ">= 55.0 mph")
kpi4.metric("Network Mean Speed", f"{avg_current_speed:.1f} mph", f"Forecast: {avg_forecast_speed:.1f} mph")
kpi5.metric("Model Architecture", "HGB v1.0", "90.0% Test Acc")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 3. Interactive Filter & Map View
col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 2])
with col_filter1:
    corridors = ["All Corridors"] + sorted(list(network_df["corridor"].dropna().unique()))
    selected_corridor = st.selectbox("Freeway Corridor", corridors)
with col_filter2:
    conditions = ["All Conditions", "High (Congested)", "Moderate", "Low (Free-Flow)"]
    selected_condition = st.selectbox("Predicted Condition (t+15m)", conditions)
with col_filter3:
    map_theme = st.selectbox("Map Cartography", ["carto-positron", "open-street-map", "carto-darkmatter"])

# Filter dataframe for map and table
filtered_df = network_df.copy()
if selected_corridor != "All Corridors":
    filtered_df = filtered_df[filtered_df["corridor"] == selected_corridor]
if selected_condition != "All Conditions":
    short_cond = selected_condition.split(" ")[0]
    filtered_df = filtered_df[filtered_df["pred_condition"] == short_cond]

# Map and Distribution Row
map_col, chart_col = st.columns([7, 3])

with map_col:
    st.markdown(f"**Interactive Highway Map** ({len(filtered_df)} stations shown • click or hover points for details)")
    fig_map = create_sensor_map(filtered_df, map_style=map_theme, height=520)
    st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    st.markdown("**Predicted Condition Distribution**")
    cond_counts = network_df["pred_condition"].value_counts().reset_index()
    cond_counts.columns = ["Condition", "Count"]
    
    fig_donut = px.pie(
        cond_counts, 
        values="Count", 
        names="Condition",
        hole=0.55,
        color="Condition",
        color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"}
    )
    fig_donut.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8")
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("**Priority Tiers**")
    tier_counts = network_df["priority_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig_bar = px.bar(
        tier_counts,
        x="Tier",
        y="Count",
        color="Tier",
        color_discrete_map={"High Priority": "#EF4444", "Medium Priority": "#F59E0B", "Low Priority": "#10B981"}
    )
    fig_bar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="Sensors", tickfont=dict(color="#94A3B8"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 4. Highest-Priority Road Segments Table
st.markdown("### Highest-Priority Highway Bottlenecks (Decision-Support Ranking)")
st.caption("Ranked by multi-factor urgency index combining speed deficit below 65 mph, predicted congestion severity, and deceleration trend.")

display_cols = [
    "rank", "sensor_id", "node_id", "corridor", "current_speed", 
    "pred_speed_15m", "speed_change_15m", "pred_condition", 
    "confidence_score", "priority_tier", "priority_score"
]
table_df = filtered_df[display_cols].copy()
table_df.columns = [
    "Rank", "Sensor Station", "Node ID", "Freeway Corridor", "Current Speed (mph)", 
    "Forecast (t+15m)", "Trend", "Predicted State", 
    "Confidence", "Priority Tier", "Priority Score"
]

# Color formatting
def highlight_tier(val):
    if val == "High Priority":
        return "background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; font-weight: bold;"
    elif val == "Medium Priority":
        return "background-color: rgba(245, 158, 11, 0.2); color: #FDE68A;"
    return "color: #A7F3D0;"

styled_table = table_df.head(25).style.applymap(highlight_tier, subset=["Priority Tier"])
st.dataframe(styled_table, use_container_width=True, hide_index=True)

st.info(
    "💡 **Navigate to Screen 2 ('Explain a Hotspot')** to inspect individual sensor kinematics, 24-hour diurnal curves, "
    "and feature attribution diagnostics for any highlighted station above."
)
