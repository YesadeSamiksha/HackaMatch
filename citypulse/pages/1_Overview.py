"""
1_Overview.py - Screen 1: Highway Network Overview
Presents network-wide status, condition distribution, interactive Mapbox map,
Top 10 critical hotspot table, and historical domain coverage details.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores
from src.mapping import create_sensor_map, get_mapbox_token
from config import GEOGRAPHIC_SCOPE, TIME_SPAN, TOTAL_MONITORED_NODES, SAMPLING_INTERVAL

# Page Title & Header
st.title("TrafficPulse AI — Network Overview")
st.caption("Predictive traffic monitoring and historical telemetry intelligence across the Los Angeles County highway grid (Caltrans PeMS District 7).")

# Domain Coverage Card
with st.container(border=True):
    col_d1, col_d2 = st.columns([7, 3])
    with col_d1:
        st.markdown(f"**🛣️ {GEOGRAPHIC_SCOPE}**")
        st.caption(f"{TOTAL_MONITORED_NODES} Monitored Loop Detector Stations • {SAMPLING_INTERVAL} Cadence")
    with col_d2:
        st.info(f"**Telemetry Window:** {TIME_SPAN}\n\n*Historical Observations*", icon="📅")

# 1. Load Data & Compute Dynamic Hotspot Rankings (Cached)
@st.cache_data(ttl=600)
def load_overview_data():
    snapshot = get_network_snapshot()
    predictions = predict_network_state(snapshot)
    ranked = calculate_hotspot_priority_scores(predictions)
    return ranked

try:
    with st.spinner("Loading highway network telemetry..."):
        network_df = load_overview_data()
    if network_df.get("is_fallback_mode", pd.Series([False])).iloc[0]:
        st.info(
            "Running in Deployment Preview Mode: Displaying verified telemetry snapshot (207 stations). "
            "To deploy the full archive to Streamlit Cloud, see Git LFS instructions in README.md."
        )
except FileNotFoundError as fnf_err:
    st.error("Primary Dataset Not Found")
    st.write(f"Diagnostic Error: {fnf_err}")
    st.stop()
except Exception as e:
    st.error(f"Error loading network data: {e}")
    st.stop()

# 2. Executive Summary Metrics
total_sensors = len(network_df)
crit_high_count = len(network_df[network_df["severity_tier"].isin(["Critical", "High"])])
avg_speed = network_df["current_speed"].mean()
free_flow_pct = (len(network_df[network_df["pred_condition"] == "Low"]) / total_sensors) * 100.0
latest_timestamp = str(network_df["t0_timestamp"].iloc[0]) if "t0_timestamp" in network_df.columns else "2012-05-23T06:40:00"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Monitored Sensors", f"{total_sensors}", "Caltrans PeMS D7")
k2.metric("Critical Hotspots", f"{crit_high_count}", f"{(crit_high_count/total_sensors)*100:.1f}% of network", delta_color="inverse")
k3.metric("Free-Flow Ratio", f"{free_flow_pct:.1f}%", ">= 55.0 mph")
k4.metric("Network Mean Speed", f"{avg_speed:.1f} mph", "65 mph benchmark")
k5.metric("Observation Timestamp", latest_timestamp.split("T")[-1][:5], latest_timestamp.split("T")[0])

st.divider()

# 3. Interactive Filter Bar
col_f1, col_f2, col_f3, col_f4 = st.columns([3, 2, 2, 3])
with col_f1:
    corridors = ["All Freeway Corridors"] + sorted(list(network_df["corridor"].dropna().unique()))
    sel_corridor = st.selectbox("Corridor Filter", corridors)
with col_f2:
    tiers = ["All Severity Tiers", "Critical", "High", "Medium", "Low"]
    sel_tier = st.selectbox("Severity Tier Filter", tiers)
with col_f3:
    mapbox_token = get_mapbox_token()
    available_styles = ["open-street-map", "dark", "streets", "light"] if mapbox_token else ["open-street-map"]
    map_style = st.selectbox("Map Cartography", available_styles)

# Filter dataset
filtered_df = network_df.copy()
if sel_corridor != "All Freeway Corridors":
    filtered_df = filtered_df[filtered_df["corridor"] == sel_corridor]
if sel_tier != "All Severity Tiers":
    filtered_df = filtered_df[filtered_df["severity_tier"] == sel_tier]

with col_f4:
    station_options = ["None (Show All Stations)"] + [
        f"Station #{row['sensor_id']} ({str(row['corridor']).split(' (')[0]})"
        for _, row in filtered_df.iterrows()
    ]
    station_node_map = {
        f"Station #{row['sensor_id']} ({str(row['corridor']).split(' (')[0]})": int(row["node_id"])
        for _, row in filtered_df.iterrows()
    }
    sel_station_label = st.selectbox("Inspect / Highlight Station", station_options)
    selected_node = station_node_map.get(sel_station_label, None)

# Pinned Station Inspector Card (when a station is selected)
if selected_node is not None:
    st_row = network_df[network_df["node_id"] == selected_node].iloc[0]
    with st.container(border=True):
        st.subheader(f"📍 Selected Station #{st_row['sensor_id']} (Node {selected_node})")
        st.caption(f"**Corridor:** {st_row['corridor']}")
        
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("Current Speed", f"{st_row['current_speed']:.1f} mph")
        c_i2.metric("Forecast (t+15m)", f"{st_row['pred_speed_15m']:.1f} mph", f"{st_row['speed_change_15m']:+.1f} mph", delta_color="inverse")
        c_i3.metric("Predicted State", f"{st_row['pred_condition']}", f"{st_row['confidence_score']:.0f}% confidence")
        c_i4.metric("Priority Tier", f"{st_row['severity_tier']}", f"{st_row['priority_score']:.1f}/100 score")
        
        st.write(f"**Ranking Reason:** {st_row['ranking_reason']}")
        st.caption(f"**Recommended Action:** {st_row['recommended_investigation']}")

# 4. Map & Condition Distribution Row
map_col, dist_col = st.columns([7, 3])

with map_col:
    st.subheader(f"Interactive Highway Network Map ({len(filtered_df)} stations shown)")
    fig_map = create_sensor_map(
        filtered_df, 
        selected_node_id=selected_node,
        map_style=map_style, 
        height=520
    )
    st.plotly_chart(fig_map, use_container_width=True)

with dist_col:
    st.subheader("Traffic Distribution")
    cond_counts = network_df["pred_condition"].value_counts().reset_index()
    cond_counts.columns = ["Condition", "Count"]
    fig_donut = px.pie(
        cond_counts,
        values="Count",
        names="Condition",
        hole=0.6,
        color="Condition",
        color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"}
    )
    fig_donut.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8")
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    
    st.subheader("Severity Breakdown")
    tier_counts = network_df["severity_tier"].value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0).reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig_bar = px.bar(
        tier_counts,
        x="Tier",
        y="Count",
        color="Tier",
        color_discrete_map={"Critical": "#DC2626", "High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
    )
    fig_bar.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="", tickfont=dict(color="#94A3B8"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# 5. Top 10 Critical Hotspots Table
st.subheader("🔥 Top 10 Critical Highway Hotspots")
st.caption("Ranked by multi-factor priority score combining speed deficit below 65 mph, predicted deceleration, historical recurrence, and queue persistence.")

top10 = network_df.head(10)[[
    "hotspot_rank", "sensor_id", "corridor", "severity_tier", "priority_score",
    "current_speed", "pred_speed_15m", "speed_change_15m", "ranking_reason", "recommended_investigation"
]].copy()

top10.columns = [
    "Rank", "Station ID", "Freeway Corridor", "Severity Tier", "Priority Score",
    "Current Speed", "Forecast (t+15m)", "Trend", "Main Ranking Reason", "Recommended Investigation"
]

st.dataframe(top10, use_container_width=True, hide_index=True)

# 6. Technical Provenance & Calibration Metadata
with st.expander("ℹ️ Technical Highway Domain & Calibration Documentation", expanded=False):
    st.markdown("""
    **Dataset & Highway Sensor Network Architecture:**
    - **Telemetry Source:** Caltrans Performance Measurement System (PeMS) District 7 highway loop telemetry.
    - **Monitored Domain:** 207 verified dual-loop detector stations across Los Angeles County freeways (I-5, I-10, US-101, I-405, I-210, SR-134, SR-110, SR-60).
    - **Observation Window:** March 1, 2012 to May 23, 2012 (historical training and evaluation benchmark).
    - **Sampling Cadence:** Aggregated 5-minute telemetry intervals (speed, volume, occupancy).
    - **Model Training:** Chronological train/test split at May 1, 2012. Models evaluated strictly on held-out post-split observations.
    """)
