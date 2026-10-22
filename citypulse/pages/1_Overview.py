"""
1_Overview.py - Screen 1: Highway Network Overview
Presents network-wide status, condition distribution, interactive Mapbox map,
Top 10 critical hotspot table, and prominent historical period badge.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores
from src.mapping import create_sensor_map

# Page Header
st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
        TrafficPulse AI — Network Overview
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #94A3B8;">
        Real-time intelligence and predictive traffic monitoring across the Los Angeles County highway grid (Caltrans PeMS District 7).
    </p>
</div>
""", unsafe_allow_html=True)

# Historical Data Provenance Banner
from config import HF_DATASET_REPO_ID

st.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1E293B; border-radius: 10px; padding: 10px 16px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.1rem;">🤗</span>
        <span style="font-size: 0.82rem; color: #E2E8F0; font-weight: 600;">Dataset Source: Hugging Face (<code>{HF_DATASET_REPO_ID}</code>)</span>
    </div>
    <div style="font-size: 0.78rem; color: #94A3B8;">
        Coverage Window: <b style="color: #38BDF8;">March 1, 2012 – May 23, 2012</b> (Historical Caltrans PeMS D7 Loop Telemetry)
    </div>
</div>
""", unsafe_allow_html=True)

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
            "ℹ️ **Running in Deployment Preview Mode:** Displaying verified telemetry snapshot (207 stations). "
            "The full 157 MB historical archive (`data/train.parquet`) was excluded from Git because it exceeds GitHub's 100 MB limit. "
            "To deploy the full archive to Streamlit Cloud, see **Git LFS** instructions in `README.md`."
        )
except FileNotFoundError as fnf_err:
    st.error("### ❌ Primary Dataset Not Found")
    st.markdown(f"""
    **Diagnostic Details:**
    - **Error:** `{fnf_err}`
    - **Expected Location:** `citypulse/data/train.parquet` (157 MB, 4,962,618 rows)
    - **Root Cause:** GitHub limits single files to 100 MB. `train.parquet` exists in your local development workspace, but was excluded from Git.
    - **How to provide the full file on Streamlit Cloud:**
      1. Track via Git LFS: `git lfs track "citypulse/data/train.parquet"`
      2. Or mount cloud object storage and set `CITYPULSE_DATA_DIR` in your deployment environment variables.
    """)
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
k4.metric("Network Mean Speed", f"{avg_speed:.1f} mph", "65 mph design speed")
k5.metric("Observation Timestamp", latest_timestamp.split("T")[-1][:5], latest_timestamp.split("T")[0])

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 3. Interactive Filter Bar
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    corridors = ["All Freeway Corridors"] + sorted(list(network_df["corridor"].dropna().unique()))
    sel_corridor = st.selectbox("Corridor Filter", corridors)
with col_f2:
    tiers = ["All Severity Tiers", "Critical", "High", "Medium", "Low"]
    sel_tier = st.selectbox("Severity Tier Filter", tiers)
with col_f3:
    map_style = st.selectbox("Map Cartography", ["carto-positron", "open-street-map", "carto-darkmatter"])

filtered_df = network_df.copy()
if sel_corridor != "All Freeway Corridors":
    filtered_df = filtered_df[filtered_df["corridor"] == sel_corridor]
if sel_tier != "All Severity Tiers":
    filtered_df = filtered_df[filtered_df["severity_tier"] == sel_tier]

# 4. Map & Condition Distribution Row
map_col, dist_col = st.columns([7, 3])

with map_col:
    st.markdown(f"**Interactive Highway Map** ({len(filtered_df)} stations shown)")
    fig_map = create_sensor_map(filtered_df, map_style=map_style, height=480)
    st.plotly_chart(fig_map, use_container_width=True)

with dist_col:
    st.markdown("**Predicted Traffic Distribution**")
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
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8")
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("**Hotspot Severity Breakdown**")
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
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="", tickfont=dict(color="#94A3B8"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 5. Top 10 Critical Hotspots Table
st.markdown("### 🔥 Top 10 Critical Highway Hotspots")
st.caption("Ranked by multi-factor priority score combining speed deficit below 65 mph, predicted deceleration, historical recurrence, and queue persistence.")

top10 = network_df.head(10)[[
    "hotspot_rank", "sensor_id", "corridor", "severity_tier", "priority_score",
    "current_speed", "pred_speed_15m", "speed_change_15m", "ranking_reason", "recommended_investigation"
]].copy()

top10.columns = [
    "Rank", "Station ID", "Freeway Corridor", "Severity Tier", "Priority Score",
    "Current Speed", "Forecast (t+15m)", "Trend", "Main Ranking Reason", "Recommended Investigation"
]

def style_tier(val):
    if val == "Critical":
        return "background-color: rgba(220, 38, 38, 0.25); color: #FCA5A5; font-weight: bold;"
    elif val == "High":
        return "background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5;"
    elif val == "Medium":
        return "background-color: rgba(245, 158, 11, 0.2); color: #FDE68A;"
    return "color: #A7F3D0;"

styled_top10 = top10.style.applymap(style_tier, subset=["Severity Tier"])
st.dataframe(styled_top10, use_container_width=True, hide_index=True)
