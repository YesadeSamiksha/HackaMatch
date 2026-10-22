"""
2_Hotspot_Intelligence.py - Screen 2: Recurring Hotspot & Rush-Hour Intelligence
Presents hourly and day-of-week congestion heatmaps, morning vs evening peak comparisons,
persistent hotspot rankings, and 3 evidence-backed automated insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_network_snapshot, get_sensor_history
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores
from src.traffic_patterns import analyze_traffic_patterns
from src.explainability import explain_node_prediction

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
        Recurring Hotspots & Rush-Hour Intelligence
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #94A3B8;">
        Historical recurrence patterns, peak-hour bottlenecks, and evidence-backed rush-hour diagnostics.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Pattern Analytics (Cached)
@st.cache_data(ttl=600)
def load_patterns():
    patterns = analyze_traffic_patterns(sample_size=120000)
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    ranked = calculate_hotspot_priority_scores(preds, historical_recurrence_map=patterns["recurrence_map"])
    return patterns, ranked

try:
    with st.spinner("Analyzing historical traffic recurrence patterns..."):
        patterns, ranked_df = load_patterns()
except Exception as e:
    st.error(f"Error computing recurring traffic patterns: {e}")
    st.stop()

# 2. Key Rush-Hour Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Morning Peak Speed", f"{patterns['morn_speed']:.1f} mph", "7:00 AM – 10:00 AM")
m2.metric("Evening Peak Speed", f"{patterns['eve_speed']:.1f} mph", "4:00 PM – 7:00 PM", delta_color="inverse")
m3.metric("Midday Off-Peak Speed", f"{patterns['midday_speed']:.1f} mph", "11:00 AM – 2:00 PM")
m4.metric("Persistent Hotspots", f"{len(patterns['persistent_ranking'][patterns['persistent_ranking']['is_persistent']])} Stations", ">= 15% Congestion Rate")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 3. Three Evidence-Backed Automated Insights
st.markdown("### 💡 Evidence-Backed Traffic Insights")
st.caption("Statistically validated comparisons extracted from historical PeMS loop telemetry.")

for ins in patterns["insights"]:
    st.markdown(f"""
    <div style="background: #111827; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
        <span style="font-size: 0.88rem; color: #F1F5F9; line-height: 1.5;">{ins}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 4. Heatmap & Hourly Trend Tabs
tab_heat, tab_hourly, tab_persistent, tab_sensor = st.tabs([
    "🗓️ Hour × Day Congestion Heatmap",
    "📈 24-Hour Diurnal Curve",
    "🏆 Persistent Hotspots Ranking",
    "🔍 Specific Location Diagnostic"
])

with tab_heat:
    st.markdown("#### Probability of Severe Congestion (< 35 mph) by Hour and Day of Week")
    heat_data = patterns["heatmap_congestion"]
    
    fig_heat = px.imshow(
        heat_data,
        labels=dict(x="Day of Week", y="Hour of Day (0–23)", color="Congestion %"),
        x=heat_data.columns,
        y=heat_data.index,
        color_continuous_scale="Reds",
        aspect="auto"
    )
    fig_heat.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"),
        yaxis=dict(tickmode="linear", tick0=0, dtick=2)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab_hourly:
    st.markdown("#### Network Average Speed Profile Across the 24-Hour Cycle")
    h_df = patterns["hourly_summary"]
    fig_line = go.Figure()
    
    # 25th-75th percentile band
    fig_line.add_trace(go.Scatter(
        x=list(h_df["hour"]) + list(h_df["hour"][::-1]),
        y=list(h_df["p75_speed"]) + list(h_df["p25_speed"][::-1]),
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th–75th Percentile Range"
    ))
    
    # Average speed line
    fig_line.add_trace(go.Scatter(
        x=h_df["hour"],
        y=h_df["avg_speed"],
        mode="lines+markers",
        name="Network Mean Speed (mph)",
        line=dict(color="#38BDF8", width=3)
    ))
    
    # 65 mph benchmark
    fig_line.add_hline(y=65.0, line_dash="dot", line_color="#10B981", annotation_text="Free-Flow (65 mph)")
    fig_line.add_hline(y=35.0, line_dash="dash", line_color="#EF4444", annotation_text="Congestion Threshold (35 mph)")
    
    fig_line.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of Day", gridcolor="#1E293B", tickmode="linear", tick0=0, dtick=2, tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="Operating Speed (mph)", gridcolor="#1E293B", tickfont=dict(color="#94A3B8")),
        font=dict(color="#F1F5F9")
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab_persistent:
    st.markdown("#### Ranked Persistent Bottlenecks (Historical Recurrence >= 15%)")
    p_df = patterns["persistent_ranking"].head(20)[[
        "rank", "sensor_id", "corridor", "avg_speed", "congestion_rate", 
        "morning_speed", "evening_speed", "obs_count"
    ]].copy()
    p_df.columns = [
        "Rank", "Station ID", "Corridor", "Historical Speed (mph)", 
        "Recurrence Rate", "Morning Peak (mph)", "Evening Peak (mph)", "Observations"
    ]
    p_df["Recurrence Rate"] = (p_df["Recurrence Rate"] * 100).round(1).astype(str) + "%"
    p_df["Historical Speed (mph)"] = p_df["Historical Speed (mph)"].round(1)
    p_df["Morning Peak (mph)"] = p_df["Morning Peak (mph)"].round(1)
    p_df["Evening Peak (mph)"] = p_df["Evening Peak (mph)"].round(1)
    st.dataframe(p_df, use_container_width=True, hide_index=True)

with tab_sensor:
    st.markdown("#### Detailed Location Speed Diagnostics")
    
    col_s1, col_s2 = st.columns([2, 3])
    with col_s1:
        corrs = ["All Corridors"] + sorted(list(ranked_df["corridor"].dropna().unique()))
        c_sel = st.selectbox("Filter Corridor", corrs, key="diag_corr")
        
    filtered_for_diag = ranked_df if c_sel == "All Corridors" else ranked_df[ranked_df["corridor"] == c_sel]
    node_opts = {
        f"Station #{row['sensor_id']} ({row['corridor']}) | {row['current_speed']:.0f} mph": int(row["node_id"])
        for _, row in filtered_for_diag.iterrows()
    }
    
    with col_s2:
        chosen_lbl = st.selectbox("Select Station", list(node_opts.keys()), key="diag_node")
        sel_node = node_opts[chosen_lbl]
        
    n_data = ranked_df[ranked_df["node_id"] == sel_node].iloc[0]
    exp = explain_node_prediction(n_data)
    
    c_e1, c_e2, c_e3 = st.columns(3)
    c_e1.metric("Current Speed", f"{n_data['current_speed']:.1f} mph")
    c_e2.metric("Forecast (t+15m)", f"{n_data['pred_speed_15m']:.1f} mph", f"{n_data['speed_change_15m']:+.1f} mph")
    c_e3.metric("Recurrence Rate", f"{n_data.get('score_recurrence', 0.2)*100:.1f}%")
    
    # 24h speed history
    h_data = get_sensor_history(sel_node, num_samples=288)
    if not h_data.empty:
        fig_s = px.line(h_data, x="timestamp", y="x_t+0_d0", title=f"24-Hour History for Station #{n_data['sensor_id']}")
        fig_s.add_hline(y=65.0, line_dash="dot", line_color="#10B981")
        fig_s.add_hline(y=35.0, line_dash="dash", line_color="#EF4444")
        fig_s.update_layout(
            height=280, 
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8")
        )
        st.plotly_chart(fig_s, use_container_width=True)
