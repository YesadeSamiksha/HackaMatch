"""
3_What_If_Lab.py - Screen 3: What-If Scenario Simulation Lab
Interactive simulation of hypothetical lane closures, capacity reductions, and adverse weather events.
Uses graph adjacency matrix (adj_mx.pkl) to project network spillover across highway corridors.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.scenario_engine import simulate_scenario, WEATHER_IMPACT_FACTORS
from src.mapping import create_sensor_map

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #FFFFFF;">
        What-If Scenario Simulation Lab
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
        Simulate hypothetical lane closures, arterial blockages, and adverse weather shockwaves.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 12px 16px; margin-bottom: 20px;">
    <span style="font-weight: 700; color: #F59E0B; font-size: 0.85rem;">⚠️ SCIENTIFIC NOTICE & SIMULATION DISCLAIMER:</span>
    <span style="font-size: 0.82rem; color: #FDE68A;">
        Scenario calculations represent deterministic sensitivity simulations based on highway capacity diagrams and graph adjacency.
        They are explicitly <b>NOT real-time observed road closures</b> or live weather advisories.
    </span>
</div>
""", unsafe_allow_html=True)

# 1. Load Data
@st.cache_data(ttl=600)
def load_data():
    snapshot = get_network_snapshot()
    predictions = predict_network_state(snapshot)
    return predictions

try:
    network_df = load_data()
except Exception as e:
    st.error(f"Error loading network data: {e}")
    st.stop()

# 2. Section A: Scenario Controls
st.markdown("### A. Scenario Parameters & Controls")

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

with ctrl_col1:
    node_list = [
        f"Node #{row['node_id']} — Station {row['sensor_id']} ({row['corridor']})"
        for _, row in network_df.iterrows()
    ]
    chosen_node_str = st.selectbox("Target Highway Sensor Station", node_list, index=0)
    target_node_id = int(chosen_node_str.split(" — ")[0].replace("Node #", ""))

with ctrl_col2:
    closure_pct = st.slider(
        "Hypothetical Lane Capacity Throttling", 
        min_value=0, 
        max_value=100, 
        value=50, 
        step=10,
        help="0% = Normal Flow; 50% = 2 of 4 lanes blocked; 100% = Full Highway Closure"
    )

with ctrl_col3:
    closure_duration = st.select_slider(
        "Incident Duration",
        options=[15, 30, 45, 60, 90, 120],
        value=30,
        format_func=lambda x: f"{x} minutes"
    )

with ctrl_col4:
    weather_cond = st.selectbox(
        "Hypothetical Weather Context",
        list(WEATHER_IMPACT_FACTORS.keys()),
        index=0
    )

# 3. Execute Simulation
node_row = network_df[network_df["node_id"] == target_node_id].iloc[0]
sim_result = simulate_scenario(
    target_node_id=target_node_id,
    baseline_node_data=node_row,
    closure_intensity_pct=closure_pct,
    closure_duration_mins=closure_duration,
    weather_condition=weather_cond
)

st.markdown("<hr style='border-color: #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)

# 4. Section B: Baseline vs Scenario Impact Summary
st.markdown("### B. Baseline vs Scenario Quantitative Comparison")

res_col1, res_col2, res_col3, res_col4 = st.columns(4)
res_col1.metric(
    "Baseline Forecast Speed", 
    f"{sim_result['baseline_speed']:.1f} mph", 
    f"State: {sim_result['baseline_condition']}"
)
res_col2.metric(
    "Simulated Scenario Speed", 
    f"{sim_result['scenario_speed']:.1f} mph", 
    f"{sim_result['speed_difference']:+.1f} mph ({sim_result['scenario_condition']})",
    delta_color="inverse"
)
res_col3.metric(
    "Network Shockwave Reach", 
    f"{len(sim_result['affected_neighbors'])} Adjacent Links", 
    f"adj_mx.pkl Spatial Reach"
)
res_col4.metric(
    "Impact Severity", 
    "CRITICAL" if sim_result['scenario_speed'] < 30 else ("MODERATE" if sim_result['scenario_speed'] < 50 else "MINIMAL"),
    f"Capacity: -{closure_pct}%",
    delta_color="inverse"
)

# Comparison Table
st.markdown("#### Structured Evidence & Assumption Audit")
df_comp = sim_result["comparison_table"]
st.dataframe(df_comp, use_container_width=True, hide_index=True)

# 5. Section C: Graph Network Spillover (adj_mx.pkl)
st.markdown("### C. Highway Network Spatial Shockwave & Spillover Analysis")
st.caption("Identifies adjacent upstream and downstream loop detector stations connected in the highway adjacency matrix (`adj_mx.pkl`).")

if sim_result["affected_neighbors"]:
    neigh_df = pd.DataFrame(sim_result["affected_neighbors"])
    neigh_df.columns = ["Neighbor Node", "Sensor Station", "Network Affinity Weight", "Estimated Delay / Speed Drop (mph)"]
    
    col_n1, col_n2 = st.columns([5, 5])
    with col_n1:
        st.markdown("**Connected Highway Links at Risk of Queue Spillback:**")
        st.dataframe(neigh_df, use_container_width=True, hide_index=True)
    with col_n2:
        fig_spill = px.bar(
            neigh_df,
            x="Sensor Station",
            y="Estimated Delay / Speed Drop (mph)",
            color="Network Affinity Weight",
            color_continuous_scale="Reds",
            title="Spillover Speed Deceleration by Connected Sensor Station"
        )
        fig_spill.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8")
        )
        st.plotly_chart(fig_spill, use_container_width=True)
else:
    st.info("No immediate adjacent highway links within threshold distance in adjacency matrix.")

st.caption(f"ℹ️ *{sim_result['methodology_disclaimer']}*")
