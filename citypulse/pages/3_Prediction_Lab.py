"""
3_Prediction_Lab.py - Screen 3: Prediction & Simulation Lab
Interactive multi-horizon forecasting, hypothetical scenario modeling with graph spillback,
and dynamic ranking weight sensitivity explorer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores, DEFAULT_WEIGHTS, DEFAULT_TIER_THRESHOLDS
from src.scenario_engine import simulate_scenario, WEATHER_IMPACT_FACTORS
from src.traffic_patterns import analyze_traffic_patterns

# Page Header
st.title("Prediction & Simulation Lab")
st.write(
    "Use this lab to explore how traffic conditions might change when a road's capacity is reduced, "
    "compare the expected impact of hypothetical incidents, and examine how forecasts evolve over time. "
    "Results are model estimates or scenario calculations, not confirmed real-world incidents."
)

# Plain Language Mode Guide Cards (Priority 4.A)
col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    with st.container(border=True):
        st.subheader("🧪 Scenario Simulation")
        st.write("**Use case:** Explore the potential impact of a temporary lane closure or adverse weather on a selected corridor and its connected segments.")
with col_g2:
    with st.container(border=True):
        st.subheader("⚖️ Ranking Sensitivity")
        st.write("**Use case:** Understand how changing congestion severity or forecast deterioration priorities changes the list of roads needing operational attention.")
with col_g3:
    with st.container(border=True):
        st.subheader("⏱️ Multi-Horizon Forecast")
        st.write("**Use case:** Compare estimated traffic speeds at supported future intervals, separating genuine ML model forecasts from kinematic approximations.")

# 1. Load Data
@st.cache_data(ttl=600)
def load_lab_data():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    patterns = analyze_traffic_patterns(sample_size=100000)
    return preds, patterns["recurrence_map"]

try:
    with st.spinner("Initializing prediction engine and highway network telemetry..."):
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
    st.info(
        "**Hypothetical Simulation Context:** All simulation inputs represent hypothetical what-if assumptions, "
        "NOT confirmed real-world observations. Calculations estimate how traffic speeds and queue shockwaves might propagate based on network capacity curves and graph adjacency.",
        icon="ℹ️"
    )
    
    # Meaningful, Validated Inputs (Priority 4.B)
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        node_opts = {
            f"Station #{row['sensor_id']} ({row['corridor'].split(' (')[0]})": int(row["node_id"])
            for _, row in predictions_df.iterrows()
        }
        chosen_sim_lbl = st.selectbox(
            "Target Highway Station", 
            list(node_opts.keys()), 
            index=0,
            help="The highway monitoring loop detector station where the hypothetical incident or capacity restriction is assumed to occur."
        )
        target_sim_node = node_opts[chosen_sim_lbl]
        
    with col_w2:
        closure_throttling = st.slider(
            "Capacity Reduction (%)", 
            min_value=0, 
            max_value=100, 
            value=50, 
            step=10, 
            help="Capacity reduction: the assumed percentage of road capacity temporarily unavailable. A 50% reduction is a hypothetical scenario (e.g., 2 of 4 lanes blocked), not a measurement of current capacity."
        )
        
    with col_w3:
        sim_duration = st.select_slider(
            "Assumed Duration (min)", 
            options=[15, 30, 45, 60, 90, 120], 
            value=30, 
            format_func=lambda x: f"{x} min",
            help="Assumed duration: the hypothetical elapsed time the capacity bottleneck persists. Longer durations increase the queue shockwave reach to connected highway segments."
        )
        
    with col_w4:
        weather_options = list(WEATHER_IMPACT_FACTORS.keys())
        sim_weather = st.selectbox(
            "Hypothetical Weather", 
            weather_options, 
            index=0,
            help="Hypothetical weather: assumed environmental condition impacting tire friction, driver headway, and free-flow operating speeds."
        )
        
    # Execute simulation with validated target row
    target_match = predictions_df[predictions_df["node_id"] == target_sim_node]
    if target_match.empty:
        st.error(f"Selected station Node #{target_sim_node} not found in current network snapshot.")
        st.stop()
        
    target_row = target_match.iloc[0]
    sim_output = simulate_scenario(
        target_node_id=target_sim_node,
        baseline_node_data=target_row,
        closure_intensity_pct=closure_throttling,
        closure_duration_mins=sim_duration,
        weather_condition=sim_weather
    )
    
    st.divider()
    
    # Results Presented in Exact Specified Order (Priority 4.D)
    st.subheader("📊 Scenario Impact Results")
    st.caption("Quantitative outcomes ordered strictly by baseline forecast, estimated scenario speed, difference, impact level, and connected segments.")
    
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric(
        "1. Baseline Forecast", 
        f"{sim_output['baseline_speed']:.1f} mph", 
        f"{sim_output['baseline_condition']}"
    )
    rc2.metric(
        "2. Estimated Scenario Speed", 
        f"{sim_output['scenario_speed']:.1f} mph", 
        f"{sim_output['speed_difference']:+.1f} mph vs baseline", 
        delta_color="inverse"
    )
    rc3.metric(
        "3. Difference from Baseline", 
        f"{sim_output['speed_difference']:+.1f} mph", 
        f"{(sim_output['speed_difference']/max(1.0, sim_output['baseline_speed']))*100:+.1f}% change"
    )
    rc4.metric(
        "4. Potential Impact Level", 
        sim_output["impact_level"], 
        f"-{closure_throttling}% capacity"
    )
    rc5.metric(
        "5. Connected Segments", 
        f"{len(sim_output['affected_neighbors'])} Segments", 
        "Network Graph Reach"
    )
    
    # 6. Key Assumptions and Explanatory Interpretation (Priority 4.D)
    with st.container(border=True):
        st.markdown("**6. Result Interpretation & Key Assumptions:**")
        st.write(sim_output["interpretation_narrative"])
        st.caption(
            "Operational Context: Lower operating speeds indicate worsening traffic congestion under standard Caltrans Level of Service (LOS) definitions: "
            "Free-Flow (>= 55 mph), Moderate Congestion (35–55 mph), and High Congestion / Breakdown (< 35 mph)."
        )
    
    # Compact Baseline-versus-Scenario Comparison Chart (Priority 4.D)
    col_chart, col_summary = st.columns([6, 4])
    
    with col_chart:
        st.subheader("Speed Comparison (mph)")
        speed_comp_df = pd.DataFrame({
            "Scenario State": ["Design Speed Benchmark", "Baseline Model Forecast", "Estimated Scenario Speed"],
            "Speed (mph)": [65.0, sim_output["baseline_speed"], sim_output["scenario_speed"]],
            "Type": ["Reference Benchmark", "Supervised ML Forecast", "Hypothetical Scenario Calculation"]
        })
        fig_comp = px.bar(
            speed_comp_df,
            x="Speed (mph)",
            y="Scenario State",
            orientation="h",
            color="Scenario State",
            color_discrete_map={
                "Design Speed Benchmark": "#64748B",
                "Baseline Model Forecast": "#38BDF8",
                "Estimated Scenario Speed": "#EF4444" if sim_output["scenario_speed"] < 35 else ("#F59E0B" if sim_output["scenario_speed"] < 55 else "#10B981")
            },
            text="Speed (mph)"
        )
        fig_comp.update_traces(texttemplate='%{text:.1f} mph', textposition='inside')
        fig_comp.update_layout(
            height=230,
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 75], gridcolor="#1E293B", title="Operating Speed (mph)"),
            yaxis=dict(title="", tickfont=dict(color="#F1F5F9", size=11))
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_summary:
        st.subheader("Simulation Summary")
        with st.container(border=True):
            st.write(f"**Target Corridor:** {target_row['corridor']}")
            st.write(f"**Capacity Reduction:** {closure_throttling}% unavailable")
            st.write(f"**Assumed Duration:** {sim_duration} minutes")
            st.write(f"**Assumed Weather:** {sim_weather}")
            st.write(f"**Impact Classification:** {sim_output['impact_level']}")
            st.caption(sim_output['impact_description'])
        
    # User-Friendly Comparison Table (Priority 4.C)
    st.subheader("Detailed Scenario Metrics")
    st.dataframe(sim_output["comparison_table"], use_container_width=True, hide_index=True)
    
    # Connected Road Segments Affected (Priority 4.C)
    st.subheader("Connected Road Segments Affected")
    st.caption(
        "Adjacent highway links evaluated using the highway network graph. "
        "These represent connected road segments considered by the model's spatial propagation formula, not confirmed real-world congestion."
    )
    
    if sim_output["affected_neighbors"]:
        neigh_df = pd.DataFrame(sim_output["affected_neighbors"])
        neigh_df.columns = [
            "Network Node ID", 
            "Connected Sensor Station", 
            "Network Proximity Weight", 
            "Estimated Speed Reduction (mph)"
        ]
        
        cn1, cn2 = st.columns([5, 5])
        with cn1:
            st.dataframe(neigh_df, use_container_width=True, hide_index=True)
        with cn2:
            fig_bar_spill = px.bar(
                neigh_df,
                x="Connected Sensor Station",
                y="Estimated Speed Reduction (mph)",
                color="Network Proximity Weight",
                color_continuous_scale="Reds",
                title="Estimated Delay on Connected Segments"
            )
            fig_bar_spill.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                xaxis=dict(title="Connected Sensor Station", tickfont=dict(color="#CBD5E1")),
                yaxis=dict(title="Estimated Speed Drop (mph)", gridcolor="#1E293B")
            )
            st.plotly_chart(fig_bar_spill, use_container_width=True)
    else:
        st.info("No adjacent highway segments within the active propagation threshold for this station.")

    # Structured Evidence & Assumptions Audit (Priority 4.E)
    st.divider()
    st.subheader("📋 Structured Evidence & Assumptions Audit")
    st.caption("Explicit separation of observed sensor telemetry, machine learning predictions, scenario assumptions, and known limitations.")
    
    audit = sim_output["evidence_audit"]
    a_col1, a_col2 = st.columns(2)
    
    with a_col1:
        with st.container(border=True):
            st.markdown("**📡 1. Observed Data**")
            st.write(f"• **Latest Observed Speed:** {audit['observed_data']['current_observed_speed']}")
            st.write(f"• **Target Station:** #{audit['observed_data']['station_id']} ({audit['observed_data']['corridor']})")
            st.write(f"• **Data Source:** {audit['observed_data']['source_telemetry']}")
            st.caption(f"• **Telemetry Status:** {audit['observed_data']['observation_type']}")
        
        with st.container(border=True):
            st.markdown("**🤖 2. Model Predictions**")
            st.write(f"• **Baseline ML Forecast:** {audit['model_predictions']['baseline_forecast_speed']}")
            st.write(f"• **Forecast Horizon:** {audit['model_predictions']['forecast_horizon']}")
            st.write(f"• **Predicted Condition:** {audit['model_predictions']['baseline_condition']}")
            st.write(f"• **Algorithm:** {audit['model_predictions']['model_type']}")
            st.write(f"• **Confidence Score:** {audit['model_predictions']['confidence_metric']}")

    with a_col2:
        with st.container(border=True):
            st.markdown("**⚙️ 3. Scenario Assumptions**")
            st.write(f"• **Capacity Throttling:** {audit['scenario_assumptions']['capacity_reduction']}")
            st.write(f"• **Assumed Duration:** {audit['scenario_assumptions']['assumed_duration']}")
            st.write(f"• **Hypothetical Weather:** {audit['scenario_assumptions']['hypothetical_weather']}")
            st.caption(f"• **Propagation Model:** {audit['scenario_assumptions']['propagation_model']}")
        
        with st.container(border=True):
            st.markdown("**⚠️ 4. Key Limitations**")
            st.write(f"• **Simulation Nature:** {audit['limitations']['nature_of_output']}")
            st.write(f"• **Spillover Scope:** {audit['limitations']['spillover_interpretation']}")
            st.caption(f"• **Historical Baseline:** {audit['limitations']['data_timeliness']}")

# -------------------------------------------------------------
# TAB 2: Dynamic Weight Sensitivity Explorer
# -------------------------------------------------------------
with tab_weights:
    st.subheader("⚖️ Dynamic Ranking Weight Sensitivity Explorer")
    st.write(
        "Use this tool to understand how changing operational priorities shifts the order of highway corridors needing attention. "
        "Each factor represents a distinct traffic risk indicator backed by actual sensor telemetry and model predictions."
    )
    
    # Plain English Factor Descriptions (Priority 4.B)
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        with st.container(border=True):
            st.markdown("**1. Congestion Severity**")
            st.caption("Speed deficit below 65 mph. Higher value prioritizes roads currently moving the slowest.")
    with f_col2:
        with st.container(border=True):
            st.markdown("**2. Deterioration**")
            st.caption("Projected 15-min speed drop. Higher value prioritizes corridors where queues are actively forming.")
    with f_col3:
        with st.container(border=True):
            st.markdown("**3. Historical Recurrence**")
            st.caption("Congestion frequency. Higher value prioritizes chronic recurring bottlenecks over one-time slowdowns.")
    with f_col4:
        with st.container(border=True):
            st.markdown("**4. Queue Persistence**")
            st.caption("Sustained low speed across 60 min. Higher value prioritizes entrenched long-standing queues.")
    
    col_wt1, col_wt2, col_wt3, col_wt4 = st.columns(4)
    with col_wt1:
        w_sev = st.slider("Severity Weight", 0.0, 1.0, DEFAULT_WEIGHTS["severity"], 0.05, help="Range: 0.0 to 1.0. Higher increases priority of currently slow roads.")
    with col_wt2:
        w_det = st.slider("Deterioration Weight", 0.0, 1.0, DEFAULT_WEIGHTS["deterioration"], 0.05, help="Range: 0.0 to 1.0. Higher increases priority of rapidly worsening corridors.")
    with col_wt3:
        w_rec = st.slider("Recurrence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["recurrence"], 0.05, help="Range: 0.0 to 1.0. Higher increases priority of chronic recurring bottlenecks.")
    with col_wt4:
        w_per = st.slider("Persistence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["persistence"], 0.05, help="Range: 0.0 to 1.0. Higher increases priority of sustained queues across past 60 min.")
        
    custom_weights = {
        "severity": w_sev,
        "deterioration": w_det,
        "recurrence": w_rec,
        "persistence": w_per
    }
    
    total_w = sum(custom_weights.values())
    if total_w == 0:
        st.warning("Total policy weight cannot be zero. Resetting to standard baseline weights.")
        custom_weights = DEFAULT_WEIGHTS.copy()
    else:
        st.info(f"Normalized Factor Allocation: Severity {w_sev/total_w*100:.0f}% • Deterioration {w_det/total_w*100:.0f}% • Recurrence {w_rec/total_w*100:.0f}% • Persistence {w_per/total_w*100:.0f}%")
        
    # Baseline ranking
    baseline_ranked = calculate_hotspot_priority_scores(
        predictions_df,
        historical_recurrence_map=rec_map,
        weights=DEFAULT_WEIGHTS
    )
    
    # Custom policy ranking recalculated from actual implementation
    custom_ranked = calculate_hotspot_priority_scores(
        predictions_df,
        historical_recurrence_map=rec_map,
        weights=custom_weights
    )
    
    # Before / After Comparison Table (Priority 4.B)
    st.subheader("Before / After Ranking Comparison (Top 15 Hotspots)")
    st.caption("Direct comparison of baseline rankings versus custom policy rankings to evaluate how priority queues shift.")
    
    # Merge baseline and custom rankings for direct comparison
    comp_df = custom_ranked.head(15)[[
        "node_id", "sensor_id", "corridor", "hotspot_rank", "priority_score", 
        "severity_tier", "current_speed", "pred_speed_15m", "ranking_reason"
    ]].copy()
    
    base_lookup = baseline_ranked.set_index("node_id")
    comp_df["baseline_rank"] = comp_df["node_id"].map(base_lookup["hotspot_rank"])
    comp_df["baseline_score"] = comp_df["node_id"].map(base_lookup["priority_score"])
    comp_df["rank_shift"] = comp_df["baseline_rank"] - comp_df["hotspot_rank"]
    
    def format_shift(val):
        if val > 0:
            return f"▲ +{int(val)} (Elevated)"
        elif val < 0:
            return f"▼ {int(val)} (Demoted)"
        return "▬ 0 (Unchanged)"
        
    comp_df["Rank Shift"] = comp_df["rank_shift"].apply(format_shift)
    
    display_comp = comp_df[[
        "hotspot_rank", "baseline_rank", "Rank Shift", "sensor_id", "corridor",
        "priority_score", "baseline_score", "severity_tier", "current_speed", "pred_speed_15m", "ranking_reason"
    ]].copy()
    
    display_comp.columns = [
        "Custom Rank", "Baseline Rank", "Rank Movement", "Station ID", "Freeway Corridor",
        "Custom Score", "Baseline Score", "Severity Tier", "Current Speed (mph)", "Forecast t+15m (mph)", "Main Ranking Factor"
    ]
    
    st.dataframe(display_comp, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: Multi-Horizon Speed Forecasts
# -------------------------------------------------------------
with tab_multi:
    st.subheader("⏱️ Multi-Horizon Speed Forecasts")
    st.write(
        "Compare estimated traffic speeds at supported future intervals. "
        "The production pipeline packages an independently trained 15-minute forecasting model "
        "(HistGradientBoostingRegressor) and classification model (HistGradientBoostingClassifier)."
    )
    
    # Methodology Distinction Card (Priority 4.B)
    st.info(
        "**Forecast Horizon Methodology Disclosure:**\n\n"
        "• **15-Minute Horizon (t+15m):** *Independently Trained Supervised ML Model* trained on 150,000 historical PeMS loop detector observations.\n\n"
        "• **30-Minute Horizon (t+30m):** *Kinematic Trend Projection (Experimental Heuristic Extrapolation)*. Not an independently trained regression model.",
        icon="ℹ️"
    )
    
    # Station Selector for Deep-Dive Multi-Horizon Profile
    st.subheader("Station Historical Telemetry & Prediction Timeline")
    st_options = {
        f"Station #{row['sensor_id']} ({row['corridor'].split(' (')[0]})": int(row["node_id"])
        for _, row in predictions_df.iterrows()
    }
    sel_st_profile = st.selectbox("Select Station for Temporal Horizon Profile", list(st_options.keys()), index=0)
    profile_node = st_options[sel_st_profile]
    prof_row = predictions_df[predictions_df["node_id"] == profile_node].iloc[0]
    
    # Compile 60-min lookback and 15-min forecast timeline
    time_points = ["t-60m", "t-45m", "t-30m", "t-15m", "t0 (Current)", "t+15m (ML Forecast)"]
    speeds_profile = [
        float(prof_row.get("x_t-11_d0", prof_row["current_speed"])),
        float(prof_row.get("x_t-8_d0", prof_row["current_speed"])),
        float(prof_row.get("x_t-5_d0", prof_row["current_speed"])),
        float(prof_row.get("x_t-2_d0", prof_row["current_speed"])),
        float(prof_row["current_speed"]),
        float(prof_row["pred_speed_15m"])
    ]
    
    fig_time = go.Figure()
    # Observed line
    fig_time.add_trace(go.Scatter(
        x=time_points[:5],
        y=speeds_profile[:5],
        mode="lines+markers",
        name="Observed Telemetry",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=8, color="#38BDF8")
    ))
    # Forecast line segment
    fig_time.add_trace(go.Scatter(
        x=time_points[4:],
        y=speeds_profile[4:],
        mode="lines+markers",
        name="ML Model Forecast (t+15m)",
        line=dict(color="#10B981", width=3, dash="dash"),
        marker=dict(size=10, color="#10B981", symbol="diamond")
    ))
    # Reference 65 mph line
    fig_time.add_hline(y=65.0, line_dash="dot", line_color="#64748B", annotation_text="65 mph Free-Flow Design Speed")
    fig_time.add_hline(y=35.0, line_dash="dot", line_color="#EF4444", annotation_text="35 mph Severe Congestion Threshold")
    
    fig_time.update_layout(
        title=f"Telemetry Speed Profile & 15-Min Forecast — Station #{prof_row['sensor_id']} ({prof_row['corridor']})",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Timeline (5-min intervals)", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1")),
        yaxis=dict(title="Operating Speed (mph)", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1"), range=[0, 75]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(color="#F1F5F9"))
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Network-Wide Multi-Horizon Telemetry Table
    st.subheader("Network Highway Forecast Inventory")
    
    multi_df = predictions_df[[
        "node_id", "sensor_id", "corridor", "current_speed", 
        "pred_speed_15m", "speed_change_15m", "pred_condition", "confidence_score"
    ]].copy()
    
    multi_df["est_speed_30m_extrap"] = (multi_df["pred_speed_15m"] + (multi_df["speed_change_15m"] * 0.5)).clip(0.0, 70.0).round(1)
    multi_df["forecast_type"] = "Supervised ML (HistGradientBoosting)"
    
    multi_df.columns = [
        "Node", "Station ID", "Freeway Corridor", "Current Observed Speed (mph)", 
        "Forecast Speed t+15m (mph)", "Projected Speed Delta (mph)", 
        "Predicted Category (t+15m)", "Model Confidence (%)", 
        "Experimental Trend t+30m (mph)", "Horizon Methodology"
    ]
    st.dataframe(multi_df.head(25), use_container_width=True, hide_index=True)
