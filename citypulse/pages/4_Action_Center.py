"""
4_Action_Center.py - Screen 4: Operational Action Center & Intervention Planner
Decision-support command center that prioritizes highway bottlenecks by operational urgency
and provides evidence-grounded action recommendations for traffic management centers.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import get_network_snapshot
from src.inference import predict_network_state
from src.hotspot_ranking import calculate_hotspot_priority_scores
from src.traffic_patterns import analyze_traffic_patterns
from src.intervention_planner import generate_intervention_plan, ACTION_PLAYBOOK

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
        Operational Action Center & Intervention Planner
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #94A3B8;">
        Decision-support intelligence answering key operational questions and recommending targeted interventions.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Load Data & Compute Dynamic Hotspot Rankings (Cached)
@st.cache_data(ttl=600)
def load_action_data():
    snapshot = get_network_snapshot()
    preds = predict_network_state(snapshot)
    patterns = analyze_traffic_patterns(sample_size=100000)
    ranked = calculate_hotspot_priority_scores(preds, historical_recurrence_map=patterns["recurrence_map"])
    return ranked

try:
    with st.spinner("Compiling operational intervention schedule..."):
        ranked_df = load_action_data()
except Exception as e:
    st.error(f"Error compiling action plan: {e}")
    st.stop()

# 2. Key Action Metrics
crit_count = len(ranked_df[ranked_df["severity_tier"] == "Critical"])
high_count = len(ranked_df[ranked_df["severity_tier"] == "High"])
med_count = len(ranked_df[ranked_df["severity_tier"] == "Medium"])
top_bottleneck = ranked_df.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Critical Action Tier", f"{crit_count}", "Immediate Intervention", delta_color="inverse")
c2.metric("High Priority Tier", f"{high_count}", "Queue Buildup Advisory", delta_color="inverse")
c3.metric("Medium Watchlist Tier", f"{med_count}", "Downstream Monitoring")
c4.metric("Top Hotspot Station", f"#{top_bottleneck['sensor_id']}", f"{top_bottleneck['corridor'].split(' (')[0]}")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 3. Explainable Intervention Plan for Top Hotspot
st.markdown("### 📋 Explainable Intervention Plan (Detailed Diagnostic)")
st.caption("Synthesizes priority score, 15-minute forecast, and historical recurrence to answer the 6 core operational questions.")

# Selector for detailed inspection
top_candidates = ranked_df.head(15)
cand_opts = {
    f"Rank #{row['hotspot_rank']} — Station #{row['sensor_id']} ({row['corridor']}) | {row['severity_tier']} Priority": int(row["node_id"])
    for _, row in top_candidates.iterrows()
}
sel_cand_lbl = st.selectbox("Inspect Hotspot Intervention Plan", list(cand_opts.keys()), index=0)
sel_node_id = cand_opts[sel_cand_lbl]

node_row = ranked_df[ranked_df["node_id"] == sel_node_id].iloc[0]
plan = generate_intervention_plan(node_row)

# The 6 Core Operational Questions
q_col1, q_col2 = st.columns([6, 4])

with q_col1:
    # Build HTML without leading indentation so markdown never parses it as a code block
    diagnostic_card_html = (
        f'<div style="background: #111827; border: 1px solid #1E293B; border-radius: 10px; padding: 18px; margin-bottom: 14px;">'
        f'<div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">1. What is happening right now?</div>'
        f'<div style="color: #F1F5F9; font-size: 0.9rem; line-height: 1.5; margin-bottom: 14px;">{plan["what_is_happening"]}</div>'
        f'<div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">2. What is predicted to happen in 15 minutes?</div>'
        f'<div style="color: #F1F5F9; font-size: 0.9rem; line-height: 1.5; margin-bottom: 14px;">{plan["what_predicted"]}</div>'
        f'<div style="color: #38BDF8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">3. Why is this hotspot prioritized?</div>'
        f'<div style="color: #F1F5F9; font-size: 0.9rem; line-height: 1.5; margin-bottom: 14px;">{plan["why_ranked"]}</div>'
        f'<div style="color: #10B981; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">5. Recommended Investigation / Action:</div>'
        f'<div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 10px 14px; border-radius: 4px; color: #D1FAE5; font-size: 0.88rem; line-height: 1.5;">'
        f'<b>{plan["recommended_action_name"]}:</b> {plan["recommended_action_details"]["description"]}'
        f'</div>'
        f'</div>'
    )
    st.markdown(diagnostic_card_html, unsafe_allow_html=True)
    
    st.caption(f"ℹ️ *{plan['limitations']}*")

with q_col2:
    st.markdown("#### 4. Supporting Telemetry Metrics")
    for metric_name, metric_val in plan["supporting_metrics"].items():
        st.markdown(f"- **{metric_name}:** `{metric_val}`")
        
    st.markdown("#### 6. Action Playbook Trigger")
    st.markdown(f"- **Operational Trigger:** {plan['recommended_action_details']['trigger']}")
    st.markdown(f"- **Expected Benefit:** {plan['recommended_action_details']['effectiveness']}")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 4. Action Queue Table with Filters
st.markdown("### 🚦 Priority Intervention Queue")

f1, f2 = st.columns(2)
with f1:
    corrs = ["All Corridors"] + sorted(list(ranked_df["corridor"].dropna().unique()))
    sel_corr = st.selectbox("Corridor", corrs, key="action_corr")
with f2:
    tiers = ["All Tiers", "Critical", "High", "Medium", "Low"]
    sel_tier = st.selectbox("Severity Tier", tiers, key="action_tier")

filtered_action_df = ranked_df.copy()
if sel_corr != "All Corridors":
    filtered_action_df = filtered_action_df[filtered_action_df["corridor"] == sel_corr]
if sel_tier != "All Tiers":
    filtered_action_df = filtered_action_df[filtered_action_df["severity_tier"] == sel_tier]

display_table = filtered_action_df[[
    "hotspot_rank", "sensor_id", "corridor", "severity_tier", "priority_score",
    "current_speed", "pred_speed_15m", "ranking_reason", "recommended_investigation"
]].copy()

display_table.columns = [
    "Rank", "Station ID", "Freeway Corridor", "Severity Tier", "Priority Score",
    "Current Speed (mph)", "Forecast (t+15m)", "Main Rationale", "Recommended Investigation Action"
]

def style_act_tier(val):
    if val == "Critical":
        return "background-color: rgba(220, 38, 38, 0.25); color: #FCA5A5; font-weight: bold;"
    elif val == "High":
        return "background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5;"
    elif val == "Medium":
        return "background-color: rgba(245, 158, 11, 0.2); color: #FDE68A;"
    return "color: #A7F3D0;"

styled_view = display_table.head(50).style.applymap(style_act_tier, subset=["Severity Tier"])
st.dataframe(styled_view, use_container_width=True, hide_index=True)

# 5. Export Action Schedule CSV
csv_data = display_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export Complete Intervention Schedule (CSV)",
    data=csv_data,
    file_name="trafficpulse_intervention_schedule.csv",
    mime="text/csv",
    help="Download prioritized schedule for dispatch and traffic engineering records."
)
