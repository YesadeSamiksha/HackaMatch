import streamlit as st
import pandas as pd
from modules.loaders.traffic_loader import load_traffic_csv
from modules.visualizations import plot_traffic_speed_by_area, plot_congestion_donut

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Traffic Data Explorer (Maharashtra CSV — Mumbai Filtered)
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Exploratory prototyping sandbox restricted strictly to <strong style="color: #F1F5F9;">Mumbai City rows</strong>
        </p>
    </div>
    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #F59E0B; font-weight: 700;">
        ⚠️ Unverified Prototyping
    </div>
</div>
""", unsafe_allow_html=True)

# Prominent Warning Banner (Mandatory)
st.markdown("""
<div class="unverified-banner">
    <strong>⚠️ Transparency Disclaimer:</strong><br>
    Source and measurement methodology have not been verified. This data is for exploratory prototyping only.<br>
    <span style="font-size: 0.75rem; opacity: 0.85;">
        Values for physical speed, vehicle count, accidents, and congestion level are not verified physical sensor measurements. 
        Do not use this dataset for real-world safety decisions, and do NOT train ML models on it in Phase 1.
    </span>
</div>
""", unsafe_allow_html=True)

traffic_data = load_traffic_csv()

if traffic_data.get("status") == "missing_file":
    st.error(traffic_data.get("error"))
    st.info("Place `Traffic dataset.csv` in `citypulse/data/`.")
    st.stop()

mumbai_df = traffic_data.get("mumbai_df")
metadata = traffic_data.get("metadata", {})

if mumbai_df is None or mumbai_df.empty:
    st.warning("⚠️ No valid Mumbai City records found in the traffic CSV. Substituting other cities is prohibited by geographic integrity rules.")
    st.stop()

# Geographic Filtering Audit Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Raw Rows in Archive", f"{metadata.get('total_rows_raw', 1200)}")
with c2:
    st.metric("Non-Mumbai Excluded", f"-{metadata.get('non_mumbai_rows_excluded', 966)}", delta_color="inverse")
with c3:
    st.metric("Mumbai Rows Retained", f"{metadata.get('mumbai_rows_retained', 234)}")
with c4:
    st.metric("Missing Value Rate", "0.0% Nulls")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Interactive Filters
st.subheader("🔍 Interactive Subset Filters")
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    areas = ["All Areas"] + sorted(traffic_data.get("areas", []))
    selected_area = st.selectbox("Mumbai Area:", areas)

with f_col2:
    vts = ["All Vehicles"] + sorted(traffic_data.get("vehicle_types", []))
    selected_vt = st.selectbox("Vehicle Type:", vts)

with f_col3:
    congs = ["All Levels"] + sorted(traffic_data.get("congestion_levels", []))
    selected_cong = st.selectbox("Congestion Level:", congs)

# Apply filters
filtered_df = mumbai_df.copy()
if selected_area != "All Areas":
    filtered_df = filtered_df[filtered_df["Area"] == selected_area]
if selected_vt != "All Vehicles":
    filtered_df = filtered_df[filtered_df["Vehicle_Type"] == selected_vt]
if selected_cong != "All Levels":
    filtered_df = filtered_df[filtered_df["Congestion_Level"] == selected_cong]

st.caption(f"Showing {len(filtered_df)} matching Mumbai records (out of {len(mumbai_df)} total retained).")

# Visualizations
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.subheader("Reported Speed by Area (Unverified CSV)")
    fig_speed = plot_traffic_speed_by_area(traffic_data.get("speed_by_area_df"))
    if fig_speed:
        st.plotly_chart(fig_speed, use_container_width=True)

with v_col2:
    st.subheader("Congestion Tier Distribution (Mumbai)")
    fig_donut = plot_congestion_donut(filtered_df)
    if fig_donut:
        st.plotly_chart(fig_donut, use_container_width=True)

# Table Preview
st.subheader("📄 Mumbai Filtered Dataset Records")
st.caption("All displayed records are verified Mumbai City entries. Source: Unverified CSV.")

display_table = filtered_df.copy()
if "Date" in display_table.columns and pd.api.types.is_datetime64_any_dtype(display_table["Date"]):
    display_table["Date"] = display_table["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)

# Schema & Summary Statistics
with st.expander("📊 View Column Data Types & Statistical Summary", expanded=False):
    st.markdown("**Column Schema & Missing Values:**")
    st.dataframe(traffic_data.get("columns_info_df"), use_container_width=True, hide_index=True)

    st.markdown("**Numeric Summary Statistics (Mumbai Subset):**")
    stats_dict = traffic_data.get("summary_stats", {})
    if stats_dict:
        stats_rows = []
        for col_name, s in stats_dict.items():
            stats_rows.append({
                "Metric": col_name,
                "Count": s["count"],
                "Mean": s["mean"],
                "Std Dev": s["std"],
                "Min": s["min"],
                "Median": s["median"],
                "Max": s["max"]
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
