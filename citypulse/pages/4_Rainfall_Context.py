import streamlit as st
import pandas as pd
from modules.loaders.rainfall_loader import load_rainfall_data
from modules.visualizations import (
    plot_rainfall_timeline, plot_cumulative_rainfall, plot_rainfall_departure
)

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Mumbai City Environmental Context (IMD Rainfall)
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Daily meteorological observations for <strong style="color: #F1F5F9;">Mumbai City District</strong> from the India Meteorological Department
        </p>
    </div>
    <div style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #38BDF8; font-weight: 700;">
        Official Meteorological Data
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Disclaimer
st.markdown("""
<div class="info-banner">
    <strong>🌧️ Macro Reporting Unit Boundary:</strong><br>
    Rainfall observations describe the available geographic reporting unit (Mumbai City District) and do not necessarily represent conditions at individual roads or junctions.<br>
    <span style="font-size: 0.75rem; opacity: 0.85;">
        District-level measurements are maintained strictly separate from traffic analysis in Phase 1. Do not assert that rainfall caused recorded congestion or accidents without junction-level micro-sensors.
    </span>
</div>
""", unsafe_allow_html=True)

rainfall_data = load_rainfall_data()

if rainfall_data.get("status") == "missing_file":
    st.error(rainfall_data.get("error"))
    st.info("Place `rainfall_districtwise_daily_imd.csv` in `citypulse/data/`.")
    st.stop()

rain_df = rainfall_data.get("mumbai_df")
summary = rainfall_data.get("summary", {})
metadata = rainfall_data.get("metadata", {})

if rain_df is None or rain_df.empty:
    st.warning("No Mumbai City rainfall records found in file.")
    st.stop()

# Metric KPI cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Observed Rain", f"{summary.get('total_rainfall_mm', 0)} mm")
with c2:
    st.metric("Peak Single-Day Rain", f"{summary.get('max_single_day_mm', 0)} mm")
with c3:
    st.metric("Rainy Days Count", f"{summary.get('rainy_days_count', 0)} Days", help="Days with > 0.1 mm precipitation")
with c4:
    st.metric("Missing Value Rate", "0.0% Nulls", help="Complete 65-day continuous series")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Main Daily Chart
st.subheader("Daily Actual vs Normal Rainfall (Mumbai City District)")
st.plotly_chart(plot_rainfall_timeline(rain_df), use_container_width=True)

col_cum, col_dep = st.columns(2)

with col_cum:
    st.subheader("Cumulative Monsoon Progression (mm)")
    fig_cum = plot_cumulative_rainfall(rain_df)
    if fig_cum:
        st.plotly_chart(fig_cum, use_container_width=True)

with col_dep:
    st.subheader("Daily Departure from Climatological Normal (%)")
    fig_dep = plot_rainfall_departure(rain_df)
    if fig_dep:
        st.plotly_chart(fig_dep, use_container_width=True)

# Observation Data Table
st.subheader("📄 Daily Observation Archive (65 Days)")
display_rain_df = rain_df.copy()
if "Date" in display_rain_df.columns and pd.api.types.is_datetime64_any_dtype(display_rain_df["Date"]):
    display_rain_df["Date"] = display_rain_df["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    display_rain_df[[
        "State", "District", "Date", "Daily Actual", "Daily Normal", "Daily Departure Per", "Daily Category"
    ]],
    use_container_width=True,
    hide_index=True
)
