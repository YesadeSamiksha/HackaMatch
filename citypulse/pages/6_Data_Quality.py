"""
6_Data_Quality.py - Comprehensive Dataset Audit & Health Hub
Presents the data quality audit, missingness metrics, schema inventory, and geographic provenance for train.parquet.
"""

import streamlit as st
import pandas as pd
import os

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #FFFFFF;">
        Data Quality & Geographic Audit
    </h1>
    <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
        Full technical audit of primary training asset (<code>train.parquet</code>), schema completeness, and sensor network integrity.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Summary Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Observations", "4,962,618", "train.parquet")
c2.metric("Missing Elements", "0 (0.0%)", "100% Complete")
c3.metric("Monitored Sensors", "207 Stations", "Los Angeles County Highway Network")
c4.metric("Temporal Span", "83.2 Days", "Mar 1 - May 23, 2012")

st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

# 2. Tabs for Reports
tab_schema, tab_profile, tab_quality, tab_feasibility = st.tabs([
    "📋 Dataset Schema (50 Columns)",
    "📊 Dataset Profile",
    "🛡️ Quality & Leakage Audit",
    "🎯 Target & Feature Feasibility"
])

reports_dir = "reports"

with tab_schema:
    schema_path = os.path.join(reports_dir, "dataset_schema.csv")
    if os.path.exists(schema_path):
        df_schema = pd.read_csv(schema_path)
        st.dataframe(df_schema, use_container_width=True, hide_index=True)
    else:
        st.warning("Schema report not found. Run `python -m src.dataset_audit` to generate.")

with tab_profile:
    profile_path = os.path.join(reports_dir, "dataset_profile.md")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Dataset profile report not found.")

with tab_quality:
    quality_path = os.path.join(reports_dir, "data_quality_report.md")
    if os.path.exists(quality_path):
        with open(quality_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Data quality report not found.")

with tab_feasibility:
    target_path = os.path.join(reports_dir, "target_feasibility_report.md")
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    st.markdown("<hr style='border-color: #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)
    feature_path = os.path.join(reports_dir, "feature_feasibility_report.md")
    if os.path.exists(feature_path):
        with open(feature_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
