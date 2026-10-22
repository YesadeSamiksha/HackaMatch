import streamlit as st
import pandas as pd
from modules.loaders.registration_loader import load_registration_csv
from modules.loaders.challan_loader import load_challan_data
from modules.visualizations import plot_vehicle_enforcement_bar

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Vehicle Trends & Fleet Classification
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Evaluating administrative vehicle registration statistics alongside official <strong style="color: #F1F5F9;">Mumbai 2024 Police Enforcement records</strong>
        </p>
    </div>
    <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #EF4444; font-weight: 700;">
        Geographic Audit Focus
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Conceptual Distinction Banner
st.markdown("""
<div class="danger-banner">
    <strong>🛑 Critical Data Science Distinction:</strong><br>
    Vehicle registrations are NOT the same as real-time road traffic counts.<br>
    <span style="font-size: 0.75rem; opacity: 0.9;">
        The vehicle registration dataset (babfb7a0-992e-487b-97dd-220d0290122e.csv) does not identify Mumbai City municipal coverage. 
        Because it represents state-wide or unspecified national administrative totals, it is classified as <strong>"Not suitable for Mumbai-only analysis"</strong> and excluded from Mumbai KPI metrics.
    </span>
</div>
""", unsafe_allow_html=True)

reg_data = load_registration_csv()
challan_data = load_challan_data()

# Conceptual explanation card
st.markdown("""
<div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; margin-bottom: 20px;">
    <h4 style="margin: 0 0 6px 0; font-size: 0.9rem; font-weight: 700; color: #A855F7;">
        Registration Stock vs Active On-Road Circulation
    </h4>
    <p style="margin: 0; font-size: 0.78rem; color: #CBD5E1; line-height: 1.5;">
        Vehicle registration tallies quantify cumulative ownership licenses issued over time. They do not describe how many vehicles travel on specific Mumbai corridors at 9:00 AM or 6:00 PM. 
        Real-world traffic operations require physical cameras and loop detectors, not static motor vehicle registers.
    </p>
</div>
""", unsafe_allow_html=True)

# Official Mumbai Police 2024 Vehicle-Type Enforcement Data
st.subheader("🚨 Verified Mumbai 2024 Enforcement by Vehicle Class (Official Report)")
st.caption("Extracted from Table 5 (Vehicle Types Wise Challan Report) of Mumbai Traffic Police 2024 report.")

veh_df = challan_data.get("vehicle_types_df")
veh_gt = challan_data.get("vehicle_grand_total", {})

if veh_df is not None and not veh_df.empty:
    st.plotly_chart(plot_vehicle_enforcement_bar(veh_df, n=12), use_container_width=True)

    st.dataframe(
        veh_df.rename(columns={
            "sr_no": "#",
            "vehicle_type": "Vehicle Type",
            "total_challan": "Total Challans",
            "total_amount": "Total Amount (₹)",
            "paid_challan": "Paid Challans",
            "unpaid_challan": "Unpaid Challans"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Grand Total: {veh_gt.get('total_challan', 6784925):,} vehicle challans across all 26 recognized classes (₹{veh_gt.get('total_amount', 5454813650):,} total fine amount).")

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Unverified Registration Dataset Audit Table
st.subheader("📋 Unverified Registration CSV Reference Table (Non-Mumbai Scoped)")
st.caption("Raw tabular records from `babfb7a0-992e-487b-97dd-220d0290122e.csv`. Excluded from Mumbai-specific metrics.")

reg_df = reg_data.get("df")
if reg_df is not None and not reg_df.empty:
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
else:
    st.info("Registration CSV not found or empty.")
