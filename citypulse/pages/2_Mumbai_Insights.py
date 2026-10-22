import streamlit as st
import pandas as pd
from modules.loaders.challan_loader import load_challan_data
from modules.loaders.traffic_loader import load_traffic_csv
from modules.loaders.rainfall_loader import load_rainfall_data
from modules.insights import compute_evidence_insights
from modules.visualizations import (
    plot_top_divisions_bar, plot_violation_categories_bar, plot_vehicle_enforcement_bar
)

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Mumbai Traffic Enforcement Insights (2024)
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Official empirical records from the <strong style="color: #F1F5F9;">Mumbai Traffic Police Annual Report</strong> • Reporting Period: <span style="font-family: monospace; color: #06B6D4;">1 January 2024 to 31 December 2024</span>
        </p>
    </div>
    <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #22D3EE; font-weight: 700;">
        Primary Mumbai Dataset
    </div>
</div>
""", unsafe_allow_html=True)

# Administrative Separation Banner
st.markdown("""
<div class="info-banner">
    <strong>🛡️ Administrative Separation Guardrail:</strong><br>
    Traffic Division records (42 divisions, 5,937,151 challans) and local Police Station records (91 stations, 847,766 challans) 
    represent distinct administrative reporting lines and must NOT be added together. 
    High challan counts reflect camera placement and officer deployment, not proof of physical congestion or accident risk.
</div>
""", unsafe_allow_html=True)

# Load data
challan_data = load_challan_data()
traffic_data = load_traffic_csv()
rainfall_data = load_rainfall_data()

if challan_data.get("status") == "missing_file":
    st.error(challan_data.get("error"))
    st.stop()

# Descriptive Evidence Insights Section
insights = compute_evidence_insights(challan_data, traffic_data, rainfall_data)

st.subheader("💡 Evidence-Based Descriptive Insights (No ML Required)")
st.caption("Empirical findings calculated directly from validated government records without black-box generative models.")

cols = st.columns(3)
for idx, item in enumerate(insights[:6]):
    col = cols[idx % 3]
    with col:
        with st.container():
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 14px; margin-bottom: 14px; min-height: 290px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.65rem; font-weight: 700; color: #06B6D4; background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); padding: 2px 6px; border-radius: 4px; text-transform: uppercase;">{item['category']}</span>
                    </div>
                    <h4 style="margin: 0 0 8px 0; font-size: 0.85rem; font-weight: 700; color: #FFFFFF;">{item['title']}</h4>
                    <div style="background: #0B0F19; border: 1px solid #1E293B; border-radius: 6px; padding: 8px; font-size: 0.75rem; color: #CBD5E1; margin-bottom: 8px;">
                        <strong style="color: #06B6D4;">Observed:</strong> {item['observed']}
                    </div>
                    <p style="font-size: 0.72rem; color: #94A3B8; margin: 0 0 8px 0; line-height: 1.4;">
                        <strong style="color: #F59E0B;">Interpretation:</strong> {item['interpretation']}
                    </p>
                </div>
                <div style="border-top: 1px solid #1E293B; padding-top: 6px; font-size: 0.68rem;">
                    <div style="color: #10B981;"><strong>✓ Can infer:</strong> {item['can_infer'][0]}</div>
                    <div style="color: #EF4444; margin-top: 2px;"><strong>✗ Cannot infer:</strong> {item['cannot_infer'][0]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Tabbed tables & charts
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Traffic Divisions (42)",
    "Police Stations (91)",
    "Violation Categories (24)",
    "Vehicle Types (26)",
    "Drunk & Drive Report"
])

# TAB 1: DIVISIONS
with tab1:
    div_df = challan_data.get("division_df")
    div_gt = challan_data.get("division_grand_total", {})
    if div_df is not None and not div_df.empty:
        st.subheader("Traffic Division Wise Recorded Challans")
        st.plotly_chart(plot_top_divisions_bar(div_df, n=12), use_container_width=True)

        col_search, col_sort = st.columns([3, 1])
        with col_search:
            search_div = st.text_input("🔍 Search Traffic Division:", placeholder="e.g. Bandra, Mankhurd, Worli...")
        with col_sort:
            sort_div_col = st.selectbox("Sort by:", ["Total Challan", "Total Amount", "Paid Challan", "Unpaid Challan"], index=0)

        sort_map = {
            "Total Challan": "total_challan",
            "Total Amount": "total_amount",
            "Paid Challan": "paid_challan",
            "Unpaid Challan": "unpaid_challan"
        }

        display_div_df = div_df.copy()
        if search_div:
            display_div_df = display_div_df[display_div_df["division"].str.contains(search_div, case=False, na=False)]

        display_div_df = display_div_df.sort_values(by=sort_map[sort_div_col], ascending=False)
        display_div_df["Unpaid Rate %"] = (display_div_df["unpaid_challan"] / display_div_df["total_challan"] * 100).round(1).astype(str) + "%"

        st.dataframe(
            display_div_df[[
                "sr_no", "division", "total_challan", "total_amount", "paid_challan", "paid_amount", "unpaid_challan", "Unpaid Rate %"
            ]].rename(columns={
                "sr_no": "#",
                "division": "Traffic Division",
                "total_challan": "Total Challans",
                "total_amount": "Total Amount (₹)",
                "paid_challan": "Paid Challans",
                "paid_amount": "Paid Amount (₹)",
                "unpaid_challan": "Unpaid Challans"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Grand Total: {div_gt.get('total_challan', 5937151):,} challans across all 42 divisions (₹{div_gt.get('total_amount', 4784033250):,} total amount).")

# TAB 2: POLICE STATIONS
with tab2:
    ps_df = challan_data.get("police_station_df")
    ps_gt = challan_data.get("police_station_grand_total", {})
    if ps_df is not None and not ps_df.empty:
        st.subheader("Mumbai Police Station Wise Recorded Challans")
        st.caption("Challans recorded by 91 local Law & Order police stations (separate administrative reporting).")

        search_ps = st.text_input("🔍 Search Police Station:", placeholder="e.g. Khar, Malwani, Andheri...")
        display_ps_df = ps_df.copy()
        if search_ps:
            display_ps_df = display_ps_df[display_ps_df["police_station"].str.contains(search_ps, case=False, na=False)]

        display_ps_df["Unpaid Rate %"] = (display_ps_df["unpaid_challan"] / display_ps_df["total_challan"] * 100).round(1).astype(str) + "%"

        st.dataframe(
            display_ps_df[[
                "sr_no", "police_station", "total_challan", "total_amount", "paid_challan", "unpaid_challan", "Unpaid Rate %"
            ]].rename(columns={
                "sr_no": "#",
                "police_station": "Police Station",
                "total_challan": "Total Challans",
                "total_amount": "Total Amount (₹)",
                "paid_challan": "Paid Challans",
                "unpaid_challan": "Unpaid Challans"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Grand Total: {ps_gt.get('total_challan', 847766):,} challans across all 91 police stations (₹{ps_gt.get('total_amount', 670772400):,} total amount).")

# TAB 3: CATEGORIES
with tab3:
    cat_df = challan_data.get("categories_df")
    cat_gt = challan_data.get("categories_grand_total", 5937151)
    if cat_df is not None and not cat_df.empty:
        st.subheader("24 Violation Categories Recorded in 2024")
        st.plotly_chart(plot_violation_categories_bar(cat_df, n=15), use_container_width=True)

        display_cat = cat_df.copy()
        display_cat["Share %"] = (display_cat["total_challans"] / cat_gt * 100).round(2).astype(str) + "%"
        display_cat["Rank"] = range(1, len(display_cat) + 1)

        st.dataframe(
            display_cat[["Rank", "category", "total_challans", "Share %"]].rename(columns={
                "category": "Violation Category",
                "total_challans": "Total Challans"
            }),
            use_container_width=True,
            hide_index=True
        )

# TAB 4: VEHICLE TYPES
with tab4:
    veh_df = challan_data.get("vehicle_types_df")
    veh_gt = challan_data.get("vehicle_grand_total", {})
    if veh_df is not None and not veh_df.empty:
        st.subheader("Official Mumbai 2024 Enforcement by Vehicle Type")
        st.caption("Extracted from Table 5 (Vehicle Types Wise Challan Report) of Mumbai Traffic Police report.")
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
        st.caption(f"Grand Total: {veh_gt.get('total_challan', 6784925):,} vehicle challans (₹{veh_gt.get('total_amount', 5454813650):,} total fine amount).")

# TAB 5: DRUNK & DRIVE
with tab5:
    dd_div_df = challan_data.get("drunk_drive_divisions_df")
    dd_ps_df = challan_data.get("drunk_drive_police_stations_df")
    st.subheader("Specialized Drunk & Drive Nakabandi Enforcement")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Division-Wise Breakdown** (Total: {challan_data.get('drunk_drive_divisions_total', 9480):,})")
        if dd_div_df is not None and not dd_div_df.empty:
            st.dataframe(
                dd_div_df.rename(columns={"sr_no": "#", "division": "Division", "total_challans": "Challans"}),
                use_container_width=True,
                hide_index=True
            )
    with c2:
        st.markdown(f"**Police Station Breakdown** (Total: {challan_data.get('drunk_drive_police_stations_total', 48)})")
        if dd_ps_df is not None and not dd_ps_df.empty:
            st.dataframe(
                dd_ps_df.rename(columns={"sr_no": "#", "police_station": "Police Station", "total_challans": "Challans"}),
                use_container_width=True,
                hide_index=True
            )
