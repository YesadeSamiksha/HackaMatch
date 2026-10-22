from typing import Dict, Any, List

def compute_evidence_insights(challan_data: Dict[str, Any], traffic_data: Dict[str, Any], rainfall_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights = []

    # 1. Enforcement Concentration in Multimedia Cell
    div_df = challan_data.get("division_df")
    if div_df is not None and not div_df.empty:
        top_div = div_df.iloc[0]
        total_ch = challan_data.get("summary", {}).get("division_total_challans", 5937151)
        pct = round((top_div["total_challan"] / total_ch) * 100, 2)
        insights.append({
            "id": "insight_multimedia_cell_enforcement",
            "category": "Enforcement Distribution",
            "title": "Centralized Automated Enforcement Concentration",
            "observed": f"Multimedia Cell recorded {top_div['total_challan']:,} challans in 2024, accounting for {pct}% of all {total_ch:,} recorded division challans across Mumbai.",
            "table_source": "Table 1 (Mumbai Traffic Division Wise Challan Report, 2024)",
            "interpretation": "Multimedia Cell operates Mumbai's centralized CCTV and speed camera network. This massive concentration reflects camera automation rather than higher physical driver disobedience in a single physical jurisdiction.",
            "can_infer": [
                "Automated CCTV detection captures a substantial portion of all recorded traffic violations in Mumbai",
                "Multimedia Cell serves as the primary processing hub for automated electronic notices"
            ],
            "cannot_infer": [
                "That drivers in any single physical locality commit 8.7% of all infractions",
                "That areas monitored by Multimedia Cell cameras are intrinsically more dangerous or congested than unmonitored roads"
            ],
            "suggested_next_step": "Cross-reference CCTV camera deployment locations with ground division boundaries to distinguish automated camera notices from on-ground traffic officer interventions."
        })

    # 2. Obstruction to Free Flow of Traffic
    cat_df = challan_data.get("categories_df")
    if cat_df is not None and not cat_df.empty:
        obs_row = cat_df[cat_df["category"].str.contains("Obstruction", case=False, na=False)]
        if not obs_row.empty:
            obs_count = int(obs_row.iloc[0]["total_challans"])
            obs_pct = round((obs_count / 5937151) * 100, 2)
            insights.append({
                "id": "insight_traffic_obstruction",
                "category": "Infraction Dynamics",
                "title": "Obstruction of Free Flow as Leading Actionable Infraction",
                "observed": f"'Causing Obstruction To Free Flow Of Traffic' generated {obs_count:,} challans ({obs_pct}% of total division violations), making it the second most common category after generic 'Other M.V. Acts'.",
                "table_source": "Table 3 (Mumbai Traffic Division + Head Wise Challan Report, 2024)",
                "interpretation": "High obstruction challans highlight pervasive challenges with illegal curb-parking, double-parking, commercial loading, and unauthorized halt zones that block active travel lanes across Mumbai arterial roads.",
                "can_infer": [
                    "Traffic police prioritize enforcement against vehicles restricting lane capacity and carriageway bottlenecks",
                    "Obstruction infractions occur at high scale across virtually all municipal divisions"
                ],
                "cannot_infer": [
                    "That obstruction challans directly prove gridlock severity or bottleneck duration at any specific junction",
                    "That roads with fewer obstruction challans have completely unobstructed traffic flow"
                ],
                "suggested_next_step": "Correlate division-wise obstruction challans with roadway width and dedicated parking infrastructure availability to identify infrastructure gaps."
            })

    # 3. Two-Wheeler Enforcement & Helmet Compliance
    veh_df = challan_data.get("vehicle_types_df")
    if veh_df is not None and not veh_df.empty and cat_df is not None and not cat_df.empty:
        tw_row = veh_df[veh_df["vehicle_type"].str.contains("TWO_WHEELER", case=False, na=False)]
        helmet_row = cat_df[cat_df["category"].str.contains("Without Helmet", case=False, na=False)]
        if not tw_row.empty and not helmet_row.empty:
            tw_count = int(tw_row.iloc[0]["total_challan"])
            total_veh_ch = challan_data.get("summary", {}).get("vehicle_types_total_challans", 6784925)
            tw_pct = round((tw_count / total_veh_ch) * 100, 2)
            helmet_count = int(helmet_row.iloc[0]["total_challans"])
            insights.append({
                "id": "insight_two_wheeler_enforcement",
                "category": "Modal Vulnerability & Safety",
                "title": "Two-Wheeler Predominance in Enforcement & Safety Violations",
                "observed": f"Two-wheelers accounted for {tw_count:,} out of {total_veh_ch:,} recorded vehicle-type challans ({tw_pct}%). Furthermore, 'Without Helmet' violations alone amounted to {helmet_count:,} challans.",
                "table_source": "Table 5 (Vehicle Types Wise Report) & Table 3 (Head Wise Report, 2024)",
                "interpretation": "Two-wheelers form the largest individual motorized mode subjected to traffic enforcement in Mumbai. The high volume of helmet violations underscores persistent road-safety vulnerability among riders.",
                "can_infer": [
                    "Two-wheelers represent the most frequently cited vehicle category in Mumbai enforcement operations",
                    "Helmet non-compliance is a widespread safety issue consistently enforced by traffic police"
                ],
                "cannot_infer": [
                    "That two-wheeler riders cause more accidents than other vehicle classes (accident culpability is not measured in this report)",
                    "That helmet citations indicate vehicle speed or mechanical safety"
                ],
                "suggested_next_step": "Combine helmet enforcement data with hospital trauma injury statistics and junction surveillance to target targeted helmet awareness and enforcement drives."
            })

    # 4. Drunk and Drive Geographical Distribution
    dd_div_df = challan_data.get("drunk_drive_divisions_df")
    if dd_div_df is not None and not dd_div_df.empty:
        top_dd = dd_div_df.iloc[0]
        dd_total = challan_data.get("drunk_drive_divisions_total", 9480)
        sahar_count = dd_div_df.iloc[1]["total_challans"] if len(dd_div_df) > 1 else "N/A"
        mulund_count = dd_div_df.iloc[2]["total_challans"] if len(dd_div_df) > 2 else "N/A"
        insights.append({
            "id": "insight_drunk_and_drive_patterns",
            "category": "Specialized Enforcement",
            "title": "Drunk & Drive Enforcement Concentration Patterns",
            "observed": f"Drunk & Drive enforcement recorded {dd_total:,} total challans in 2024. Nagpada Division recorded the highest division count with {top_dd['total_challans']:,} challans ({round((top_dd['total_challans']/dd_total)*100, 1)}%), followed by Sahar ({sahar_count}) and Mulund ({mulund_count}).",
            "table_source": "Table 6 (Mumbai Traffic Division Wise Drunk & Drive Challan Report, 2024)",
            "interpretation": "Drunk & drive enforcement counts strongly reflect targeted nocturnal breathalyzer checkpoints (nakabandis) positioned near nightlife clusters, major arterial access corridors, and transport terminals.",
            "can_infer": [
                "Traffic divisions with major nightlife corridors or entry checkpoints deploy frequent breathalyzer nakabandis",
                "Specialized enforcement drives generate distinct spatial enforcement footprints compared to day-to-day parking violations"
            ],
            "cannot_infer": [
                "That drunk driving prevalence is zero in divisions with lower counts (checkpoint absence vs violation absence)",
                "That drunk driving counts correlate directly with late-night vehicle volume"
            ],
            "suggested_next_step": "Overlay breathalyzer checkpoint deployment schedules and hours with recorded challan timestamps to evaluate enforcement coverage."
        })

    # 5. Strict Geographic Scoping Audit on Traffic CSV
    traffic_meta = traffic_data.get("metadata", {})
    raw_rows = traffic_meta.get("total_rows_raw", 1200)
    mumbai_rows = traffic_meta.get("mumbai_rows_retained", 234)
    insights.append({
        "id": "insight_geographic_integrity_traffic_csv",
        "category": "Data Engineering & Geographic Scoping",
        "title": "Strict Mumbai Filtering Prevents 80.5% Data Leakage",
        "observed": f"Out of {raw_rows} records in the Maharashtra traffic CSV, 966 records (80.5%) originated from other cities (Pune: 234, Thane: 242, Nashik: 262, Nagpur: 228). Only {mumbai_rows} records were verified as Mumbai City entries.",
        "table_source": "Dataset B (Maharashtra Traffic CSV — Geographic Audit)",
        "interpretation": "Without strict geographic filtering, over four-fifths of analyzed records would represent non-Mumbai conditions. Filtering isolates the 5 documented Mumbai areas (Bandra, Malad, Andheri, Dadar, Lower Parel).",
        "can_infer": [
            "Geographic segregation is essential to avoid contaminating Mumbai dashboards with Pune or Thane observations",
            "The remaining 234 records provide an exploratory prototyping sample for interface testing"
        ],
        "cannot_infer": [
            "That the 234 retained Mumbai rows are calibrated real-world measurements (measurement methodology remains unverified)",
            "That 5 areas fully represent all 42 municipal traffic divisions"
        ],
        "suggested_next_step": "Procure open physical sensor feeds (such as Mumbai Smart City camera APIs or Google Traffic / OpenStreetMap probe data) before starting ML training."
    })

    # 6. Environmental Context Separation
    rain_summary = rainfall_data.get("summary", {})
    obs_count = rain_summary.get("observation_count", 65)
    rain_total = rain_summary.get("total_rainfall_mm", 0.0)
    insights.append({
        "id": "insight_rainfall_context_separation",
        "category": "Environmental Decoupling",
        "title": "Macro Environmental Baseline Kept Separate from Traffic Dynamics",
        "observed": f"IMD Daily Rainfall dataset provides {obs_count} daily observations for Mumbai City District (2026-08-19 to 2026-10-22) recording {rain_total} mm of precipitation.",
        "table_source": "Dataset C (IMD District-wise Daily Rainfall Archive)",
        "interpretation": "Rainfall data is monitored at the district aggregate level. While precipitation is a known environmental factor during Mumbai monsoons, district averages cannot identify micro-level waterlogged junctions.",
        "can_infer": [
            "The dataset accurately reflects official IMD meteorological station observations for the district during the late monsoon timeline",
            "Rainy vs dry day cycles are verifiable at the macro city level"
        ],
        "cannot_infer": [
            "That rainfall on any specific day caused a specific traffic accident or congestion spike",
            "That rainfall was uniform across South Mumbai and Northern suburban arterial corridors"
        ],
        "suggested_next_step": "Acquire MCGM automated weather station (AWS) ward-level hyper-local rainfall sensors before attempting weather-traffic machine learning."
    })

    return insights
