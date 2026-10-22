from pathlib import Path
from typing import Dict, Any, List

from config import DATA_DIR, CHALLAN_PDF_NAME, TRAFFIC_CSV_NAME, RAINFALL_CSV_NAME, REGISTRATION_CSV_NAME

def get_dataset_registry() -> List[Dict[str, Any]]:
    datasets = []

    # 1. Primary Mumbai Challans
    p1 = DATA_DIR / CHALLAN_PDF_NAME
    e1 = p1.exists()
    datasets.append({
        "id": "mumbai_traffic_challans_2024",
        "name": "Mumbai Traffic Violations Data 2024",
        "role": "Primary Mumbai-specific dataset",
        "file_name": CHALLAN_PDF_NAME,
        "file_path": str(p1),
        "file_format": "PDF (Vector text & tabular grids)",
        "file_size_kb": round(p1.stat().st_size / 1024, 1) if e1 else 0,
        "file_found": e1,
        "loading_status": "Loaded" if e1 else "Missing file",
        "status_badge_type": "success",
        "geographic_coverage": "Mumbai City, Maharashtra, India",
        "is_mumbai_specific": True,
        "reporting_period": "2024-01-01 to 2024-12-31",
        "source": "Mumbai Traffic Police Department (Official Violations Report)",
        "source_verification_status": "Verified Official Government Report",
        "data_quality_status": "High (Verified against printed Grand Totals)",
        "suitable_use": "Division-wise recorded challan comparisons, police-station comparisons, violation categories, enforcement ranking",
        "limitations": [
            "Represents recorded traffic challans and enforcement activity, not proof of congestion or definitive accident risk",
            "Division-level and police-station-level tables represent distinct reporting hierarchies and must not be combined",
            "High challan counts reflect camera placement and police deployment in addition to infractions"
        ],
        "record_counts": "42 Divisions, 91 Police Stations, 24 Categories, 26 Vehicle Types"
    })

    # 2. Secondary Maharashtra Traffic CSV
    p2 = DATA_DIR / TRAFFIC_CSV_NAME
    e2 = p2.exists()
    datasets.append({
        "id": "maharashtra_traffic_csv",
        "name": "Maharashtra Traffic Dataset CSV (Filtered to Mumbai City)",
        "role": "Exploratory prototyping dataset (Strictly filtered)",
        "file_name": TRAFFIC_CSV_NAME,
        "file_path": str(p2),
        "file_format": "CSV (Delimited text)",
        "file_size_kb": round(p2.stat().st_size / 1024, 1) if e2 else 0,
        "file_found": e2,
        "loading_status": "Loaded (Prototyping Only)" if e2 else "Missing file",
        "status_badge_type": "warning",
        "geographic_coverage": "Mumbai City only (234 Mumbai rows retained, 966 non-Mumbai rows excluded)",
        "is_mumbai_specific": True,
        "reporting_period": "2025-01-02 to 2025-12-31",
        "source": "Unverified — source and measurement methodology not confirmed",
        "source_verification_status": "Source and measurement methodology have not been verified",
        "data_quality_status": "Exploratory Prototyping Only (Unverified measurements)",
        "suitable_use": "Filter testing, UI/UX prototyping, schema readiness, future model architecture wiring",
        "limitations": [
            "Speed, vehicle count, and congestion values are NOT confirmed physical sensor measurements",
            "Do NOT use for real-world traffic management or safety decisions",
            "Do NOT train ML models on this dataset in Phase 1",
            "Never silently combine with official challan records"
        ],
        "record_counts": "1,200 raw rows -> 234 Mumbai rows (966 non-Mumbai rows eliminated)"
    })

    # 3. IMD Rainfall CSV
    p3 = DATA_DIR / RAINFALL_CSV_NAME
    e3 = p3.exists()
    datasets.append({
        "id": "imd_mumbai_rainfall",
        "name": "IMD District-wise Daily Rainfall CSV (Mumbai City)",
        "role": "Mumbai environmental context",
        "file_name": RAINFALL_CSV_NAME,
        "file_path": str(p3),
        "file_format": "CSV (Delimited text)",
        "file_size_kb": round(p3.stat().st_size / 1024, 1) if e3 else 0,
        "file_found": e3,
        "loading_status": "Loaded" if e3 else "Missing file",
        "status_badge_type": "success",
        "geographic_coverage": "Mumbai City District, Maharashtra, India (All other districts excluded)",
        "is_mumbai_specific": True,
        "reporting_period": "2026-08-19 to 2026-10-22",
        "source": "India Meteorological Department (IMD) District Monitoring",
        "source_verification_status": "Verified Official Meteorological Data",
        "data_quality_status": "Complete for observed monsoon timeline (65 daily records)",
        "suitable_use": "Macro environmental context, daily precipitation timeline, monsoon departure tracking",
        "limitations": [
            "Rainfall observations describe the district reporting unit and do not represent individual roads or junctions",
            "Kept completely separate from traffic analysis in Phase 1 (no causal claims)",
            "Covers late monsoon transition dates present in archive"
        ],
        "record_counts": "65 Daily Observations (filtered from 32,800 All-India rows)"
    })

    # 4. Vehicle Registration Summary
    p4 = DATA_DIR / REGISTRATION_CSV_NAME
    e4 = p4.exists()
    datasets.append({
        "id": "vehicle_registrations_summary",
        "name": "Vehicle Category Registration Summary CSV",
        "role": "Macro vehicle fleet context (Excluded from Mumbai metrics)",
        "file_name": REGISTRATION_CSV_NAME,
        "file_path": str(p4),
        "file_format": "CSV (Delimited text)",
        "file_size_kb": round(p4.stat().st_size / 1024, 1) if e4 else 0,
        "file_found": e4,
        "loading_status": "Not suitable for Mumbai-only analysis",
        "status_badge_type": "danger",
        "geographic_coverage": "State-wide / Unspecified National Scope (No Mumbai City Identifier)",
        "is_mumbai_specific": False,
        "reporting_period": "2022-01-01 to 2023-01-01",
        "source": "Administrative vehicle registration tables (scope unspecified in file)",
        "source_verification_status": "Unverified administrative scope",
        "data_quality_status": "Not suitable for Mumbai-only analysis",
        "suitable_use": "Macro vehicular category distribution reference only",
        "limitations": [
            "Does NOT identify Mumbai City specifically (only general category tallies)",
            "Excluded from Mumbai-specific charts and KPI cards",
            "Vehicle registrations are NOT the same as real-time road traffic counts"
        ],
        "record_counts": "10 vehicle categories (Zero verified Mumbai records)"
    })

    # 5. METR-LA / PEMS-BAY Benchmark Files
    p5 = DATA_DIR / "PEMS-BAY.h5"
    e5 = p5.exists()
    datasets.append({
        "id": "metr_la_pems_benchmark",
        "name": "METR-LA / PEMS-BAY Benchmark Sensor Files",
        "role": "Excluded non-Mumbai benchmark files",
        "file_name": "PEMS-BAY.h5 / adj_mx.pkl / adj_mx_bay.pkl",
        "file_path": str(DATA_DIR),
        "file_format": "HDF5 (.h5) & Python Pickle (.pkl)",
        "file_size_kb": round(p5.stat().st_size / 1024, 1) if e5 else 0,
        "file_found": e5,
        "loading_status": "Excluded (Non-Mumbai)",
        "status_badge_type": "muted",
        "geographic_coverage": "Los Angeles / Bay Area, California, USA (Strictly Non-Mumbai)",
        "is_mumbai_specific": False,
        "reporting_period": "Caltrans PeMS Highway Sensor Records (US)",
        "source": "Caltrans Performance Measurement System (PeMS)",
        "source_verification_status": "Verified US Sensor Data (Irrelevant to Mumbai)",
        "data_quality_status": "Excluded from Mumbai Analytics",
        "suitable_use": "None in Mumbai CityPulse — strictly excluded per challenge rules",
        "limitations": [
            "Represents Los Angeles / California highway loop detectors, NOT Mumbai road networks",
            "Must NEVER be displayed or integrated into Mumbai dashboards",
            "Documented here solely to record and verify its deliberate exclusion"
        ],
        "record_counts": "325 US Highway loop detectors (0 Mumbai records)"
    })

    return datasets
