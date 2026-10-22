from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import streamlit as st

from config import DATA_DIR, TRAFFIC_CSV_NAME

@st.cache_data(show_spinner="Loading and filtering Maharashtra traffic CSV...")
def load_traffic_csv(file_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(file_path) if file_path else DATA_DIR / TRAFFIC_CSV_NAME
    if not path.exists():
        return {
            "status": "missing_file",
            "error": f"Traffic dataset CSV not found at {path}",
            "file_name": TRAFFIC_CSV_NAME,
            "file_found": False
        }

    try:
        raw_df = pd.read_csv(path)
        total_rows_raw = len(raw_df)

        # Strict filtering to Mumbai City only
        if "City" in raw_df.columns:
            mumbai_df = raw_df[raw_df["City"].astype(str).str.strip().str.upper() == "MUMBAI"].copy()
            non_mumbai_cities = [c for c in raw_df["City"].unique() if str(c).strip().upper() != "MUMBAI"]
        else:
            mumbai_df = pd.DataFrame()
            non_mumbai_cities = []

        total_rows_mumbai = len(mumbai_df)
        excluded_rows = total_rows_raw - total_rows_mumbai

        # Clean numeric fields
        numeric_cols = ["Vehicle_Count", "Accidents", "Traffic_Violations", "Avg_Speed_kmph"]
        for col in numeric_cols:
            if col in mumbai_df.columns:
                mumbai_df[col] = pd.to_numeric(mumbai_df[col], errors="coerce")

        if "Date" in mumbai_df.columns:
            mumbai_df["Date"] = pd.to_datetime(mumbai_df["Date"], errors="coerce")
            mumbai_df = mumbai_df.sort_values(by="Date")

        min_date = mumbai_df["Date"].min().strftime("%Y-%m-%d") if not mumbai_df.empty and pd.notnull(mumbai_df["Date"].min()) else None
        max_date = mumbai_df["Date"].max().strftime("%Y-%m-%d") if not mumbai_df.empty and pd.notnull(mumbai_df["Date"].max()) else None

        # Column stats
        columns_info = []
        for col in mumbai_df.columns:
            null_count = int(mumbai_df[col].isnull().sum())
            null_pct = round((null_count / total_rows_mumbai) * 100, 2) if total_rows_mumbai > 0 else 0.0
            columns_info.append({
                "column_name": col,
                "data_type": str(mumbai_df[col].dtype),
                "null_count": null_count,
                "null_pct": null_pct
            })

        # Summary stats
        summary_stats = {}
        for col in numeric_cols:
            if col in mumbai_df.columns and not mumbai_df[col].dropna().empty:
                s = mumbai_df[col].dropna()
                summary_stats[col] = {
                    "count": int(s.count()),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "min": float(s.min()),
                    "median": round(float(s.median()), 2),
                    "max": float(s.max())
                }

        # Speed by area
        speed_by_area = []
        if "Area" in mumbai_df.columns and "Avg_Speed_kmph" in mumbai_df.columns:
            for area, group in mumbai_df.groupby("Area"):
                speed_by_area.append({
                    "Area": area,
                    "Avg_Speed_kmph": round(float(group["Avg_Speed_kmph"].mean()), 1),
                    "Sample_Count": int(len(group))
                })
            speed_by_area.sort(key=lambda x: x["Avg_Speed_kmph"])

        return {
            "status": "loaded",
            "file_found": True,
            "metadata": {
                "dataset_name": "Maharashtra Traffic Dataset CSV (Filtered to Mumbai City)",
                "file_name": TRAFFIC_CSV_NAME,
                "file_size_bytes": path.stat().st_size,
                "file_format": "CSV",
                "geographic_scope": "Mumbai City records only (filtered from Maharashtra)",
                "reporting_period": f"{min_date} to {max_date}",
                "total_rows_raw": total_rows_raw,
                "mumbai_rows_retained": total_rows_mumbai,
                "non_mumbai_rows_excluded": excluded_rows,
                "excluded_cities": non_mumbai_cities,
                "source_verification_status": "Unverified — source and measurement methodology not confirmed",
                "data_quality_status": "Exploratory Prototyping Only",
                "disclaimer": "Source and measurement methodology have not been verified. This data is for exploratory prototyping only. Do not use for real-world safety decisions.",
                "limitations": [
                    "Speed, vehicle count, and congestion values are not confirmed physical sensor measurements",
                    "No metadata or sensor calibration documentation provided",
                    "Do NOT train models on this dataset in Phase 1",
                    "Do NOT combine these values with official challan records"
                ]
            },
            "columns_info_df": pd.DataFrame(columns_info),
            "summary_stats": summary_stats,
            "speed_by_area_df": pd.DataFrame(speed_by_area),
            "mumbai_df": mumbai_df,
            "raw_df": raw_df,
            "areas": mumbai_df["Area"].dropna().unique().tolist() if "Area" in mumbai_df.columns else [],
            "vehicle_types": mumbai_df["Vehicle_Type"].dropna().unique().tolist() if "Vehicle_Type" in mumbai_df.columns else [],
            "congestion_levels": mumbai_df["Congestion_Level"].dropna().unique().tolist() if "Congestion_Level" in mumbai_df.columns else []
        }

    except Exception as e:
        return {
            "status": "load_error",
            "error": str(e),
            "file_name": TRAFFIC_CSV_NAME,
            "file_found": True
        }
