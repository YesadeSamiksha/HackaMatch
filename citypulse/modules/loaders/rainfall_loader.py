from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from config import DATA_DIR, RAINFALL_CSV_NAME

@st.cache_data(show_spinner="Loading IMD Mumbai City rainfall archive...")
def load_rainfall_data(file_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(file_path) if file_path else DATA_DIR / RAINFALL_CSV_NAME
    if not path.exists():
        return {
            "status": "missing_file",
            "error": f"IMD Rainfall CSV not found at {path}",
            "file_name": RAINFALL_CSV_NAME,
            "file_found": False
        }

    try:
        raw_df = pd.read_csv(path)
        total_rows_raw = len(raw_df)

        is_maha = raw_df["State"].astype(str).str.strip().str.upper() == "MAHARASHTRA"
        is_mumbai_city = raw_df["District"].astype(str).str.strip().str.upper() == "MUMBAI CITY"

        mumbai_df = raw_df[is_maha & is_mumbai_city].copy()

        if "Date" in mumbai_df.columns:
            mumbai_df["Date"] = pd.to_datetime(mumbai_df["Date"], errors="coerce")
            mumbai_df = mumbai_df.sort_values(by="Date")

        for col in ["Daily Actual", "Daily Normal", "Cumulative Actual", "Cumulative Normal"]:
            if col in mumbai_df.columns:
                mumbai_df[col] = pd.to_numeric(
                    mumbai_df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce"
                )

        if "Daily Departure Per" in mumbai_df.columns:
            mumbai_df["Daily Departure Per_num"] = pd.to_numeric(
                mumbai_df["Daily Departure Per"].astype(str).str.replace("%", "").str.replace(",", "").str.strip(),
                errors="coerce"
            )

        min_date = mumbai_df["Date"].min().strftime("%Y-%m-%d") if not mumbai_df.empty and pd.notnull(mumbai_df["Date"].min()) else None
        max_date = mumbai_df["Date"].max().strftime("%Y-%m-%d") if not mumbai_df.empty and pd.notnull(mumbai_df["Date"].max()) else None

        actual_series = mumbai_df["Daily Actual"].dropna()
        total_rain = round(float(actual_series.sum()), 2) if not actual_series.empty else 0.0
        max_rain = round(float(actual_series.max()), 2) if not actual_series.empty else 0.0
        avg_rain = round(float(actual_series.mean()), 2) if not actual_series.empty else 0.0
        rainy_days = int((actual_series > 0.1).sum())

        return {
            "status": "loaded",
            "file_found": True,
            "metadata": {
                "dataset_name": "IMD District-wise Daily Rainfall CSV (Mumbai City)",
                "file_name": RAINFALL_CSV_NAME,
                "file_size_bytes": path.stat().st_size,
                "file_format": "CSV",
                "geographic_scope": "Mumbai City District only (filtered from All-India)",
                "reporting_period": f"{min_date} to {max_date}",
                "total_rows_all_india": total_rows_raw,
                "mumbai_city_rows": len(mumbai_df),
                "source": "India Meteorological Department (IMD) District Daily Monitoring",
                "source_verification_status": "Verified Official Meteorological Data",
                "disclaimer": "Rainfall observations describe the district reporting unit and do not necessarily represent conditions at individual roads or junctions. Kept separate from traffic analysis in Phase 1.",
                "limitations": [
                    "District-level aggregates do not pinpoint waterlogged junctions or arterial bottlenecks",
                    "Do NOT claim that rainfall caused recorded congestion or accidents in Phase 1",
                    "Observations cover late monsoon timeline (August to October 2026 in archive)"
                ]
            },
            "summary": {
                "observation_count": len(mumbai_df),
                "date_range": f"{min_date} to {max_date}",
                "total_rainfall_mm": total_rain,
                "max_single_day_mm": max_rain,
                "average_daily_mm": avg_rain,
                "rainy_days_count": rainy_days,
                "dry_days_count": len(mumbai_df) - rainy_days
            },
            "mumbai_df": mumbai_df
        }

    except Exception as e:
        return {
            "status": "load_error",
            "error": str(e),
            "file_name": RAINFALL_CSV_NAME,
            "file_found": True
        }
