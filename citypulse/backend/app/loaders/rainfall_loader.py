import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.config import DATA_DIR, RAINFALL_CSV_NAME

logger = logging.getLogger(__name__)

class RainfallLoader:
    def __init__(self):
        self.file_path = DATA_DIR / RAINFALL_CSV_NAME
        self._raw_df: Optional[pd.DataFrame] = None
        self._mumbai_df: Optional[pd.DataFrame] = None

    def exists(self) -> bool:
        return self.file_path.exists()

    def get_file_size(self) -> int:
        return self.file_path.stat().st_size if self.exists() else 0

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        if self._mumbai_df is not None and not force_reload:
            return self._build_result()

        if not self.exists():
            return {
                "status": "missing_file",
                "error": f"IMD Rainfall CSV not found at {self.file_path}",
                "records": []
            }

        try:
            raw_df = pd.read_csv(self.file_path)
            self._raw_df = raw_df

            total_rows_raw = len(raw_df)

            # Filter strictly to Maharashtra and Mumbai City
            is_maha = raw_df["State"].astype(str).str.strip().str.upper() == "MAHARASHTRA"
            is_mumbai_city = raw_df["District"].astype(str).str.strip().str.upper() == "MUMBAI CITY"

            mumbai_df = raw_df[is_maha & is_mumbai_city].copy()

            # Clean and parse columns
            if "Date" in mumbai_df.columns:
                mumbai_df["Date"] = pd.to_datetime(mumbai_df["Date"], errors="coerce")
                mumbai_df = mumbai_df.sort_values(by="Date")

            # Clean numeric columns (handle commas or spaces)
            num_cols = ["Daily Actual", "Daily Normal", "Cumulative Actual", "Cumulative Normal"]
            for col in num_cols:
                if col in mumbai_df.columns:
                    mumbai_df[col] = pd.to_numeric(
                        mumbai_df[col].astype(str).str.replace(",", "").str.strip(),
                        errors="coerce"
                    )

            # Clean departure % columns
            for col in ["Daily Departure Per", "Cumulative Departue Per"]:
                if col in mumbai_df.columns:
                    mumbai_df[col + "_num"] = pd.to_numeric(
                        mumbai_df[col].astype(str).str.replace("%", "").str.replace(",", "").str.strip(),
                        errors="coerce"
                    )

            self._mumbai_df = mumbai_df
            return self._build_result()

        except Exception as e:
            logger.error(f"Error loading rainfall CSV: {e}", exc_info=True)
            return {
                "status": "load_error",
                "error": str(e),
                "records": []
            }

    def _build_result(self) -> Dict[str, Any]:
        df = self._mumbai_df
        raw_df = self._raw_df

        if df is None or raw_df is None:
            return {"status": "not_loaded", "records": []}

        min_date = df["Date"].min().strftime("%Y-%m-%d") if not df.empty and pd.notnull(df["Date"].min()) else None
        max_date = df["Date"].max().strftime("%Y-%m-%d") if not df.empty and pd.notnull(df["Date"].max()) else None

        # Format records for charts
        records = []
        for _, row in df.iterrows():
            date_str = row["Date"].strftime("%Y-%m-%d") if pd.notnull(row["Date"]) else None
            daily_act = float(row["Daily Actual"]) if pd.notnull(row.get("Daily Actual")) else 0.0
            daily_norm = float(row["Daily Normal"]) if pd.notnull(row.get("Daily Normal")) else 0.0
            dep_pct = float(row.get("Daily Departure Per_num")) if pd.notnull(row.get("Daily Departure Per_num")) else None
            cat = str(row.get("Daily Category", "")).strip()

            records.append({
                "date": date_str,
                "daily_actual_mm": round(daily_act, 2),
                "daily_normal_mm": round(daily_norm, 2),
                "departure_percent": round(dep_pct, 2) if dep_pct is not None else None,
                "category": cat,
                "cumulative_actual_mm": round(float(row["Cumulative Actual"]), 2) if pd.notnull(row.get("Cumulative Actual")) else None,
                "cumulative_normal_mm": round(float(row["Cumulative Normal"]), 2) if pd.notnull(row.get("Cumulative Normal")) else None
            })

        # Summary statistics
        actual_series = df["Daily Actual"].dropna()
        total_observed_rain = round(float(actual_series.sum()), 2) if not actual_series.empty else 0.0
        max_single_day = round(float(actual_series.max()), 2) if not actual_series.empty else 0.0
        mean_daily = round(float(actual_series.mean()), 2) if not actual_series.empty else 0.0
        rainy_days_count = int((actual_series > 0.1).sum())

        return {
            "status": "loaded",
            "metadata": {
                "dataset_name": "IMD District-wise Daily Rainfall CSV (Mumbai City)",
                "file_name": RAINFALL_CSV_NAME,
                "file_size_bytes": self.get_file_size(),
                "file_format": "CSV",
                "geographic_scope": "Mumbai City District only (filtered from All-India)",
                "reporting_period": f"{min_date} to {max_date}",
                "total_rows_all_india": len(raw_df),
                "mumbai_city_rows": len(df),
                "source": "India Meteorological Department (IMD) District Daily Monitoring",
                "source_verification_status": "Verified Official Meteorological Data",
                "data_quality_status": "Complete for recorded date window",
                "disclaimer": "Rainfall observations describe the district reporting unit and do not necessarily represent conditions at individual roads or junctions. Kept separate from traffic analysis in Phase 1.",
                "suitable_use": "Macro environmental context, monsoon season timeline, precipitation intensity baseline",
                "limitations": [
                    "District-level aggregates do not pinpoint waterlogged junctions or specific arterial bottlenecks",
                    "Do NOT claim that rainfall caused recorded congestion or accidents in Phase 1 without rigorous causal modeling",
                    "Observations currently cover monsoon transition period (August 2026 to October 2026 in provided IMD archive)"
                ]
            },
            "summary": {
                "observation_count": len(df),
                "date_range": f"{min_date} to {max_date}",
                "total_rainfall_mm": total_observed_rain,
                "max_single_day_mm": max_single_day,
                "average_daily_mm": mean_daily,
                "rainy_days_count": rainy_days_count,
                "dry_days_count": len(df) - rainy_days_count
            },
            "records": records
        }
