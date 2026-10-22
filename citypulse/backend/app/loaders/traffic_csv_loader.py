import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.config import DATA_DIR, TRAFFIC_CSV_NAME

logger = logging.getLogger(__name__)

class TrafficCsvLoader:
    def __init__(self):
        self.file_path = DATA_DIR / TRAFFIC_CSV_NAME
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
                "error": f"Traffic dataset CSV not found at {self.file_path}",
                "records": []
            }

        try:
            raw_df = pd.read_csv(self.file_path)
            self._raw_df = raw_df

            total_rows_before = len(raw_df)
            total_cities = raw_df["City"].dropna().unique().tolist() if "City" in raw_df.columns else []

            # Strict filtering to Mumbai City only
            if "City" in raw_df.columns:
                mumbai_df = raw_df[raw_df["City"].astype(str).str.strip().str.upper() == "MUMBAI"].copy()
            else:
                mumbai_df = pd.DataFrame()

            # Ensure numeric fields are properly typed
            numeric_cols = ["Vehicle_Count", "Accidents", "Traffic_Violations", "Avg_Speed_kmph"]
            for col in numeric_cols:
                if col in mumbai_df.columns:
                    mumbai_df[col] = pd.to_numeric(mumbai_df[col], errors="coerce")

            if "Date" in mumbai_df.columns:
                mumbai_df["Date"] = pd.to_datetime(mumbai_df["Date"], errors="coerce")
                mumbai_df = mumbai_df.sort_values(by="Date")

            self._mumbai_df = mumbai_df
            return self._build_result()

        except Exception as e:
            logger.error(f"Error loading traffic CSV: {e}", exc_info=True)
            return {
                "status": "load_error",
                "error": str(e),
                "records": []
            }

    def _build_result(self) -> Dict[str, Any]:
        raw_df = self._raw_df
        df = self._mumbai_df

        if df is None or raw_df is None:
            return {"status": "not_loaded", "records": []}

        total_rows_before = len(raw_df)
        total_rows_after = len(df)
        excluded_rows_count = total_rows_before - total_rows_after
        non_mumbai_cities = [c for c in raw_df["City"].unique() if str(c).strip().upper() != "MUMBAI"]

        # Date range
        min_date = df["Date"].min().strftime("%Y-%m-%d") if not df.empty and pd.notnull(df["Date"].min()) else None
        max_date = df["Date"].max().strftime("%Y-%m-%d") if not df.empty and pd.notnull(df["Date"].max()) else None

        # Column metadata and null counts
        columns_info = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / total_rows_after) * 100, 2) if total_rows_after > 0 else 0.0
            columns_info.append({
                "column_name": col,
                "data_type": str(df[col].dtype),
                "null_count": null_count,
                "null_percentage": null_pct
            })

        # Summary statistics
        numeric_stats = {}
        for col in ["Vehicle_Count", "Accidents", "Traffic_Violations", "Avg_Speed_kmph"]:
            if col in df.columns and not df[col].dropna().empty:
                s = df[col].dropna()
                numeric_stats[col] = {
                    "count": int(s.count()),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "min": float(s.min()),
                    "p25": round(float(np.percentile(s, 25)), 2),
                    "median": round(float(s.median()), 2),
                    "p75": round(float(np.percentile(s, 75)), 2),
                    "max": float(s.max())
                }

        # Area breakdown
        area_counts = df["Area"].value_counts().to_dict() if "Area" in df.columns else {}
        vehicle_type_counts = df["Vehicle_Type"].value_counts().to_dict() if "Vehicle_Type" in df.columns else {}
        congestion_counts = df["Congestion_Level"].value_counts().to_dict() if "Congestion_Level" in df.columns else {}

        # Speed distribution by area
        speed_by_area = []
        if "Area" in df.columns and "Avg_Speed_kmph" in df.columns:
            for area, group in df.groupby("Area"):
                speed_by_area.append({
                    "area": area,
                    "avg_speed": round(float(group["Avg_Speed_kmph"].mean()), 1),
                    "median_speed": round(float(group["Avg_Speed_kmph"].median()), 1),
                    "sample_count": int(len(group))
                })
            speed_by_area.sort(key=lambda x: x["avg_speed"])

        # Congestion breakdown by area
        congestion_by_area = []
        if "Area" in df.columns and "Congestion_Level" in df.columns:
            cross_tab = pd.crosstab(df["Area"], df["Congestion_Level"]).to_dict(orient="index")
            for area, counts in cross_tab.items():
                congestion_by_area.append({
                    "area": area,
                    "Low": counts.get("Low", 0),
                    "Medium": counts.get("Medium", 0),
                    "High": counts.get("High", 0)
                })

        return {
            "status": "loaded",
            "metadata": {
                "dataset_name": "Maharashtra Traffic Dataset CSV (Mumbai Filtered)",
                "file_name": TRAFFIC_CSV_NAME,
                "file_size_bytes": self.get_file_size(),
                "file_format": "CSV",
                "geographic_scope": "Mumbai City records only (filtered from Maharashtra)",
                "reporting_period": f"{min_date} to {max_date}",
                "total_rows_raw": total_rows_before,
                "mumbai_rows_retained": total_rows_after,
                "non_mumbai_rows_excluded": excluded_rows_count,
                "excluded_cities": non_mumbai_cities,
                "source_verification_status": "Unverified — source and measurement methodology not confirmed",
                "data_quality_status": "Prototyping / Exploratory Only",
                "disclaimer": "Source and measurement methodology have not been verified. This data is for exploratory prototyping only. Do not use for real-world safety decisions.",
                "suitable_use": "UI/UX exploration, filter prototyping, schema verification, future ML architecture wiring",
                "limitations": [
                    "Speed, vehicle count, and congestion values are not confirmed as verified physical sensor measurements",
                    "No metadata or sensor calibration documentation provided for data acquisition methodology",
                    "Do NOT train models on this dataset in Phase 1",
                    "Do NOT combine these values with official challan records or IMD rainfall data"
                ]
            },
            "columns_info": columns_info,
            "filter_options": {
                "areas": list(area_counts.keys()),
                "vehicle_types": list(vehicle_type_counts.keys()),
                "congestion_levels": list(congestion_counts.keys()),
                "min_date": min_date,
                "max_date": max_date
            },
            "summary_statistics": numeric_stats,
            "distributions": {
                "by_area": area_counts,
                "by_vehicle_type": vehicle_type_counts,
                "by_congestion": congestion_counts,
                "speed_by_area": speed_by_area,
                "congestion_by_area": congestion_by_area
            }
        }

    def get_filtered_records(
        self,
        area: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        congestion_level: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        if self._mumbai_df is None:
            self.load()

        df = self._mumbai_df
        if df is None or df.empty:
            return {"total_records": 0, "page": page, "page_size": page_size, "records": []}

        filtered = df.copy()

        if area and area != "all":
            filtered = filtered[filtered["Area"].astype(str).str.lower() == area.lower()]
        if vehicle_type and vehicle_type != "all":
            filtered = filtered[filtered["Vehicle_Type"].astype(str).str.lower() == vehicle_type.lower()]
        if congestion_level and congestion_level != "all":
            filtered = filtered[filtered["Congestion_Level"].astype(str).str.lower() == congestion_level.lower()]
        if start_date:
            filtered = filtered[filtered["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered["Date"] <= pd.to_datetime(end_date)]

        total_matching = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        page_df = filtered.iloc[start_idx:end_idx]
        
        # Format date for JSON output
        records = []
        for _, row in page_df.iterrows():
            r = row.to_dict()
            if pd.notnull(r.get("Date")):
                r["Date"] = r["Date"].strftime("%Y-%m-%d")
            # Replace NaNs with None for valid JSON
            r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            records.append(r)

        return {
            "total_records": total_matching,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_matching + page_size - 1) // page_size if page_size > 0 else 1,
            "records": records
        }
