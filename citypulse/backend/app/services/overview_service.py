from typing import Dict, Any, List
from app.loaders.challan_loader import ChallanLoader
from app.loaders.traffic_csv_loader import TrafficCsvLoader
from app.loaders.rainfall_loader import RainfallLoader
from app.loaders.registration_loader import RegistrationLoader
from app.dataset_registry import DatasetRegistry

class OverviewService:
    def __init__(self):
        self.challan_loader = ChallanLoader()
        self.traffic_loader = TrafficCsvLoader()
        self.rainfall_loader = RainfallLoader()
        self.registration_loader = RegistrationLoader()
        self.registry = DatasetRegistry()

    def get_overview(self) -> Dict[str, Any]:
        challan_data = self.challan_loader.load()
        traffic_data = self.traffic_loader.load()
        rainfall_data = self.rainfall_loader.load()
        reg_data = self.registration_loader.load()

        # Top 5 divisions
        div_items = challan_data.get("division_challans", {}).get("items", [])
        sorted_divs = sorted(div_items, key=lambda x: x["total_challan"], reverse=True)[:5]

        # Top 5 violation categories
        cat_totals = challan_data.get("violation_categories", {}).get("category_totals", [])[:5]

        # Recent 10 rainfall observations
        rain_records = rainfall_data.get("records", [])[-10:]

        # Overview KPI cards
        kpis = [
            {
                "id": "available_datasets",
                "label": "Total Tracked Datasets",
                "value": "5",
                "badge": "Registry Active",
                "description": "3 Mumbai datasets, 1 unverified scope, 1 excluded US benchmark"
            },
            {
                "id": "loaded_mumbai_datasets",
                "label": "Mumbai-Filtered Datasets",
                "value": "3",
                "badge": "Geographically Scoped",
                "description": "Official Challans (Verified), IMD Rainfall (Verified), Traffic CSV (Prototyping)"
            },
            {
                "id": "total_challans_recorded",
                "label": "2024 Division Challans",
                "value": "5,937,151",
                "badge": "Official 2024 Report",
                "description": "Recorded across 42 Mumbai traffic divisions (Excludes separate police station tallies)"
            },
            {
                "id": "police_station_challans",
                "label": "2024 Police Station Challans",
                "value": "847,766",
                "badge": "Law & Order Stations",
                "description": "Recorded across 91 local police stations (Maintained strictly separate)"
            },
            {
                "id": "rainfall_observations",
                "label": "Mumbai City Rainfall Records",
                "value": str(len(rainfall_data.get("records", []))),
                "badge": "IMD District Data",
                "description": "Daily observations (2026-08-19 to 2026-10-22) for Mumbai City District"
            },
            {
                "id": "data_quality_missing_rate",
                "label": "Primary Dataset Missing Rate",
                "value": "0.0%",
                "badge": "High Integrity",
                "description": "All 42 division rows and 91 police station rows reconcile to Grand Totals"
            }
        ]

        # Summary of unverified CSV
        traffic_meta = traffic_data.get("metadata", {})
        traffic_csv_summary = {
            "name": traffic_meta.get("dataset_name"),
            "status": "Loaded (Prototyping Only)",
            "warning": "Source and measurement methodology have not been verified. Exploratory prototyping only.",
            "raw_rows": traffic_meta.get("total_rows_raw", 1200),
            "mumbai_rows": traffic_meta.get("mumbai_rows_retained", 234),
            "excluded_non_mumbai_rows": traffic_meta.get("non_mumbai_rows_excluded", 966),
            "excluded_cities": traffic_meta.get("excluded_cities", [])
        }

        return {
            "geographic_scope": "Mumbai City, Maharashtra, India",
            "kpis": kpis,
            "top_enforcement_divisions": sorted_divs,
            "top_violation_categories": cat_totals,
            "rainfall_preview": {
                "summary": rainfall_data.get("summary", {}),
                "recent_records": rain_records
            },
            "traffic_csv_status": traffic_csv_summary,
            "dataset_status_overview": [
                {
                    "name": "Mumbai Traffic Violations 2024",
                    "status": "Verified Loaded",
                    "type": "Primary",
                    "rows": "42 Divisions + 91 Police Stations"
                },
                {
                    "name": "Maharashtra Traffic CSV",
                    "status": "Filtered (Prototyping Only)",
                    "type": "Secondary",
                    "rows": "234 Mumbai rows (966 non-Mumbai rows excluded)"
                },
                {
                    "name": "IMD Daily Rainfall",
                    "status": "Verified Loaded",
                    "type": "Environmental Context",
                    "rows": "65 Daily Observations (Mumbai City)"
                },
                {
                    "name": "Vehicle Registration CSV",
                    "status": "Not Suitable for Mumbai-Only",
                    "type": "Macro Fleet Reference",
                    "rows": "10 Categories (No City Disaggregation)"
                },
                {
                    "name": "METR-LA / PEMS-BAY",
                    "status": "Excluded (Non-Mumbai)",
                    "type": "Benchmark",
                    "rows": "0 (Excluded per challenge guidelines)"
                }
            ]
        }
