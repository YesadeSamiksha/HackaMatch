from pathlib import Path
from typing import Dict, Any, List

from app.config import DATA_DIR, CHALLAN_PDF_NAME, TRAFFIC_CSV_NAME, RAINFALL_CSV_NAME, REGISTRATION_CSV_NAME, METR_LA_FILES
from app.loaders.challan_loader import ChallanLoader
from app.loaders.traffic_csv_loader import TrafficCsvLoader
from app.loaders.rainfall_loader import RainfallLoader
from app.loaders.registration_loader import RegistrationLoader

class DatasetRegistry:
    def __init__(self):
        self.challan_loader = ChallanLoader()
        self.traffic_loader = TrafficCsvLoader()
        self.rainfall_loader = RainfallLoader()
        self.registration_loader = RegistrationLoader()

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        datasets = [
            self._get_challan_entry(),
            self._get_traffic_csv_entry(),
            self._get_rainfall_entry(),
            self._get_registration_entry(),
            self._get_metr_la_entry()
        ]
        return datasets

    def get_dataset_by_id(self, dataset_id: str) -> Dict[str, Any]:
        all_ds = {d["id"]: d for d in self.get_all_datasets()}
        if dataset_id in all_ds:
            return all_ds[dataset_id]
        return {
            "status": "not_found",
            "error": f"Dataset '{dataset_id}' not found in registry"
        }

    def _get_challan_entry(self) -> Dict[str, Any]:
        p = DATA_DIR / CHALLAN_PDF_NAME
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return {
            "id": "mumbai_traffic_challans_2024",
            "name": "Mumbai Traffic Violations Data 2024",
            "role": "Primary Mumbai-specific dataset",
            "file_name": CHALLAN_PDF_NAME,
            "file_path": str(p),
            "file_format": "PDF (Vector text & tabular grids)",
            "file_size_bytes": size,
            "file_found": exists,
            "loading_status": "Loaded" if exists else "Missing file",
            "geographic_coverage": "Mumbai City, Maharashtra, India",
            "is_mumbai_specific": True,
            "reporting_period": "2024-01-01 to 2024-12-31",
            "source": "Mumbai Traffic Police Department (Official Violations Report)",
            "source_verification_status": "Verified Official Government Report",
            "data_quality_status": "Verified (Sum of all rows matches official Grand Totals)",
            "suitable_use": "Division-wise recorded challan comparisons, police-station comparisons, violation categories, enforcement ranking",
            "limitations": [
                "Represents recorded traffic challans and enforcement activity, not proof of congestion or definitive accident risk",
                "Division-level and police-station-level tables represent distinct reporting hierarchies and must not be combined",
                "High challan counts reflect camera placement and police deployment in addition to infractions"
            ],
            "record_counts": {
                "traffic_divisions": 42,
                "police_stations": 91,
                "violation_categories": 24,
                "vehicle_types": 26,
                "division_challans_total": 5937151,
                "police_station_challans_total": 847766
            }
        }

    def _get_traffic_csv_entry(self) -> Dict[str, Any]:
        p = DATA_DIR / TRAFFIC_CSV_NAME
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return {
            "id": "maharashtra_traffic_csv",
            "name": "Maharashtra Traffic Dataset CSV (Filtered to Mumbai City)",
            "role": "Exploratory prototyping dataset (Strictly filtered)",
            "file_name": TRAFFIC_CSV_NAME,
            "file_path": str(p),
            "file_format": "CSV (Delimited text)",
            "file_size_bytes": size,
            "file_found": exists,
            "loading_status": "Loaded (Prototyping Only)" if exists else "Missing file",
            "geographic_coverage": "Mumbai City only (234 Mumbai rows retained, 966 non-Mumbai rows excluded)",
            "is_mumbai_specific": True,  # When filtered to Mumbai City
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
            "record_counts": {
                "total_rows_raw": 1200,
                "mumbai_rows": 234,
                "excluded_rows": 966
            }
        }

    def _get_rainfall_entry(self) -> Dict[str, Any]:
        p = DATA_DIR / RAINFALL_CSV_NAME
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return {
            "id": "imd_mumbai_rainfall",
            "name": "IMD District-wise Daily Rainfall CSV (Mumbai City)",
            "role": "Mumbai environmental context",
            "file_name": RAINFALL_CSV_NAME,
            "file_path": str(p),
            "file_format": "CSV (Delimited text)",
            "file_size_bytes": size,
            "file_found": exists,
            "loading_status": "Loaded" if exists else "Missing file",
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
                "Covers monsoon transition dates present in archive"
            ],
            "record_counts": {
                "total_rows_all_india": 32800,
                "mumbai_city_observations": 65
            }
        }

    def _get_registration_entry(self) -> Dict[str, Any]:
        p = DATA_DIR / REGISTRATION_CSV_NAME
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return {
            "id": "vehicle_registrations_summary",
            "name": "Vehicle Category Registration Summary CSV",
            "role": "Macro vehicle fleet context (Excluded from Mumbai metrics)",
            "file_name": REGISTRATION_CSV_NAME,
            "file_path": str(p),
            "file_format": "CSV (Delimited text)",
            "file_size_bytes": size,
            "file_found": exists,
            "loading_status": "Not suitable for Mumbai-only analysis",
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
            "record_counts": {
                "vehicle_categories": 10
            }
        }

    def _get_metr_la_entry(self) -> Dict[str, Any]:
        pems_file = DATA_DIR / "PEMS-BAY.h5"
        adj_file = DATA_DIR / "adj_mx.pkl"
        exists = pems_file.exists() or adj_file.exists()
        return {
            "id": "metr_la_pems_benchmark",
            "name": "METR-LA / PEMS-BAY Benchmark Sensor Files",
            "role": "Excluded non-Mumbai benchmark files",
            "file_name": "PEMS-BAY.h5 / adj_mx.pkl / adj_mx_bay.pkl",
            "file_path": str(DATA_DIR),
            "file_format": "HDF5 (.h5) & Python Pickle (.pkl)",
            "file_size_bytes": pems_file.stat().st_size if pems_file.exists() else 0,
            "file_found": exists,
            "loading_status": "Excluded (Non-Mumbai)",
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
            "record_counts": {
                "sensors": 325,
                "mumbai_records": 0
            }
        }
