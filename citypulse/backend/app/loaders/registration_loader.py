import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from app.config import DATA_DIR, REGISTRATION_CSV_NAME

logger = logging.getLogger(__name__)

class RegistrationLoader:
    def __init__(self):
        self.file_path = DATA_DIR / REGISTRATION_CSV_NAME
        self._df: Optional[pd.DataFrame] = None

    def exists(self) -> bool:
        return self.file_path.exists()

    def get_file_size(self) -> int:
        return self.file_path.stat().st_size if self.exists() else 0

    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        if not self.exists():
            return {
                "status": "missing_file",
                "error": f"Registration CSV file not found at {self.file_path}",
                "records": []
            }

        try:
            # Read CSV (may contain UTF-8 BOM)
            df = pd.read_csv(self.file_path, encoding="utf-8-sig")
            # Drop trailing unnamed empty columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Inspect columns and geography
            cols = df.columns.tolist()
            has_mumbai = any("mumbai" in str(c).lower() for c in cols)
            for col in cols:
                if df[col].astype(str).str.contains("mumbai", case=False).any():
                    has_mumbai = True

            records = []
            for _, row in df.iterrows():
                cat = str(row.iloc[0]).strip()
                if not cat or cat == "nan":
                    continue
                val_2022 = str(row.iloc[1]).strip() if len(row) > 1 else None
                val_2023 = str(row.iloc[2]).strip() if len(row) > 2 else None
                pct_change = float(row.iloc[3]) if len(row) > 3 and pd.notnull(row.iloc[3]) else None

                records.append({
                    "vehicle_category": cat,
                    "registered_count_2022": val_2022,
                    "registered_count_2023": val_2023,
                    "percentage_change": pct_change
                })

            return {
                "status": "loaded",
                "geographic_suitability": "Not suitable for Mumbai-only analysis",
                "has_mumbai_scope": has_mumbai,
                "metadata": {
                    "dataset_name": "Vehicle Category Registration Summary CSV",
                    "file_name": REGISTRATION_CSV_NAME,
                    "file_size_bytes": self.get_file_size(),
                    "file_format": "CSV",
                    "geographic_scope": "State-wide / Unspecified National Scope (No Mumbai City Identifier)",
                    "reporting_period": "2022-01-01 to 2023-01-01",
                    "source_verification_status": "Unverified administrative scope",
                    "data_quality_status": "Valid tabular data, but lacks city-level geographic disaggregation",
                    "disclaimer": "Vehicle registrations are not the same as real-time road traffic counts. Excluded from Mumbai-specific metrics due to missing city identifier.",
                    "suitable_use": "Macro fleet composition reference only; must not be treated as live street traffic",
                    "limitations": [
                        "File does not specify whether figures are Mumbai City, MMR region, Maharashtra State, or National",
                        "Vehicle registrations indicate vehicle ownership stock, not active on-road circulation volumes",
                        "Excluded from Mumbai KPI cards and Mumbai-only charts"
                    ]
                },
                "records": records
            }
        except Exception as e:
            logger.error(f"Error loading registration CSV: {e}", exc_info=True)
            return {
                "status": "load_error",
                "error": str(e),
                "records": []
            }
