from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from config import DATA_DIR, REGISTRATION_CSV_NAME

@st.cache_data(show_spinner="Auditing vehicle registration records...")
def load_registration_csv(file_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(file_path) if file_path else DATA_DIR / REGISTRATION_CSV_NAME
    if not path.exists():
        return {
            "status": "missing_file",
            "error": f"Registration CSV not found at {path}",
            "file_name": REGISTRATION_CSV_NAME,
            "file_found": False
        }

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        records = []
        for _, row in df.iterrows():
            cat = str(row.iloc[0]).strip()
            if not cat or cat == "nan":
                continue
            val_2022 = str(row.iloc[1]).strip() if len(row) > 1 else None
            val_2023 = str(row.iloc[2]).strip() if len(row) > 2 else None
            pct_change = float(row.iloc[3]) if len(row) > 3 and pd.notnull(row.iloc[3]) else None

            records.append({
                "Category of Vehicles": cat,
                "01-01-2022": val_2022,
                "01-01-2023": val_2023,
                "%age change": pct_change
            })

        parsed_df = pd.DataFrame(records)

        return {
            "status": "loaded",
            "file_found": True,
            "geographic_suitability": "Not suitable for Mumbai-only analysis",
            "has_mumbai_scope": False,
            "metadata": {
                "dataset_name": "Vehicle Category Registration Summary CSV",
                "file_name": REGISTRATION_CSV_NAME,
                "file_size_bytes": path.stat().st_size,
                "file_format": "CSV",
                "geographic_scope": "State-wide / Unspecified National Scope (No Mumbai City Identifier)",
                "reporting_period": "2022-01-01 to 2023-01-01",
                "source_verification_status": "Unverified administrative scope",
                "data_quality_status": "Not suitable for Mumbai-only analysis",
                "disclaimer": "Vehicle registrations are not the same as real-time road traffic counts. Excluded from Mumbai-specific metrics due to missing city identifier.",
                "limitations": [
                    "File does not specify whether figures represent Mumbai City, MMR region, or State-wide totals",
                    "Registrations indicate vehicle ownership stock, not active on-road circulation volumes",
                    "Excluded from Mumbai KPI cards and Mumbai-only charts"
                ]
            },
            "df": parsed_df
        }

    except Exception as e:
        return {
            "status": "load_error",
            "error": str(e),
            "file_name": REGISTRATION_CSV_NAME,
            "file_found": True
        }
