import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("CITYPULSE_DATA_DIR", PROJECT_DIR / "data"))
CACHE_DIR = Path(os.getenv("CITYPULSE_CACHE_DIR", BASE_DIR / "cache"))

# Ensure cache dir exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Dataset filenames
CHALLAN_PDF_NAME = "141d484b-bbb2-4b93-8797-6e88533fd506.pdf"
TRAFFIC_CSV_NAME = "Traffic dataset.csv"
RAINFALL_CSV_NAME = "rainfall_districtwise_daily_imd.csv"
REGISTRATION_CSV_NAME = "babfb7a0-992e-487b-97dd-220d0290122e.csv"

# Non-Mumbai benchmark files (to document exclusion)
METR_LA_FILES = ["PEMS-BAY.h5", "adj_mx.pkl", "adj_mx_bay.pkl"]

# Geographic scope
GEOGRAPHIC_SCOPE = "Mumbai City, Maharashtra, India"
APP_NAME = "CityPulse"
APP_TAGLINE = "TrafficPulse AI — Predict. Explain. Act."
API_VERSION = "1.0.0"
PHASE = "Phase 1: Mumbai-Only Website Skeleton & Descriptive Analytics"
