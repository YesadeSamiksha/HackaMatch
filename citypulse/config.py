import os
from pathlib import Path

# Paths
# BASE_DIR is always the directory where config.py lives (the citypulse/ package root)
BASE_DIR = Path(__file__).resolve().parent

# Primary Directory Resolvers
DATA_DIR = Path(os.getenv("CITYPULSE_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("CITYPULSE_MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.getenv("CITYPULSE_REPORTS_DIR", BASE_DIR / "reports"))

def resolve_data_path(filename: str = "") -> Path:
    """
    Robustly resolves the absolute path to a file or directory in data/.
    Searches DATA_DIR, BASE_DIR/data, CWD/citypulse/data, and CWD/data.
    """
    if not filename:
        return DATA_DIR if DATA_DIR.exists() else (BASE_DIR / "data")
    candidates = [
        DATA_DIR / filename,
        BASE_DIR / "data" / filename,
        Path.cwd() / "citypulse" / "data" / filename,
        Path.cwd() / "data" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return DATA_DIR / filename

def resolve_model_path(filename: str = "") -> Path:
    """
    Robustly resolves the absolute path to a file or directory in models/.
    Searches MODELS_DIR, BASE_DIR/models, CWD/citypulse/models, and CWD/models.
    """
    if not filename:
        return MODELS_DIR if MODELS_DIR.exists() else (BASE_DIR / "models")
    candidates = [
        MODELS_DIR / filename,
        BASE_DIR / "models" / filename,
        Path.cwd() / "citypulse" / "models" / filename,
        Path.cwd() / "models" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return MODELS_DIR / filename

def resolve_report_path(filename: str = "") -> Path:
    """
    Robustly resolves the absolute path to a file or directory in reports/.
    Searches REPORTS_DIR, BASE_DIR/reports, CWD/citypulse/reports, and CWD/reports.
    """
    if not filename:
        return REPORTS_DIR if REPORTS_DIR.exists() else (BASE_DIR / "reports")
    candidates = [
        REPORTS_DIR / filename,
        BASE_DIR / "reports" / filename,
        Path.cwd() / "citypulse" / "reports" / filename,
        Path.cwd() / "reports" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return REPORTS_DIR / filename

# Primary Dataset Filenames
PRIMARY_TRAIN_PARQUET = "train.parquet"
SENSOR_LOCATIONS_CSV = "sensor_locations_enriched.csv"
ADJACENCY_MATRIX_PKL = "adj_mx.pkl"

# App Metadata
APP_NAME = "CityPulse"
APP_TAGLINE = "TrafficPulse AI — Predict. Explain. Act."
GEOGRAPHIC_SCOPE = "Los Angeles County Highway Network (Caltrans PeMS District 7)"
TOTAL_MONITORED_NODES = 207
SAMPLING_INTERVAL = "5 minutes"
TIME_SPAN = "March 1, 2012 – May 23, 2012"

# Plotly Theme Colors
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PRIMARY_CYAN = "#06B6D4"
COLOR_SECONDARY_BLUE = "#38BDF8"
COLOR_ACCENT_AMBER = "#F59E0B"
COLOR_SUCCESS_EMERALD = "#10B981"
COLOR_DANGER_ROSE = "#EF4444"
COLOR_PURPLE = "#A855F7"
COLOR_BG_DARK = "#0B0F19"
COLOR_PANEL_DARK = "#111827"
COLOR_BORDER = "#1E293B"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_TEXT_LIGHT = "#F1F5F9"

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(17, 24, 39, 0.7)",
    plot_bgcolor="rgba(11, 15, 25, 0.8)",
    font=dict(family="sans-serif", color=COLOR_TEXT_LIGHT, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER),
    yaxis=dict(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER),
)
