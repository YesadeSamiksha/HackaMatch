import sys
from pathlib import Path

# Ensure citypulse package root is always in sys.path
citypulse_path = Path(__file__).resolve().parent / "citypulse"
if citypulse_path.exists() and str(citypulse_path) not in sys.path:
    sys.path.insert(0, str(citypulse_path))

# Ignore standalone backend tests during main suite runs
collect_ignore = ["citypulse/backend/tests"]
