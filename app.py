import sys
import runpy
from pathlib import Path

# Ensure citypulse package root is in sys.path
CITYPULSE_DIR = Path(__file__).resolve().parent / "citypulse"
if str(CITYPULSE_DIR) not in sys.path:
    sys.path.insert(0, str(CITYPULSE_DIR))

# Execute citypulse/app.py
runpy.run_path(str(CITYPULSE_DIR / "app.py"), run_name="__main__")
