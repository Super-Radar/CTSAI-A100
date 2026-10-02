"""Make src/ importable and locate the repository data directory."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# .../<repo>/community/projects/a100-log-buffer-budget/tests/conftest.py
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = REPO_ROOT / "RadarTools" / "RadarTools_Release" / "data"
