import sys
from pathlib import Path

# Make src/requant importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
