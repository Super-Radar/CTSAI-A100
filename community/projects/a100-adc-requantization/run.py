#!/usr/bin/env python3
"""Convenience launcher for the requantisation study.

Usage (Python 3.12 with numpy/scipy/matplotlib installed)::

    python run.py                 # auto-detects the repo root
    python run.py --repo-root /path/to/CTSAI-A100

Equivalent to ``PYTHONPATH=src python -m requant.cli``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from requant.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
