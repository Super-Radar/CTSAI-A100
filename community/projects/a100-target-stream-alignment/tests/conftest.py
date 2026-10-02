"""Shared pytest configuration: import path + real-data locator."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC = PROJECT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO_ROOT = PROJECT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "RadarTools" / "RadarTools_Release" / "data"

import pytest  # noqa: E402


@pytest.fixture()
def real_data_dir():
    return DATA_DIR if DATA_DIR.exists() else None
