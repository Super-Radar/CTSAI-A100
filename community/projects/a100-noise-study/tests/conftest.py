"""Make the src/ layout importable and provide shared fixtures."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../community/projects/a100-noise-study
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402

from noise_study.pipeline import default_data_dir, default_hxx_dir, repo_root  # noqa: E402


@pytest.fixture(scope="session")
def repo_root_dir() -> Path:
    return repo_root()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d = default_data_dir()
    if not d.is_dir():
        pytest.skip(f"example ADC data not found under {d}")
    return d


@pytest.fixture(scope="session")
def hxx_dir() -> Path:
    d = default_hxx_dir()
    if not d.is_dir():
        pytest.skip(f".hxx config directory not found under {d}")
    return d
