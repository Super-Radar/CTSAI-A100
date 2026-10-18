import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_A_DIR = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果"
DATA_B_DIR = (
    REPO_ROOT
    / "ADC数据采集"
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "data"
)
