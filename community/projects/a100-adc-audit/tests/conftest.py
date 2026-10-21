"""pytest 共享夹具：src 路径注入、最小配置、合成 ADC 数据生成。"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC = PROJECT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 仓库根（tests -> a100-adc-audit -> projects -> community -> 仓库根）
REPO_ROOT = Path(__file__).resolve().parents[4]

HXX_TEMPLATE = """/* minimal synthetic profile for tests */
\t.fmcw_startfreq = {startfreq},
\t.fmcw_bandwidth = {bw},
\t.fmcw_chirp_rampup = 43,
\t.fmcw_chirp_period = 48,
\t.nchirp = {nchirp},
\t.adc_freq = 25,
\t.dec_factor = 1,
\t.adc_sample_start = 2,
\t.adc_sample_end = {sample_end},
\t.rng_nfft = {rng_nfft},
\t.vel_nfft = {vel_nfft},
"""

# 标准测试配置：rng_nfft=256, nchirp=16 -> 期望 128*16 = 2048 字 / 4096 样本
STD_RNG_NFFT = 256
STD_NCHIRP = 16
STD_EXPECTED = (STD_RNG_NFFT // 2) * STD_NCHIRP  # 2048


@pytest.fixture()
def std_config(tmp_path):
    from adc_audit.config import parse_hxx

    p = tmp_path / "sensor_config_init1.hxx"
    p.write_text(
        HXX_TEMPLATE.format(
            startfreq=76.119,
            bw=750,
            nchirp=STD_NCHIRP,
            sample_end=12.24,
            rng_nfft=STD_RNG_NFFT,
            vel_nfft=64,
        ),
        encoding="utf-8",
    )
    return parse_hxx(p)


def make_words(n: int, seed: int = 7) -> np.ndarray:
    """确定性合成 uint32 字：高低 16 位均在 ±20000 内（无饱和、无常值 chirp）。"""
    rng = np.random.default_rng(seed)
    hi = rng.integers(-20000, 20000, size=n).astype(np.int64)
    lo = rng.integers(-20000, 20000, size=n).astype(np.int64)
    return (((hi & 0xFFFF) << 16) | (lo & 0xFFFF)).astype(np.uint32)


def write_adc(path: Path, words, header: tuple[int, int, int] | None = None) -> Path:
    """按仓内布局写一个 ADC txt（逗号分隔，无换行）。"""
    vals = ([*header] if header else []) + [int(w) for w in words]
    path.write_text(",".join(map(str, vals)), encoding="utf-8")
    return path


def make_channel_dir(
    base: Path, profile: int = 1, n_rx: int = 4, n_words: int = STD_EXPECTED
) -> Path:
    """生成一个含 n_rx 个完好通道文件的目录（f{profile}_{i}_... 命名）。"""
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n_rx):
        write_adc(
            base / f"adc_test_20260512141857_f{profile}_{i}_14_19_0{i}_000.txt",
            make_words(n_words, seed=100 + i),
        )
    return base
