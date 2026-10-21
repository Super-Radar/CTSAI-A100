"""解析 CTSAI-A100 的 sensor_config_initN.hxx 配置文件（只读输入）。

仅提取 ADC 审计所需的标量字段；配置文件本身不做任何修改。
字段含义与官方 MATLAB 平台 `load_adc_data.m` 保持一致：
每通道期望的 uint32 字数 = rng_nfft / 2 * nchirp。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_REQUIRED_FIELDS = (
    "fmcw_startfreq",
    "fmcw_bandwidth",
    "fmcw_chirp_rampup",
    "fmcw_chirp_period",
    "nchirp",
    "adc_freq",
    "adc_sample_start",
    "adc_sample_end",
    "rng_nfft",
    "vel_nfft",
)


@dataclass(frozen=True)
class SensorConfig:
    """一份 hxx 波形配置的审计视图。"""

    profile: int  # init 序号；仓内约定 0=远波 f0，1=近波 f1
    path: Path
    start_freq_ghz: float
    bandwidth_mhz: float
    chirp_rampup_us: float
    chirp_period_us: float
    nchirp: int
    adc_freq_mhz: float
    sample_start_us: float
    sample_end_us: float
    rng_nfft: int
    vel_nfft: int

    @property
    def expected_words(self) -> int:
        """每个 RX 通道期望的 uint32 字数（与官方 load_adc_data.m 一致）。"""
        return (self.rng_nfft // 2) * self.nchirp

    @property
    def window_samples(self) -> int:
        """采样窗内的理论样本数 = 窗长(us) * ADC 频率(MHz)。"""
        span_us = self.sample_end_us - self.sample_start_us
        return int(round(span_us * self.adc_freq_mhz))

    def as_dict(self) -> dict:
        return {
            "profile": self.profile,
            "path": str(self.path),
            "start_freq_ghz": self.start_freq_ghz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "chirp_rampup_us": self.chirp_rampup_us,
            "chirp_period_us": self.chirp_period_us,
            "nchirp": self.nchirp,
            "adc_freq_mhz": self.adc_freq_mhz,
            "sample_start_us": self.sample_start_us,
            "sample_end_us": self.sample_end_us,
            "rng_nfft": self.rng_nfft,
            "vel_nfft": self.vel_nfft,
            "expected_words": self.expected_words,
            "window_samples": self.window_samples,
        }


class ConfigError(ValueError):
    """hxx 解析失败（缺字段或数值非法）。"""


def parse_hxx(path: str | Path) -> SensorConfig:
    """从 hxx 文本中提取审计所需标量字段。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    def scalar(name: str) -> float:
        match = re.search(rf"\.{name}\s*=\s*([-+0-9.eE]+)\s*,", text)
        if not match:
            raise ConfigError(f"{path.name}: 缺少字段 {name}")
        return float(match.group(1))

    values: dict[str, float] = {}
    for name in _REQUIRED_FIELDS:
        try:
            values[name] = scalar(name)
        except ConfigError:
            raise
    if values["rng_nfft"] <= 0 or values["nchirp"] <= 0:
        raise ConfigError(f"{path.name}: rng_nfft/nchirp 必须为正")
    if int(values["rng_nfft"]) % 2 != 0:
        raise ConfigError(f"{path.name}: rng_nfft 必须为偶数 (每字 2 样本)")

    profile_match = re.search(r"sensor_config_init(\d+)\.hxx$", path.name, re.IGNORECASE)
    profile = int(profile_match.group(1)) if profile_match else -1

    return SensorConfig(
        profile=profile,
        path=path.resolve(),
        start_freq_ghz=values["fmcw_startfreq"],
        bandwidth_mhz=values["fmcw_bandwidth"],
        chirp_rampup_us=values["fmcw_chirp_rampup"],
        chirp_period_us=values["fmcw_chirp_period"],
        nchirp=int(values["nchirp"]),
        adc_freq_mhz=values["adc_freq"],
        sample_start_us=values["adc_sample_start"],
        sample_end_us=values["adc_sample_end"],
        rng_nfft=int(values["rng_nfft"]),
        vel_nfft=int(values["vel_nfft"]),
    )
