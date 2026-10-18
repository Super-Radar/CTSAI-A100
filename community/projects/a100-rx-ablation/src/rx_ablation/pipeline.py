# -*- coding: utf-8 -*-
"""官方 CTASI-A100 信号处理链路的 Python 复现（严格功率域）。

本模块逐条移植厂商 MATLAB 代码，行号与文件一一对应，勿改变语义：

- ``init_function/get_radar_paramete.m``
  .hxx 解析与派生参数（adc_sample_num 的取整分支、rng_res / V_res 公式）。
- ``init_function/load_adc_data.m``
  无头 uint32 逗号分隔文本 -> 截断/补零到 ``rng_nfft/2 * nchirp`` 个字
  -> 列优先 reshape 为 ``[rng_nfft/2, nchirp]`` -> 转置为 chirp 行。
  注意：截断发生在解包**之前**，文件首个值是有效样本，丢弃的是末尾多余值。
- ``init_function/adc_mem_2_real_fixed.m``
  每个 uint32 字 -> 高 16 位在前、低 16 位在后的两个有符号 int16 样本。
- ``FFT/do_fft_2d.m``
  距离维 hann(adc_sample_num) 只作用于前 adc_sample_num 个样本 -> Range FFT
  到 rng_nfft（隐式补零）并除以 rng_fft_scalar -> 取前 rng_nfft/2 行 ->
  速度维 hann(nchirp/nvirtual_chirp) -> Doppler FFT 到 vel_nfft 并除以
  vel_fft_scalar。
- ``ct_signal_processing_main_simple_CTASIA100.m``
  官方“非相干积累”定义::

      RVimage = sum_ch power(|velFftData(:,:,ch)|, 2) / nvirtual_array

  即各通道**功率图的平均**（本模块 ``accumulate_power``）。

诚实边界：仅工作在功率域。四通道采集是否严格同步未知，本模块与整个项目
不做任何通道间相位 / 同步性 / 标定断言。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal.windows import hann

# get_radar_paramete.m 全程使用 3e8 而非 299792458，保持官方常数。
C_LIGHT = 3e8


def matlab_round(x: float) -> float:
    """MATLAB round()：四舍五入远离零（Python 内建 round 为银行家舍入）。"""
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


_LINE_RE = re.compile(r"^\s*\.([A-Za-z_]\w*)\s*=\s*(.+?),\s*$")


def _parse_scalar(tok: str):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
        return tok[1:-1]
    if tok == "true":
        return True
    if tok == "false":
        return False
    if tok.lower().startswith("0x"):
        return int(tok, 16)
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def parse_hxx(path: str) -> Dict[str, object]:
    """解析 .hxx 配置中的 ``.key = value,`` 行。

    支持：十进制数、0x 十六进制、带引号字符串、true/false、
    单行 ``{a, b, c}`` 列表。与官方 textscan 流程等价（仅覆盖本仓库
    两个配置文件中出现的形态，多行列表不在需求内）。
    """
    out: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if val.startswith("{") and val.endswith("}"):
                inner = val[1:-1].strip()
                out[key] = [_parse_scalar(t) for t in inner.split(",")] if inner else []
            else:
                out[key] = _parse_scalar(val)
    return out


@dataclass
class RadarConfig:
    """从 .hxx 解析的雷达配置 + 官方派生参数（见 get_radar_paramete.m）。"""

    source_file: str
    fmcw_startfreq: float          # GHz
    fmcw_bandwidth: float          # MHz
    fmcw_chirp_rampup: float       # us
    fmcw_chirp_period: float       # us
    nchirp: int
    adc_freq: float                # MHz
    dec_factor: int
    adc_sample_start: float        # us
    adc_sample_end: float          # us
    rng_win: str
    vel_win: str
    rng_nfft: int
    vel_nfft: int
    rng_fft_scalar: int
    vel_fft_scalar: int
    tx_groups: List[int]
    anti_velamb_en: bool = False
    anti_velamb_delay: float = 10.0

    # ---- 官方派生量（get_radar_paramete.m 逐式对应） -------------------
    @classmethod
    def from_hxx(cls, path: str) -> "RadarConfig":
        raw = parse_hxx(path)
        return cls(
            source_file=os.path.abspath(path),
            fmcw_startfreq=float(raw["fmcw_startfreq"]),
            fmcw_bandwidth=float(raw["fmcw_bandwidth"]),
            fmcw_chirp_rampup=float(raw["fmcw_chirp_rampup"]),
            fmcw_chirp_period=float(raw["fmcw_chirp_period"]),
            nchirp=int(raw["nchirp"]),
            adc_freq=float(raw["adc_freq"]),
            dec_factor=int(raw["dec_factor"]),
            adc_sample_start=float(raw["adc_sample_start"]),
            adc_sample_end=float(raw["adc_sample_end"]),
            rng_win=str(raw["rng_win"]),
            vel_win=str(raw["vel_win"]),
            rng_nfft=int(raw["rng_nfft"]),
            vel_nfft=int(raw["vel_nfft"]),
            rng_fft_scalar=int(raw["rng_fft_scalar"]),
            vel_fft_scalar=int(raw["vel_fft_scalar"]),
            tx_groups=[int(v) for v in raw["tx_groups"]],
            anti_velamb_en=bool(raw.get("anti_velamb_en", False)),
            anti_velamb_delay=float(raw.get("anti_velamb_delay", 10.0)),
        )

    @property
    def nvirtual_chirp(self) -> int:
        """get_radar_paramete.m：从 max(tx_groups) 高位起找第一个非零 TX 组。"""
        tmp_tx = max(self.tx_groups)
        for itx in range(4, 0, -1):
            bit_pos = 1 + (itx - 1) * 4          # MATLAB bitget 位号（1 基）
            if (tmp_tx >> (bit_pos - 1)) & 1:
                return itx
        return 1

    @property
    def nvirtual_array(self) -> int:
        return self.nvirtual_chirp * 4

    @property
    def vel_chirp_num(self) -> int:
        """do_fft_2d.m 中参与 FFT 的 chirp 数 = nchirp / nvirtual_chirp。"""
        return self.nchirp // self.nvirtual_chirp

    @property
    def adc_sample_num(self) -> int:
        """官方取整分支 + rng_nfft 截断（本仓两配置 dec_factor=1，无 +1）。"""
        span = (self.adc_sample_end - self.adc_sample_start) * self.adc_freq
        if self.dec_factor > 1:
            n = int(matlab_round(span / self.dec_factor)) + 1
        else:
            n = int(matlab_round(span / self.dec_factor))
        return min(n, self.rng_nfft)

    @property
    def rng_bin_num(self) -> int:
        return self.rng_nfft // 2

    @property
    def rng_res(self) -> float:
        """官方 rng_res（m/bin）：3e8/2 / (BW/rampup * 采样窗宽) / 1e6。"""
        span = self.adc_sample_end - self.adc_sample_start
        return (C_LIGHT / 2.0) / (
            (self.fmcw_bandwidth / self.fmcw_chirp_rampup) * span
        ) / 1e6

    @property
    def fram_time(self) -> float:
        anti = 0 if not self.anti_velamb_en else 1
        return (
            self.fmcw_chirp_period * self.nchirp * 1e-3
            + anti
            * (self.fmcw_chirp_period + self.anti_velamb_delay)
            * self.nchirp
            / self.nvirtual_chirp
            * 1e-3
        )

    @property
    def v_res(self) -> float:
        """官方 V_res（m/s/bin）。"""
        return 0.5 * C_LIGHT / self.fmcw_startfreq / 1e9 * (
            1.0 / self.fram_time / 1e-3
        )

    @property
    def max_v(self) -> float:
        return self.v_res * self.nchirp / self.nvirtual_chirp

    def summary(self) -> Dict[str, float]:
        return {
            "f0_GHz": self.fmcw_startfreq,
            "BW_MHz": self.fmcw_bandwidth,
            "rampup_us": self.fmcw_chirp_rampup,
            "period_us": self.fmcw_chirp_period,
            "nchirp": self.nchirp,
            "adc_freq_MHz": self.adc_freq,
            "dec_factor": self.dec_factor,
            "adc_sample_window_us": (self.adc_sample_start, self.adc_sample_end),
            "adc_sample_num": self.adc_sample_num,
            "rng_nfft": self.rng_nfft,
            "vel_nfft": self.vel_nfft,
            "nvirtual_chirp": self.nvirtual_chirp,
            "nvirtual_array": self.nvirtual_array,
            "rng_res_m": self.rng_res,
            "v_res_mps": self.v_res,
        }


# ---------------------------------------------------------------------------
# 数据加载（load_adc_data.m + adc_mem_2_real_fixed.m）
# ---------------------------------------------------------------------------

def load_channel_raw(path: str) -> np.ndarray:
    """读取无头逗号分隔 uint32 文本为 1 维数组。"""
    vals = np.loadtxt(path, delimiter=",", dtype=np.float64).ravel()
    return vals.astype(np.uint32)


def unpack_words(words_chirp_major: np.ndarray) -> np.ndarray:
    """adc_mem_2_real_fixed.m：[nchirp, nword] uint32 -> [nchirp, 2*nword] int16。

    每个字：高 16 位 -> 前一个样本，低 16 位 -> 后一个样本（两补码）。
    """
    w = words_chirp_major.astype(np.uint32)
    hi = (w >> 16).astype(np.int16)
    lo = (w & 0xFFFF).astype(np.int16)
    out = np.empty((w.shape[0], w.shape[1] * 2), dtype=np.int16)
    out[:, 0::2] = hi
    out[:, 1::2] = lo
    return out


def load_channel(path: str, cfg: RadarConfig) -> Tuple[np.ndarray, int]:
    """加载单个通道，返回 ([rng_nfft, nchirp] int16, 实际读到的字数)。

    流程与 load_adc_data.m 一致：截断/补零到 expected = rng_nfft/2 * nchirp
    个字（保留**前** expected 个值），列优先 reshape 为 [rng_nfft/2, nchirp]
    （等价 MATLAB reshape），转置为 chirp 行，解包，再转置回 [sample, chirp]。
    """
    raw = load_channel_raw(path)
    actual = int(raw.size)
    expected = (cfg.rng_nfft // 2) * cfg.nchirp
    if actual > expected:
        raw = raw[:expected]
    elif actual < expected:
        raw = np.concatenate([raw, np.zeros(expected - actual, dtype=raw.dtype)])
    # MATLAB reshape(x, [rng_nfft/2, nchirp]) 为列优先
    words = raw.reshape((cfg.rng_nfft // 2, cfg.nchirp), order="F")
    chirp_major = unpack_words(words.T)            # [nchirp, rng_nfft]
    return chirp_major.T, actual                   # [rng_nfft, nchirp]


# ---------------------------------------------------------------------------
# 2D-FFT（do_fft_2d.m）
# ---------------------------------------------------------------------------

_WIN_MAP = {"hanning": 2, "cheb": 1, "hamming": 3}


def _window(name: str, n: int) -> np.ndarray:
    code = _WIN_MAP.get(name, 0)
    if code == 2:
        return hann(n, sym=True)      # MATLAB hann() 为对称窗
    if code == 3:
        from scipy.signal.windows import hamming
        return hamming(n, sym=True)
    if code == 1:
        from scipy.signal.windows import chebwin
        return chebwin(n, 80.0)       # 官方配置 rng_win_params[0]=80
    return np.ones(n)


def do_fft_2d(adc_data: np.ndarray, cfg: RadarConfig) -> Tuple[np.ndarray, np.ndarray]:
    """官方 2D-FFT。adc_data: [rng_nfft, nchirp, nch]。

    返回 (rng_fft [rng_nfft, vel_chirp_num, nch],
          vel_fft [rng_nfft/2, vel_nfft, nch])，均已除以官方 FFT scalar。
    """
    if adc_data.shape[0] < cfg.adc_sample_num:
        raise ValueError("adc_data 采样点数不足 adc_sample_num")
    rng_win = _window(cfg.rng_win, cfg.adc_sample_num)
    vel_win = _window(cfg.vel_win, cfg.vel_chirp_num)

    used = adc_data[: cfg.adc_sample_num, : cfg.vel_chirp_num, :]
    rng_fft_win = used * rng_win[:, None, None]
    rng_fft = np.fft.fft(rng_fft_win, n=cfg.rng_nfft, axis=0) / float(
        cfg.rng_fft_scalar
    )
    vel_fft_win = rng_fft[:, : cfg.vel_chirp_num, :] * vel_win[None, :, None]
    vel_fft = np.fft.fft(vel_fft_win, n=cfg.vel_nfft, axis=1) / float(
        cfg.vel_fft_scalar
    )
    return rng_fft, vel_fft[: cfg.rng_nfft // 2, :, :]


# ---------------------------------------------------------------------------
# 官方非相干积累（功率平均）
# ---------------------------------------------------------------------------

def channel_power(vel_fft: np.ndarray) -> np.ndarray:
    """power(|velFftData(:,:,ch)|, 2)：逐通道功率图，shape 不变。"""
    return np.abs(vel_fft) ** 2


def accumulate_power(power: np.ndarray, channels: Sequence[int] | None = None) -> np.ndarray:
    """官方“非相干积累”：对所选通道功率图求**平均**（sum / 通道数）。

    对应 ct_signal_processing_main_simple_CTASIA100.m::

        RVimage = RVimage + power(abs(velFftData(:,:,iArray)), 2);
        RVimage = RVimage / SensorConfig.nvirtual_array;
    """
    sel = power if channels is None else power[:, :, list(channels)]
    return sel.mean(axis=2)


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))


# ---------------------------------------------------------------------------
# 数据集发现
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, str]] = {
    "far": {
        "marker": "f0",
        "hxx": "sensor_config_init0.hxx",
        "label": "远波 f0",
        "jpg": "远波4个通道非相干积累.jpg",
    },
    "near": {
        "marker": "f1",
        "hxx": "sensor_config_init1.hxx",
        "label": "近波 f1",
        "jpg": "近波4个通道非相干积累.jpg",
    },
}


def find_channel_files(data_dir: str, marker: str) -> List[str]:
    """按 ``_<marker>_<k>_`` 的 k = RX 通道号排序返回 4 个文件路径。"""
    pat = re.compile(r"_" + re.escape(marker) + r"_(\d+)_")
    found = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".txt"):
            continue
        m = pat.search(name)
        if m:
            found[int(m.group(1))] = os.path.join(data_dir, name)
    if sorted(found) != [0, 1, 2, 3]:
        raise FileNotFoundError(
            f"数据目录 {data_dir} 未找到 marker={marker} 的 RX0..3 四个文件，"
            f"实际通道索引: {sorted(found)}"
        )
    return [found[k] for k in range(4)]


def load_dataset(data_dir: str, cfg: RadarConfig, marker: str) -> Dict[str, object]:
    """加载一个数据集（4 通道）并跑官方 2D-FFT。"""
    files = find_channel_files(data_dir, marker)
    mats = []
    words = []
    for f in files:
        mat, n_words = load_channel(f, cfg)
        mats.append(mat.astype(np.float64))
        words.append(n_words)
    adc = np.stack(mats, axis=2)
    rng_fft, vel_fft = do_fft_2d(adc, cfg)
    power = channel_power(vel_fft)
    return {
        "files": files,
        "word_counts": words,
        "adc": adc,
        "rng_fft": rng_fft,
        "vel_fft": vel_fft,
        "power": power,                        # [rng_bin_num, vel_nfft, 4]
        "rv_image": accumulate_power(power),   # 官方 4 通道功率平均
    }


def repo_root_from_module() -> str:
    """从本文件位置推导仓库根（src/rx_ablation 上溯 5 级）。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        d = os.path.dirname(d)
    return d


DEFAULT_DATA_DIR = os.path.join("ADC数据采集", "示例adc数据和结果")
DEFAULT_CFG_DIR = os.path.join(
    "ADC数据采集",
    "matlab_signal_processing_platform_231023_for_txt_A100",
    "cfg",
    "CTASI-A100配置",
)
