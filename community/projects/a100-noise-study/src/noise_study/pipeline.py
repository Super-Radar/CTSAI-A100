"""Sensor configuration parsing and the ADC -> 2D-FFT processing chain.

This module is a line-faithful Python re-implementation of the official
MATLAB signal-processing platform shipped in the repository at
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/``:

==============================  ===========================================
MATLAB source (read-only ref.)   Python counterpart here
==============================  ===========================================
``init_function/get_radar_paramete.m``   :func:`parse_sensor_config` / :class:`SensorConfig`
``init_function/load_adc_data.m``        :func:`load_adc_data`
``init_function/adc_mem_2_real_fixed.m`` :func:`adc_mem_2_real_fixed`
``FFT/do_fft_2d.m``                      :func:`do_fft_2d`
==============================  ===========================================

Deliberate deviations from a naive transcription (all verified against the
MATLAB sources and documented in the project README):

1. ``adc_sample_num`` follows the *exact* branch structure of
   ``get_radar_paramete.m``: with ``dec_factor == 1`` the value is
   ``round((adc_sample_end - adc_sample_start) * adc_freq)`` **without** the
   ``+1`` (the ``+1`` only applies when ``dec_factor > 1``), afterwards
   clamped to ``rng_nfft``.  For ``sensor_config_init0.hxx`` this yields
   ``round((42.8 - 2) * 25) = 1020`` (not 1021); for
   ``sensor_config_init1.hxx`` it yields ``round((104.4 - 2) * 25) = 2560``
   which is then clamped to ``rng_nfft = 2048``.
2. MATLAB ``reshape(col, [rng_nfft/2, nchirp])`` is column-major; combined
   with the subsequent transpose this equals a C-order ``reshape`` to
   ``(nchirp, rng_nfft/2)`` in NumPy.
3. MATLAB ``hann(N)`` (symmetric) equals ``numpy.hanning``.
4. The FFTs are divided by ``rng_fft_scalar`` / ``vel_fft_scalar``
   (``0xff`` = 255 and ``0x3f`` = 63 for both public profiles), exactly as
   ``do_fft_2d.m`` does.  These scalars shift the absolute dB level of the
   RV image, so they must be kept to reproduce official noise values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository default paths (the worktree / main repo both contain the data).
# ---------------------------------------------------------------------------

REPO_SUBDIR_DATA = Path("ADC数据采集") / "示例adc数据和结果"
REPO_SUBDIR_HXX = (
    Path("ADC数据采集")
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "cfg"
    / "CTASI-A100配置"
)


def repo_root() -> Path:
    """Return the git repository root that contains this project.

    Layout: <repo>/community/projects/a100-noise-study/src/noise_study/pipeline.py
    """
    return Path(__file__).resolve().parents[5]


def default_data_dir() -> Path:
    return repo_root() / REPO_SUBDIR_DATA


def default_hxx_dir() -> Path:
    return repo_root() / REPO_SUBDIR_HXX


def channel_files(data_dir: Path, kind: str) -> List[Path]:
    """Return the four per-RX-channel ADC files in channel order 0..3.

    ``kind`` is ``"far"`` (远波, profile init0) or ``"near"`` (近波, init1).
    """
    if kind not in ("far", "near"):
        raise ValueError("kind must be 'far' or 'near'")
    prefix = "adc_远波_" if kind == "far" else "adc_近波_"
    tag = "f0" if kind == "far" else "f1"
    files: List[Path] = []
    for ch in range(4):
        matches = sorted((Path(data_dir)).glob(f"{prefix}*_{tag}_{ch}_*.txt"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one ADC file for kind={kind} channel {ch}, "
                f"found {len(matches)} under {data_dir}"
            )
        files.append(matches[0])
    return files


# ---------------------------------------------------------------------------
# MATLAB-compatible helpers
# ---------------------------------------------------------------------------


def matlab_round(x: float) -> int:
    """MATLAB ``round``: round half away from zero (unlike Python's banker's)."""
    xf = float(x)
    if xf >= 0:
        return int(np.floor(xf + 0.5))
    return -int(np.floor(-xf + 0.5))


def hann_window(n: int) -> np.ndarray:
    """Symmetric Hann window, identical to MATLAB ``hann(n)`` / ``np.hanning``."""
    return np.hanning(n)


def hamming_window(n: int) -> np.ndarray:
    """Symmetric Hamming window, identical to MATLAB ``hamming(n)``."""
    return np.hamming(n)


def make_window(code: int, n: int) -> np.ndarray:
    """Window selection mirroring the switch block of ``do_fft_2d.m``."""
    if code == 1:
        raise NotImplementedError(
            "chebwin (Cfg.rng_win/vel_win == 1) is not used by the public "
            "A100 profiles and is not re-implemented here"
        )
    if code == 2:
        return hann_window(n)
    if code == 3:
        return hamming_window(n)
    return np.ones(n)


# ---------------------------------------------------------------------------
# .hxx configuration parsing (subset of get_radar_paramete.m)
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

_WIN_STRING_TO_CODE = {"cheb": 1, "hanning": 2, "hamming": 3}


def _parse_scalar(token: str):
    token = token.strip()
    if token.startswith(("\"", "'")) and token.endswith(("\"", "'")):
        return token[1:-1]
    if token == "true":
        return True
    if token == "false":
        return False
    if token.startswith("0x") or token.startswith("0X"):
        return int(token, 16)
    try:
        return int(token)
    except ValueError:
        return float(token)


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1]
        items = [t for t in (s.strip() for s in inner.split(",")) if t != ""]
        return [_parse_scalar(t) for t in items]
    return _parse_scalar(raw)


@dataclass
class SensorConfig:
    """Fields mirroring the MATLAB ``Cfg`` struct used by the platform."""

    # raw fields from the .hxx file
    fmcw_startfreq: float
    fmcw_bandwidth: float
    fmcw_chirp_rampup: float
    fmcw_chirp_period: float
    tx_groups: List[int]
    nchirp: int
    adc_freq: float
    dec_factor: int
    adc_sample_start: float
    adc_sample_end: float
    rng_win: int  # code: 1 cheb / 2 hann / 3 hamming / 0 rect
    vel_win: int
    rng_nfft: int
    vel_nfft: int
    rng_fft_scalar: int
    vel_fft_scalar: int
    # derived fields (get_radar_paramete.m "Calculation of radar parameters")
    nvirtual_chirp: int = field(init=False)
    nvirtual_array: int = field(init=False)
    adc_sample_num: int = field(init=False)
    rng_bin_num: int = field(init=False)

    def __post_init__(self) -> None:
        # --- TX count: identical loop to get_radar_paramete.m ---
        tmp_tx = int(max(self.tx_groups))
        n_tx = 1
        for i_tx in range(4, 0, -1):
            if (tmp_tx >> (1 + (i_tx - 1) * 4)) & 1:
                n_tx = i_tx
                break
        self.nvirtual_chirp = n_tx
        self.nvirtual_array = self.nvirtual_chirp * 4

        # --- adc_sample_num: exact branch structure of get_radar_paramete.m ---
        span = (self.adc_sample_end - self.adc_sample_start) * self.adc_freq
        if self.dec_factor > 1:
            n_samples = matlab_round(span / self.dec_factor) + 1
        else:
            n_samples = matlab_round(span / self.dec_factor)
        if n_samples > self.rng_nfft:
            n_samples = self.rng_nfft
        self.adc_sample_num = n_samples
        self.rng_bin_num = self.rng_nfft // 2

    # -- convenience ------------------------------------------------------
    @property
    def rng_res_m(self) -> float:
        """Range resolution [m], same expression as get_radar_paramete.m."""
        return (
            3e8
            / 2
            / (self.fmcw_bandwidth / self.fmcw_chirp_rampup * (self.adc_sample_end - self.adc_sample_start))
            / 1e6
        )

    def summary(self) -> Dict[str, object]:
        return {
            "fmcw_startfreq_GHz": self.fmcw_startfreq,
            "fmcw_bandwidth_MHz": self.fmcw_bandwidth,
            "fmcw_chirp_rampup_us": self.fmcw_chirp_rampup,
            "fmcw_chirp_period_us": self.fmcw_chirp_period,
            "nchirp": self.nchirp,
            "adc_freq_MHz": self.adc_freq,
            "dec_factor": self.dec_factor,
            "adc_sample_start_us": self.adc_sample_start,
            "adc_sample_end_us": self.adc_sample_end,
            "adc_sample_num": self.adc_sample_num,
            "rng_nfft": self.rng_nfft,
            "vel_nfft": self.vel_nfft,
            "rng_fft_scalar": self.rng_fft_scalar,
            "vel_fft_scalar": self.vel_fft_scalar,
            "rng_win": self.rng_win,
            "vel_win": self.vel_win,
            "nvirtual_chirp": self.nvirtual_chirp,
            "nvirtual_array": self.nvirtual_array,
        }


_CFG_LINE_RE = re.compile(r"^\s*\.\s*(\w+)\s*=\s*(.+?),\s*$", re.MULTILINE)

# fields we need from the .hxx (the MATLAB parser reads far more)
_REQUIRED_FIELDS = (
    "fmcw_startfreq",
    "fmcw_bandwidth",
    "fmcw_chirp_rampup",
    "fmcw_chirp_period",
    "tx_groups",
    "nchirp",
    "adc_freq",
    "dec_factor",
    "adc_sample_start",
    "adc_sample_end",
    "rng_win",
    "vel_win",
    "rng_nfft",
    "vel_nfft",
    "rng_fft_scalar",
    "vel_fft_scalar",
)


def parse_sensor_config(hxx_path: Path | str) -> SensorConfig:
    """Parse the needed subset of a ``sensor_config_initN.hxx`` file.

    Mirrors ``get_radar_paramete.m``: hex literals are converted to decimal,
    window strings are mapped to numeric codes, ``tx_groups`` feeds the
    nvirtual_chirp loop, and the derived parameters follow the same formulas
    (including the ``dec_factor`` branch and the ``rng_nfft`` clamp).
    """
    text = Path(hxx_path).read_text(encoding="utf-8", errors="replace")
    parsed: Dict[str, object] = {}
    for match in _CFG_LINE_RE.finditer(text):
        name, raw_value = match.group(1), match.group(2)
        try:
            value = _parse_value(raw_value)
        except ValueError:
            # nested-brace initializer lists (e.g. .ant_pos = {{...}, ...})
            # are not needed by this study -> skip gracefully
            continue
        if name in ("rng_win", "vel_win", "cfar_mimo_win", "doa_win") and isinstance(value, str):
            value = _WIN_STRING_TO_CODE.get(value, 0)
        parsed[name] = value
    missing = [f for f in _REQUIRED_FIELDS if f not in parsed]
    if missing:
        raise ValueError(f"{hxx_path}: missing required fields: {missing}")
    kwargs = {f: parsed[f] for f in _REQUIRED_FIELDS}
    return SensorConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ADC loading / unpacking
# ---------------------------------------------------------------------------


def adc_mem_2_real_fixed(words: np.ndarray) -> np.ndarray:
    """uint32 words -> interleaved signed int16 samples (adc_mem_2_real_fixed.m).

    ``words`` has shape ``(..., n_words)``; the high 16 bits of each word
    become the earlier sample, the low 16 bits the later sample, both
    re-interpreted as two's-complement signed values.  Output shape is
    ``(..., 2 * n_words)``.
    """
    u = np.asarray(words, dtype=np.uint32)
    high = u >> 16  # floor(raw / 2^16)
    low = u & 0xFFFF  # mod(raw, 2^16)
    high_signed = np.where(high >= 0x8000, high.astype(np.int64) - 0x10000, high.astype(np.int64))
    low_signed = np.where(low >= 0x8000, low.astype(np.int64) - 0x10000, low.astype(np.int64))
    out = np.empty(u.shape[:-1] + (2 * u.shape[-1],), dtype=np.int16)
    out[..., 0::2] = high_signed
    out[..., 1::2] = low_signed
    return out


@dataclass
class ChannelLoadReport:
    """Explicit record of the length handling performed by load_adc_data."""

    file: str
    n_values: int
    expected_len: int
    action: str  # "truncate" | "pad" | "exact"
    dropped_tail_values: List[int] = field(default_factory=list)
    padded_zeros: int = 0


def _read_uint32_csv(path: Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    tokens = text.strip().split(",")
    return np.array(tokens, dtype=np.int64).astype(np.uint32)


def load_adc_data(files: Sequence[Path], cfg: SensorConfig):
    """Load per-channel uint32 ADC dumps and build the sample-major matrix.

    Mirrors ``load_adc_data.m`` for the ``nvirtual_chirp == 1`` case used by
    both public A100 profiles (1TX4RX):

    1. each file is read as a flat comma-separated uint32 stream;
    2. the stream is aligned to ``rng_nfft/2 * nchirp`` values by truncation
       or zero padding (the public files carry one extra trailing value,
       see :class:`ChannelLoadReport`);
    3. the MATLAB column-major ``reshape(col, [rng_nfft/2, nchirp])`` followed
       by a transpose equals a C-order reshape to ``(nchirp, rng_nfft/2)``;
    4. ``adc_mem_2_real_fixed`` unpacks word k into samples 2k-1 / 2k
       (1-based), giving ``(nchirp, rng_nfft)``;
    5. the final ``permute(..., [2, 1])`` gives ``adcData[sample, chirp, ch]``.

    Returns ``(adcData, reports)`` with ``adcData`` of shape
    ``(rng_nfft, nchirp // nvirtual_chirp, nvirtual_array)`` (int16).
    """
    if cfg.nvirtual_chirp != 1:
        raise NotImplementedError(
            "only the nvirtual_chirp == 1 (1TX) layout of the public A100 "
            "profiles is supported; load_adc_data.m also has a DDMA branch"
        )
    n_words_rng = cfg.rng_nfft // 2
    nchirp_eff = cfg.nchirp // cfg.nvirtual_chirp
    expected_len = n_words_rng * nchirp_eff

    adc_data = np.empty((cfg.rng_nfft, nchirp_eff, cfg.nvirtual_array), dtype=np.int16)
    reports: List[ChannelLoadReport] = []
    for ch, path in enumerate(files):
        values = _read_uint32_csv(path)
        n_values = int(values.size)
        if n_values > expected_len:
            dropped = [int(v) for v in values[expected_len:]]
            values = values[:expected_len]
            action = "truncate"
        elif n_values < expected_len:
            dropped = []
            values = np.concatenate([values, np.zeros(expected_len - n_values, dtype=np.uint32)])
            action = "pad"
        else:
            dropped = []
            action = "exact"
        reports.append(
            ChannelLoadReport(
                file=Path(path).name,
                n_values=n_values,
                expected_len=expected_len,
                action=action,
                dropped_tail_values=dropped,
                padded_zeros=expected_len - n_values if action == "pad" else 0,
            )
        )
        # step 3: == MATLAB reshape(col, [rng_nfft/2, nchirp])'
        chirp_major_words = values.reshape(nchirp_eff, n_words_rng)
        # steps 4-5: unpack then transpose -> [sample, chirp]
        adc_data[:, :, ch] = adc_mem_2_real_fixed(chirp_major_words).T
    return adc_data, reports


# ---------------------------------------------------------------------------
# 2D FFT chain (do_fft_2d.m)
# ---------------------------------------------------------------------------


def do_fft_2d(adc_data: np.ndarray, cfg: SensorConfig):
    """Hann-windowed Range FFT + Doppler FFT, mirroring ``do_fft_2d.m``.

    ``adc_data`` has shape ``(rng_nfft, nchirp_eff, nvirtual_array)``.

    Returns ``(rngFftData, velFftData)``:

    * ``rngFftData``: ``(rng_nfft, nchirp_eff, nvirtual_array)`` complex --
      the Range FFT output (divided by ``rng_fft_scalar``);
    * ``velFftData``: ``(rng_nfft/2, vel_nfft, nvirtual_array)`` complex --
      first half of range bins after the Doppler FFT (divided by
      ``vel_fft_scalar``), i.e. the ``Radar_cube`` consumed by
      ``calculate_global_noise.m``.
    """
    nchirp_eff = cfg.nchirp // cfg.nvirtual_chirp
    if adc_data.shape != (cfg.rng_nfft, nchirp_eff, cfg.nvirtual_array):
        raise ValueError(
            f"adc_data shape {adc_data.shape} does not match cfg "
            f"({cfg.rng_nfft}, {nchirp_eff}, {cfg.nvirtual_array})"
        )
    rng_win = make_window(cfg.rng_win, cfg.adc_sample_num)
    vel_win = make_window(cfg.vel_win, nchirp_eff)

    rng_fft = np.empty((cfg.rng_nfft, nchirp_eff, cfg.nvirtual_array), dtype=np.complex128)
    vel_fft = np.empty((cfg.rng_nfft // 2, cfg.vel_nfft, cfg.nvirtual_array), dtype=np.complex128)
    for ch in range(cfg.nvirtual_array):
        # Range windowing on the first adc_sample_num samples (zero pad to
        # rng_nfft happens inside fft(..., n=rng_nfft)), then / rng_fft_scalar.
        windowed = adc_data[: cfg.adc_sample_num, :, ch].astype(np.float64) * rng_win[:, None]
        rng_fft[:, :, ch] = np.fft.fft(windowed, n=cfg.rng_nfft, axis=0) / cfg.rng_fft_scalar
        # Velocity windowing along chirps, Doppler FFT to vel_nfft
        # (nchirp_eff == vel_nfft for both public profiles), / vel_fft_scalar.
        vel_win_mat = rng_fft[:, :nchirp_eff, ch] * vel_win[None, :]
        vel_full = np.fft.fft(vel_win_mat, n=cfg.vel_nfft, axis=1) / cfg.vel_fft_scalar
        vel_fft[:, :, ch] = vel_full[: cfg.rng_nfft // 2, :]
    return rng_fft, vel_fft


def rv_power(vel_fft: np.ndarray) -> np.ndarray:
    """Channel-averaged power image, i.e. ``sum(power)/nvirtual_array``."""
    return np.mean(np.abs(vel_fft) ** 2, axis=2)


def power_to_db(p: np.ndarray) -> np.ndarray:
    """MATLAB ``db(x, 'power')`` == ``10*log10(x)`` for power quantities."""
    return 10.0 * np.log10(p)
