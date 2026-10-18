"""Loading, unpacking, MATLAB-compatible reshaping and range FFT for A100 ADC dumps.

This module reproduces, in NumPy, the data path of the official MATLAB
reference platform (read-only reference inside the repository:
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/``):

1. ``init_function/load_adc_data.m``
   reads a comma-separated uint32 dump, flattens it, forces the length to
   ``rng_nfft/2 * nchirp`` (the sample files carry one superfluous trailing 0
   which the reference code drops with a warning -- we log it explicitly and
   truncate), reshapes column-major to ``[rng_nfft/2, nchirp]`` and transposes
   to ``[chirp, packed words]``;
2. ``init_function/adc_mem_2_real_fixed.m``
   splits every uint32 word into two signed int16 samples -- the *earlier*
   sample is the high 16 bits, the *later* sample the low 16 bits
   (``ADC[.., sample-1] = H, ADC[.., sample] = L``);
3. ``FFT/do_fft_2d.m``
   multiplies the first ``adc_sample_num`` samples of each chirp by a window of
   exactly that length (MATLAB ``hann`` default = symmetric), FFTs with length
   ``rng_nfft`` (implicit zero fill of the few samples beyond the ADC window)
   and keeps the first ``rng_nfft/2`` bins.  The official code additionally
   divides by ``rng_fft_scalar = 0xff`` (255); see ``OFFICIAL_RNG_FFT_SCALAR``.

``adc_sample_num`` is computed with inclusive endpoints,
``round((t_end - t_start) * f_adc) + 1`` (init0: round(40.8 us * 25 MHz) + 1 =
1021), and clamped to ``rng_nfft`` exactly as ``get_radar_paramete.m`` does.
Note the official MATLAB itself uses ``round(...)`` *without* ``+1`` when
``dec_factor == 1`` (i.e. 1020 for init0); the one-sample difference is
documented in the project README and only shifts which trailing samples are
windowed, never the conclusions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .windows import make_window

logger = logging.getLogger("window_study")

C0 = 299792458.0

#: divisor applied to the range FFT by the official MATLAB chain (``.rng_fft_scalar = 0xff``)
OFFICIAL_RNG_FFT_SCALAR = 255.0


@dataclass(frozen=True)
class RadarConfig:
    """Static parameters of one A100 measurement profile (from sensor_config_initN.hxx)."""

    key: str
    label: str
    file_prefix: str  # 远波 / 近波
    file_tag: str  # f0 / f1
    start_freq_ghz: float
    bandwidth_mhz: float
    ramp_us: float
    chirp_down_us: float
    period_us: float
    nchirp: int
    adc_freq_mhz: float
    sample_start_us: float
    sample_end_us: float
    rng_nfft: int
    vel_nfft: int
    rng_win: str = "hann"

    # --- derived quantities -------------------------------------------------
    @property
    def adc_freq_hz(self) -> float:
        return self.adc_freq_mhz * 1e6

    @property
    def slope_hz_per_s(self) -> float:
        return self.bandwidth_mhz * 1e6 / (self.ramp_us * 1e-6)

    @property
    def words_per_chirp(self) -> int:
        return self.rng_nfft // 2

    @property
    def samples_per_chirp(self) -> int:
        return self.rng_nfft

    @property
    def expected_words(self) -> int:
        return self.words_per_chirp * self.nchirp

    @property
    def adc_sample_num(self) -> int:
        """Number of valid samples per chirp (inclusive endpoints, clamped to rng_nfft)."""
        n = int(round((self.sample_end_us - self.sample_start_us) * self.adc_freq_mhz)) + 1
        return min(n, self.rng_nfft)

    @property
    def zero_fill(self) -> int:
        """Samples of the FFT that are zero-filled beyond the ADC window."""
        return self.rng_nfft - self.adc_sample_num

    @property
    def nominal_range_res_m(self) -> float:
        """Nominal resolution c / (2B) (init0: 0.5 m for 300 MHz)."""
        return C0 / (2.0 * self.bandwidth_mhz * 1e6)

    @property
    def bin_length_m(self) -> float:
        """Range step per FFT bin: R(k) = k * c * f_adc / (2 * slope * rng_nfft)."""
        return C0 * self.adc_freq_hz / (2.0 * self.slope_hz_per_s * self.rng_nfft)

    @property
    def official_rng_res_m(self) -> float:
        """rng_res as computed by get_radar_paramete.m: c/2 / (slope * sample window)."""
        return C0 / 2.0 / (self.slope_hz_per_s * (self.sample_end_us - self.sample_start_us) * 1e-6)

    def glob_pattern(self, channel: int) -> str:
        return f"adc_{self.file_prefix}_*_{self.file_tag}_{channel}_*.txt"


#: 远波 f0 (init0)
FAR_F0 = RadarConfig(
    key="far_f0",
    label="far wave f0 (init0)",
    file_prefix="远波",
    file_tag="f0",
    start_freq_ghz=76.3,
    bandwidth_mhz=300.0,
    ramp_us=43.0,
    chirp_down_us=3.0,
    period_us=48.0,
    nchirp=256,
    adc_freq_mhz=25.0,
    sample_start_us=2.0,
    sample_end_us=42.8,
    rng_nfft=1024,
    vel_nfft=256,
    rng_win="hann",
)

#: 近波 f1 (init1)
NEAR_F1 = RadarConfig(
    key="near_f1",
    label="near wave f1 (init1)",
    file_prefix="近波",
    file_tag="f1",
    start_freq_ghz=76.119,
    bandwidth_mhz=750.0,
    ramp_us=104.65,
    chirp_down_us=2.5,
    period_us=108.5,
    nchirp=128,
    adc_freq_mhz=25.0,
    sample_start_us=2.0,
    sample_end_us=104.4,
    rng_nfft=2048,
    vel_nfft=128,
    rng_win="hann",
)

CONFIGS = {"far_f0": FAR_F0, "near_f1": NEAR_F1}


def find_data_file(data_dir, cfg: RadarConfig, channel: int) -> Path:
    """Locate the sample dump of one receive channel inside ``data_dir``."""
    data_dir = Path(data_dir)
    matches = sorted(data_dir.glob(cfg.glob_pattern(channel)))
    if not matches:
        raise FileNotFoundError(
            f"no file matching {cfg.glob_pattern(channel)!r} in {data_dir}"
        )
    if len(matches) > 1:
        logger.warning("multiple matches for %s, using %s", cfg.glob_pattern(channel), matches[0].name)
    return matches[0]


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_words(path, expected_len: int):
    """Read one uint32 dump (comma and/or whitespace separated) and force its length.

    Policy for superfluous leading/trailing values: the payload length is fully
    determined by the configuration (``expected_len``); everything beyond it is
    *explicitly recorded and then truncated*, mirroring the reference MATLAB
    behaviour (``warning(...) + adcTempCol(1:expected_len)``).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    vals = np.array(tokens, dtype=np.int64)
    if vals.size < expected_len:
        raise ValueError(
            f"{path.name}: {vals.size} values < expected payload {expected_len}"
        )
    if vals.min() < 0 or vals.max() > 0xFFFFFFFF:
        raise ValueError(f"{path.name}: values outside uint32 range")
    dropped = vals[expected_len:]
    info = {
        "file": path.name,
        "values_read": int(vals.size),
        "payload_len": int(expected_len),
        "dropped_tail": dropped.tolist(),
        "trailing_zeros_in_payload": 0,
    }
    if dropped.size:
        payload = vals[:expected_len]
        info["trailing_zeros_in_payload"] = int(np.sum(payload[-8:] == 0))
        logger.info(
            "%s: read %d values, payload %d -> dropped %d trailing value(s) %s "
            "(payload tail zeros in last 8 words: %d)",
            path.name,
            vals.size,
            expected_len,
            dropped.size,
            dropped.tolist()[:4],
            info["trailing_zeros_in_payload"],
        )
    return vals[:expected_len].astype(np.uint32), info


def reshape_words_column_major(payload: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """MATLAB ``reshape(payload, [rng_nfft/2, nchirp])`` (column-major) transposed
    to ``[chirp, packed words]``."""
    m = np.reshape(payload, (cfg.words_per_chirp, cfg.nchirp), order="F")
    return m.T


def _signed16(u16: np.ndarray) -> np.ndarray:
    """uint16 -> signed value (two's complement) as float."""
    return np.where(u16 >= 0x8000, u16.astype(np.int32) - 0x10000, u16.astype(np.int32)).astype(
        np.float64
    )


def unpack_words(words: np.ndarray) -> np.ndarray:
    """``[chirp, packed]`` uint32 -> ``[chirp, 2*packed]`` float samples.

    Matches ``adc_mem_2_real_fixed.m``: the earlier sample of each pair is the
    *high* 16 bits, the later sample the *low* 16 bits.
    """
    u = np.asarray(words, dtype=np.uint32)
    hi = _signed16((u >> 16) & 0xFFFF)
    lo = _signed16(u & 0xFFFF)
    out = np.empty(u.shape[:-1] + (2 * u.shape[-1],), dtype=np.float64)
    out[..., 0::2] = hi
    out[..., 1::2] = lo
    return out


def load_channel(path, cfg: RadarConfig):
    """Full chain file -> ``[nchirp, rng_nfft]`` real samples plus loading info."""
    payload, info = load_raw_words(path, cfg.expected_words)
    words = reshape_words_column_major(payload, cfg)
    adc = unpack_words(words)
    return adc, info


def range_fft(
    chirp_samples,
    cfg: RadarConfig,
    window_name: str | None = None,
    valid_n: int | None = None,
    sym: bool = True,
    divide_by: float | None = None,
):
    """Windowed range FFT of a single chirp; returns ``(spectrum, window)``.

    Window length = number of valid samples (zero-filled to ``rng_nfft`` by the
    FFT), symmetric shape by default to match the official MATLAB chain.  Only
    the first ``rng_nfft/2`` bins are returned, as in ``do_fft_2d.m``.
    """
    window_name = cfg.rng_win if window_name is None else window_name
    valid_n = cfg.adc_sample_num if valid_n is None else valid_n
    x = np.asarray(chirp_samples)
    if not np.iscomplexobj(x):
        x = x.astype(float)
    x = x[:valid_n]
    w = make_window(window_name, len(x), sym=sym)
    X = np.fft.fft(x * w, cfg.rng_nfft)[: cfg.rng_nfft // 2]
    if divide_by is not None:
        X = X / float(divide_by)
    return X, w


def range_axis(cfg: RadarConfig, n: int | None = None) -> np.ndarray:
    """Range (m) of the first ``n`` bins."""
    n = cfg.rng_nfft // 2 if n is None else n
    return np.arange(n) * cfg.bin_length_m
