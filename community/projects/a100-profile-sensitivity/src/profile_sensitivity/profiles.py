"""Sensor profile parsing and physical-parameter derivation.

This module parses the two official CTSAI-A100 ``.hxx`` sensor configurations
(``sensor_config_init0.hxx`` far wave / ``sensor_config_init1.hxx`` near wave)
and derives every physical quantity exactly the way the official MATLAB
platform does in ``init_function/get_radar_paramete.m``.

Every derived property exists twice:

* ``<name>``        -- a clean SI reformulation used by the rest of the code;
* ``ml_<name>()``   -- a literal character-by-character transcription of the
  MATLAB expression (including MATLAB's left-to-right ``a / b / c`` chaining).

``tests/test_profiles.py`` asserts that both versions agree to floating point
round-off and that both match hand-computed reference numbers, so the parity
claim is executable, not rhetorical.

Units
-----
``fmcw_startfreq`` GHz, ``fmcw_bandwidth`` MHz, ``fmcw_chirp_rampup`` /
``fmcw_chirp_period`` / ``adc_sample_start`` / ``adc_sample_end`` /
``anti_velamb_delay`` us, ``adc_freq`` MHz.  Derived quantities are SI unless
the name says otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Speed of light used by ``get_radar_paramete.m`` (3e8, not 299792458.0).
C_LIGHT = 3.0e8

PROJECT_DIR = Path(__file__).resolve().parents[2]  # .../a100-profile-sensitivity
REPO_ROOT = PROJECT_DIR.parents[2]                 # repository root

DEFAULT_HXX_DIR = (
    REPO_ROOT / "ADC数据采集"
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "cfg" / "CTASI-A100配置"
)
DEFAULT_DATA_DIR = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果"

_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _read_scalar(text: str, name: str) -> float:
    m = re.search(r"\.%s\s*=\s*(%s)" % (re.escape(name), _FLOAT), text)
    if m is None:
        raise ValueError(f"hxx field .{name} not found")
    return float(m.group(1))


def _read_int(text: str, name: str) -> int:
    return int(round(_read_scalar(text, name)))


def _read_bool(text: str, name: str) -> bool:
    m = re.search(r"\.%s\s*=\s*(true|false)" % re.escape(name), text)
    if m is None:
        raise ValueError(f"hxx field .{name} (bool) not found")
    return m.group(1) == "true"


def _read_hex_list(text: str, name: str) -> tuple[int, ...]:
    m = re.search(r"\.%s\s*=\s*\{([^}]*)\}" % re.escape(name), text)
    if m is None:
        raise ValueError(f"hxx field .{name} (hex list) not found")
    return tuple(int(tok.strip(), 16) for tok in m.group(1).split(",") if tok.strip())


def _read_string(text: str, name: str) -> str:
    m = re.search(r'\.%s\s*=\s*"([^"]*)"' % re.escape(name), text)
    if m is None:
        raise ValueError(f'hxx field .{name} (string) not found')
    return m.group(1)


@dataclass(frozen=True)
class Profile:
    """One sensor configuration plus its derived physical quantities."""

    name: str                 # 'init0' | 'init1'
    label: str                # human readable, e.g. 'far wave (f0)'
    fmcw_startfreq: float     # GHz
    fmcw_bandwidth: float     # MHz
    fmcw_chirp_rampup: float  # us
    fmcw_chirp_period: float  # us
    nchirp: int
    adc_freq: float           # MHz (complex sample rate)
    dec_factor: int
    adc_sample_start: float   # us
    adc_sample_end: float     # us
    rng_nfft: int
    vel_nfft: int
    anti_velamb_en: bool
    anti_velamb_delay: float  # us
    tx_groups: tuple[int, ...]
    rng_win: str
    vel_win: str
    source: Path = field(default_factory=lambda: Path())

    # ------------------------------------------------------------------ parse
    @classmethod
    def from_hxx(cls, path: Path, name: str, label: str) -> "Profile":
        text = path.read_text(encoding="utf-8", errors="replace")
        return cls(
            name=name,
            label=label,
            fmcw_startfreq=_read_scalar(text, "fmcw_startfreq"),
            fmcw_bandwidth=_read_scalar(text, "fmcw_bandwidth"),
            fmcw_chirp_rampup=_read_scalar(text, "fmcw_chirp_rampup"),
            fmcw_chirp_period=_read_scalar(text, "fmcw_chirp_period"),
            nchirp=_read_int(text, "nchirp"),
            adc_freq=_read_scalar(text, "adc_freq"),
            dec_factor=_read_int(text, "dec_factor"),
            adc_sample_start=_read_scalar(text, "adc_sample_start"),
            adc_sample_end=_read_scalar(text, "adc_sample_end"),
            rng_nfft=_read_int(text, "rng_nfft"),
            vel_nfft=_read_int(text, "vel_nfft"),
            anti_velamb_en=_read_bool(text, "anti_velamb_en"),
            anti_velamb_delay=_read_scalar(text, "anti_velamb_delay"),
            tx_groups=_read_hex_list(text, "tx_groups"),
            rng_win=_read_string(text, "rng_win"),
            vel_win=_read_string(text, "vel_win"),
            source=path,
        )

    # ------------------------------------------- derived: MATLAB parity ----
    # The following ``ml_*`` methods transcribe get_radar_paramete.m verbatim
    # (MATLAB evaluates a/b/c left-to-right).  They exist so tests can prove
    # the SI properties below are the same formulas.

    def ml_adc_sample_num(self) -> int:
        if self.dec_factor > 1:
            n = round((self.adc_sample_end - self.adc_sample_start)
                      * self.adc_freq / self.dec_factor) + 1
        else:
            n = round((self.adc_sample_end - self.adc_sample_start)
                      * self.adc_freq / self.dec_factor)
        return int(min(n, self.rng_nfft))

    def ml_max_dis(self) -> float:
        return (3e8 / 2
                * (self.adc_freq * 1e6 / 2)
                / (self.fmcw_bandwidth * 1e6 / self.fmcw_chirp_rampup * 1e6)
                / self.dec_factor)

    def ml_rng_res(self) -> float:
        return (3e8 / 2
                / (self.fmcw_bandwidth / self.fmcw_chirp_rampup
                   * (self.adc_sample_end - self.adc_sample_start))
                / 1e6)

    def ml_fram_time_ms(self) -> float:
        ft = self.fmcw_chirp_period * self.nchirp * 1e-3
        ft += (self.anti_nchirp
               * (self.fmcw_chirp_period + self.anti_velamb_delay)
               * self.nchirp / self.nvirtual_chirp * 1e-3)
        return ft

    def ml_max_v(self) -> float:
        ft = self.ml_fram_time_ms()
        return (0.5 * 3e8 / self.fmcw_startfreq / 1e9
                * (1 / ft / 1e-3) * self.nchirp / self.nvirtual_chirp)

    def ml_v_res(self) -> float:
        ft = self.ml_fram_time_ms()
        return 0.5 * 3e8 / self.fmcw_startfreq / 1e9 * (1 / ft / 1e-3)

    # -------------------------------------------------- derived: SI form ----
    @property
    def fs_hz(self) -> float:
        """Complex ADC sample rate [Hz]."""
        return self.adc_freq * 1e6

    @property
    def ramp_time_s(self) -> float:
        return self.fmcw_chirp_rampup * 1e-6

    @property
    def chirp_period_s(self) -> float:
        return self.fmcw_chirp_period * 1e-6

    @property
    def window_s(self) -> float:
        """Duration of the sampling window (adc_sample_start..adc_sample_end)."""
        return (self.adc_sample_end - self.adc_sample_start) * 1e-6

    @property
    def nvirtual_chirp(self) -> int:
        """Number of TX antennas actually enabled (get_radar_paramete.m bitget loop)."""
        tmp = max(self.tx_groups)
        for itx in range(4, 0, -1):
            if (tmp >> ((itx - 1) * 4)) & 1:      # MATLAB bitget(tmp, 1+(itx-1)*4)
                return itx
        return 1

    @property
    def nvirtual_array(self) -> int:
        return self.nvirtual_chirp * 4

    @property
    def anti_nchirp(self) -> int:
        return 1 if self.anti_velamb_en else 0

    @property
    def adc_sample_num(self) -> int:
        """Valid samples per chirp slot (round rule + clamp to rng_nfft).

        init0: round((42.8-2)*25) = 1020 (dec_factor==1 has no +1);
        init1: round((104.4-2)*25) = 2560 -> clamped to rng_nfft = 2048.
        """
        return self.ml_adc_sample_num()

    @property
    def dead_slots_per_chirp(self) -> int:
        """Zero-filled sample slots at the end of every chirp slot.

        The public far-wave captures really do store exact zeros here (see
        README "检测特征"), which is what the mismatch detector keys on.
        """
        return self.rng_nfft - self.adc_sample_num

    @property
    def slope_hz_per_s(self) -> float:
        """FMCW slope S = B / T_ramp [Hz/s]."""
        return self.fmcw_bandwidth * 1e6 / self.ramp_time_s

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / (self.fmcw_startfreq * 1e9)

    @property
    def max_dis_m(self) -> float:
        """Maximum unambiguous range [m]; = C/2 * (Fs/2) / S / dec_factor."""
        return self.ml_max_dis()

    @property
    def rng_res_m(self) -> float:
        """Range resolution over the sampling window [m] (get_radar_paramete.m)."""
        return self.ml_rng_res()

    @property
    def rng_bin_num(self) -> int:
        return self.rng_nfft // 2

    @property
    def rng_res_1dfft_m(self) -> float:
        """Metres per range bin of the RV image [m] = max_dis / (rng_nfft/2)."""
        return self.max_dis_m / self.rng_bin_num

    @property
    def fram_time_s(self) -> float:
        """Frame duration [s] (anti-velocity-ambiguity term included)."""
        return self.ml_fram_time_ms() * 1e-3

    @property
    def max_v_mps(self) -> float:
        """Total unambiguous velocity span [m/s] (v in (-maxV/2, +maxV/2])."""
        return self.ml_max_v()

    @property
    def v_res_mps(self) -> float:
        """Velocity resolution per frame [m/s]."""
        return self.ml_v_res()

    @property
    def v_acu_mps(self) -> float:
        """Metres per Doppler bin = maxV / vel_nfft (== V_res here)."""
        return self.max_v_mps / self.vel_nfft

    # ------------------------------------------------------- layout helpers --
    @property
    def packed_words_per_chirp(self) -> int:
        """32-bit words stored per chirp slot = rng_nfft / 2."""
        return self.rng_nfft // 2

    @property
    def expected_payload_words(self) -> int:
        """Expected uint32 words per channel file = packed_words * nchirp.

        NOTE THE TRAP: this is 131072 for *both* profiles, which is exactly
        why a length check cannot tell near-wave and far-wave data apart.
        """
        return self.packed_words_per_chirp * self.nchirp


def load_profiles(hxx_dir: Path | None = None) -> dict[str, Profile]:
    """Load the official near/far profiles.

    Returns ``{'init0': far profile, 'init1': near profile}``.
    """
    hxx_dir = Path(hxx_dir) if hxx_dir else DEFAULT_HXX_DIR
    return {
        "init0": Profile.from_hxx(hxx_dir / "sensor_config_init0.hxx",
                                  "init0", "远波 far wave (f0)"),
        "init1": Profile.from_hxx(hxx_dir / "sensor_config_init1.hxx",
                                  "init1", "近波 near wave (f1)"),
    }


def mismatch_ratios(capture: Profile, assumed: Profile) -> dict[str, float]:
    """Deterministic apparent-position ratios when ``capture`` data is
    processed with the ``assumed`` profile.

    Range: a de-chirped tone at beat frequency f_b is placed at bin
    k = f_b * rng_nfft / Fs, and one range bin is worth
    max_dis/(rng_nfft/2) = c*Fs/(2*S*rng_nfft) metres, so the rng_nfft
    cancels and the *apparent* range only depends on the assumed slope:

        R_app = f_b * c / (2 * S_assumed) = R_true * S_capture / S_assumed

    Velocity: the wrong reshape changes the number of chirps, so the true
    slow-time spacing between consecutive (fake) chirps becomes
    nchirp_cap * Tp_cap / nchirp_ass, while the processor assumes Tp_ass.
    The Doppler phase per fake chirp is 4*pi*v_true*T_slow_true/lambda_cap,
    interpreted with Tp_ass and lambda_ass:

        v_app = v_true * (nchirp_cap/nchirp_ass) * (Tp_cap/Tp_ass)
                      * (lambda_ass/lambda_cap)
    """
    return {
        "range_ratio": capture.slope_hz_per_s / assumed.slope_hz_per_s,
        "vel_ratio": ((capture.nchirp / assumed.nchirp)
                      * (capture.chirp_period_s / assumed.chirp_period_s)
                      * (assumed.wavelength_m / capture.wavelength_m)),
    }


def parameter_rows(profs: dict[str, Profile]) -> list[tuple[str, str, str, str]]:
    """(quantity, unit, init0 value, init1 value) rows for the README table."""
    p0, p1 = profs["init0"], profs["init1"]

    def f(x: float, nd: int = 6) -> str:
        return f"{x:.{nd}g}"

    return [
        ("起始频率 fmcw_startfreq", "GHz", f(p0.fmcw_startfreq), f(p1.fmcw_startfreq)),
        ("扫频带宽 fmcw_bandwidth", "MHz", f(p0.fmcw_bandwidth), f(p1.fmcw_bandwidth)),
        ("斜坡时长 fmcw_chirp_rampup", "us", f(p0.fmcw_chirp_rampup), f(p1.fmcw_chirp_rampup)),
        ("斜率 S = B/T", "MHz/us", f(p0.slope_hz_per_s / 1e12), f(p1.slope_hz_per_s / 1e12)),
        ("chirp 周期 fmcw_chirp_period", "us", f(p0.fmcw_chirp_period), f(p1.fmcw_chirp_period)),
        ("chirp 数 nchirp", "-", str(p0.nchirp), str(p1.nchirp)),
        ("ADC 采样率 adc_freq", "MHz", f(p0.adc_freq), f(p1.adc_freq)),
        ("采样窗 adc_sample_start..end", "us",
         f"{p0.adc_sample_start:g}..{p0.adc_sample_end:g}",
         f"{p1.adc_sample_start:g}..{p1.adc_sample_end:g}"),
        ("adc_sample_num (round+钳位)", "样点",
         f"{p0.adc_sample_num} (1020, 未钳位)", f"{p1.adc_sample_num} (2560→2048)"),
        ("每 chirp 32bit 字数 rng_nfft/2", "字", str(p0.packed_words_per_chirp),
         str(p1.packed_words_per_chirp)),
        ("rng_nfft / vel_nfft", "-",
         f"{p0.rng_nfft} / {p0.vel_nfft}", f"{p1.rng_nfft} / {p1.vel_nfft}"),
        ("最大不模糊距离 max_dis", "m", f(p0.max_dis_m), f(p1.max_dis_m)),
        ("距离分辨率 rng_res", "m", f(p0.rng_res_m), f(p1.rng_res_m)),
        ("RV 图距离 bin rng_res_1dfft", "m/bin", f(p0.rng_res_1dfft_m), f(p1.rng_res_1dfft_m)),
        ("帧时长 fram_time", "ms", f(p0.fram_time_s * 1e3), f(p1.fram_time_s * 1e3)),
        ("不模糊速度跨度 maxV", "m/s", f(p0.max_v_mps), f(p1.max_v_mps)),
        ("速度分辨率 V_res = V_acu", "m/s", f(p0.v_acu_mps), f(p1.v_acu_mps)),
        ("每通道期望字数 payload", "uint32",
         str(p0.expected_payload_words), str(p1.expected_payload_words)),
    ]
