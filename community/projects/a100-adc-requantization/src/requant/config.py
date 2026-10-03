"""Radar frame configurations for the CTSAI-A100 demo data.

The two measured configurations mirror the firmware files
``sensor_config_init0.hxx`` (far-beam, init0) and ``sensor_config_init1.hxx``
(near-beam, init1) under
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/cfg/CTASI-A100配置/``.

``adc_sample_num`` reproduces the derivation in the official
``init_function/get_radar_paramete.m``::

    if dec_factor > 1:
        adc_sample_num = round((adc_sample_end - adc_sample_start)
                               * adc_freq / dec_factor) + 1
    else:
        adc_sample_num = round((adc_sample_end - adc_sample_start)
                               * adc_freq / dec_factor)
    adc_sample_num = min(adc_sample_num, rng_nfft)

For init0 this gives round((42.8 - 2) * 25) = 1020 (out of 1024 padded bins);
for init1 it gives round((104.4 - 2) * 25) = 2560 which is then capped to
rng_nfft = 2048, i.e. the full record is used.  The historical "1020 vs 1021"
ambiguity comes from floating-point rounding of (42.8-2)*25; IEEE-754 double
gives 1019.9999999999999 -> round -> 1020 in both MATLAB and NumPy.  This
convention is recorded in the README.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadarCfg:
    """One radar frame configuration (per firmware sensor_config_*.hxx)."""

    name: str
    f0_hz: float              # start carrier frequency [Hz]
    bw_hz: float              # sweep bandwidth [Hz]
    rampup_us: float          # chirp ramp-up duration [us]
    period_us: float          # chirp period [us]
    nchirp: int               # chirps per frame
    adc_freq_mhz: float       # ADC sample rate [MHz]
    dec_factor: int
    adc_sample_start: float   # [us]
    adc_sample_end: float     # [us]
    rng_nfft: int             # range FFT size
    vel_nfft: int             # Doppler FFT size
    rng_fft_scalar: int       # firmware FFT output divisor
    vel_fft_scalar: int
    nvirtual_chirp: int = 1
    label: str = ""

    # ------------------------------------------------------------------ defs
    @property
    def adc_sample_num(self) -> int:
        """Valid fast-time samples, mirroring get_radar_paramete.m."""
        span = (self.adc_sample_end - self.adc_sample_start) * self.adc_freq_mhz
        if self.dec_factor > 1:
            n = round(span / self.dec_factor) + 1
        else:
            n = round(span / self.dec_factor)
        return min(int(n), self.rng_nfft)

    @property
    def words_per_channel(self) -> int:
        """uint32 words per channel file = (rng_nfft/2) * nchirp."""
        return (self.rng_nfft // 2) * self.nchirp

    @property
    def rng_res_m(self) -> float:
        """Range resolution [m] from the valid sample span."""
        return 3e8 / 2.0 / (self.bw_hz / (self.rampup_us * 1e-6)
                            * (self.adc_sample_end - self.adc_sample_start) * 1e-6)

    @property
    def vel_res_mps(self) -> float:
        """Doppler resolution [m/s]."""
        return 0.5 * 3e8 / self.f0_hz / (self.period_us * 1e-6)

    @property
    def max_vel_mps(self) -> float:
        return self.vel_nfft * self.vel_res_mps


INIT0 = RadarCfg(
    name="init0",
    label="init0 far-beam (76.3 GHz / 300 MHz / 43 us ramp)",
    f0_hz=76.3e9, bw_hz=300e6, rampup_us=43.0, period_us=48.0,
    nchirp=256, adc_freq_mhz=25.0, dec_factor=1,
    adc_sample_start=2.0, adc_sample_end=42.8,
    rng_nfft=1024, vel_nfft=256,
    rng_fft_scalar=0xFF, vel_fft_scalar=0x3F,
)

INIT1 = RadarCfg(
    name="init1",
    label="init1 near-beam (76.119 GHz / 750 MHz / 104.65 us ramp)",
    f0_hz=76.119e9, bw_hz=750e6, rampup_us=104.65, period_us=108.5,
    nchirp=128, adc_freq_mhz=25.0, dec_factor=1,
    adc_sample_start=2.0, adc_sample_end=104.4,
    rng_nfft=2048, vel_nfft=128,
    rng_fft_scalar=0xFF, vel_fft_scalar=0x3F,
)

SYNTH = RadarCfg(
    name="synthetic",
    label="synthetic int16 benchmark frame",
    f0_hz=76.3e9, bw_hz=300e6, rampup_us=43.0, period_us=48.0,
    nchirp=128, adc_freq_mhz=25.0, dec_factor=1,
    adc_sample_start=0.0, adc_sample_end=40.96,
    rng_nfft=1024, vel_nfft=128,
    rng_fft_scalar=1, vel_fft_scalar=1,
)

DATA_DIR = "ADC数据采集/示例adc数据和结果"
REAL_FILES = (
    ("adc_远波_20260512135827_f0_0_13_58_36_748.txt", INIT0, 0),
    ("adc_远波_20260512135827_f0_1_13_58_45_367.txt", INIT0, 1),
    ("adc_远波_20260512135827_f0_2_13_58_53_977.txt", INIT0, 2),
    ("adc_远波_20260512135827_f0_3_13_59_02_596.txt", INIT0, 3),
    ("adc_近波_20260512141857_f1_0_14_19_05_965.txt", INIT1, 0),
    ("adc_近波_20260512141857_f1_1_14_19_14_570.txt", INIT1, 1),
    ("adc_近波_20260512141857_f1_2_14_19_23_170.txt", INIT1, 2),
    ("adc_近波_20260512141857_f1_3_14_19_31_788.txt", INIT1, 3),
)
