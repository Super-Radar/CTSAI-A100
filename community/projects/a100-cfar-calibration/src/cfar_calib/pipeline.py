"""Signal-processing chain mirroring the official CTSAI-A100 MATLAB platform.

The chain reproduced here (see README "方法" for the file-level mapping):

* real int16 samples, two per uint32 word (high 16 bits first, low 16 bits
  second) -- ``init_function/adc_mem_2_real_fixed.m``;
* Hann range window over the *actual* sample count, range FFT padded to
  ``rng_nfft``, first ``rng_nfft // 2`` bins kept -- ``FFT/do_fft_2d.m``;
* Hann velocity window over the chirps, Doppler FFT of length ``vel_nfft``
  (fftshifted only for display, mirroring the community validator);
* K-channel power (|.|^2) mean, then dB(power) -- main script +
  ``FFT/calculate_global_noise.m``.

Synthetic input is real white Gaussian noise per (sample, chirp, channel),
which is the noise model of the real ADC stream, so synthetic frames go
through exactly the same operator chain as captured frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

C0 = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    """Processing parameters of the far-beam f0 profile (init0.hxx).

    ``adc_sample_num`` follows the physical sampling window
    round((42.8 us - 2 us) * 25 MHz) + 1 = 1021.  The official
    ``get_radar_paramete.m`` omits the ``+ 1`` when ``dec_factor == 1`` and
    would report 1020; both conventions are discussed in the README.
    """

    n_chirps: int = 256
    adc_sample_num: int = 1021
    rng_nfft: int = 1024
    vel_nfft: int = 256
    fc_ghz: float = 76.3
    bandwidth_mhz: float = 300.0
    ramp_us: float = 43.0
    period_us: float = 48.0
    adc_freq_mhz: float = 25.0
    sample_start_us: float = 2.0
    sample_end_us: float = 42.8
    win: str = "hann"  # "hann" (official) or "rect" (uncorrelated-bins control)

    @property
    def rng_bins(self) -> int:
        return self.rng_nfft // 2

    @property
    def slope(self) -> float:
        """Chirp slope in Hz/s."""
        return self.bandwidth_mhz * 1e6 / (self.ramp_us * 1e-6)

    @property
    def range_bin_m(self) -> float:
        return C0 * (self.adc_freq_mhz * 1e6) / (2.0 * self.slope * self.rng_nfft)

    def reduced(self, n_chirps: int = 64) -> "RadarConfig":
        """Reduced frame used for Monte-Carlo sweeps (see README "局限")."""
        return replace(self, n_chirps=n_chirps, vel_nfft=n_chirps)


def window_vector(n: int, kind: str) -> np.ndarray:
    if kind == "hann":
        return np.hanning(n)  # symmetric, matches MATLAB hann(n)
    if kind == "rect":
        return np.ones(n)
    raise ValueError(f"unknown window kind: {kind!r}")


def synth_real_noise(
    rng: np.random.Generator,
    cfg: RadarConfig,
    n_channels: int,
    rho: float = 0.0,
) -> np.ndarray:
    """Real white Gaussian samples shaped ``(adc_sample_num, n_chirps, K)``.

    Equicorrelated channels are synthesised with one shared random
    component per (sample, chirp)::

        x_i = sqrt(rho) * s + sqrt(1 - rho) * e_i

    ``rho`` is the per-sample amplitude correlation coefficient between any
    two channels; the resulting *power*-domain correlation is ``rho**2``.
    ``rho == 1`` makes all channels identical (cross-check: K-channel
    averaging then degenerates to the single-channel case).
    """
    shape = (cfg.adc_sample_num, cfg.n_chirps)
    if rho == 0.0:
        return rng.standard_normal(shape + (n_channels,))
    shared = rng.standard_normal(shape)
    indep = rng.standard_normal(shape + (n_channels,))
    return np.sqrt(rho) * shared[..., None] + np.sqrt(1.0 - rho) * indep


def range_doppler_power(adc: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Official window + 2D-FFT chain -> averaged power map.

    Parameters
    ----------
    adc : real array ``(n_samples, n_chirps, K)`` (already DC-conditioned).
        ``n_samples <= rng_nfft``; the remainder is zero padded.

    Returns
    -------
    power : ``(rng_nfft // 2, vel_nfft)`` mean over channels of |velFFT|^2.
    """
    n_samples, n_chirps, n_ch = adc.shape
    if n_samples > cfg.rng_nfft:
        raise ValueError("n_samples exceeds rng_nfft")
    if n_chirps > cfg.vel_nfft:
        raise ValueError("n_chirps exceeds vel_nfft")
    padded = np.zeros((cfg.rng_nfft, n_chirps, n_ch), dtype=adc.dtype)
    padded[:n_samples] = adc * window_vector(n_samples, cfg.win)[:, None, None]
    rng_fft = np.fft.fft(padded, axis=0)[: cfg.rng_bins]
    y = rng_fft * window_vector(n_chirps, cfg.win)[None, :, None]
    vel_fft = np.fft.fftshift(np.fft.fft(y, n=cfg.vel_nfft, axis=1), axes=1)
    return np.mean(np.abs(vel_fft) ** 2, axis=2)


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-300))


def global_noise_db(power_db: np.ndarray) -> float:
    """Mirror of ``FFT/calculate_global_noise.m`` (fixed-region mean in dB)."""
    n_rng, n_vel = power_db.shape
    r0, r1 = round(0.75 * n_rng), round(0.8 * n_rng)
    region1 = power_db[r0:r1, 6 : n_vel // 2 - 6]
    region2 = power_db[r0:r1, n_vel // 2 + 6 : n_vel - 6]
    if region1.size == 0 or region2.size == 0:
        raise ValueError("noise region empty; map too small")
    return float(np.minimum(region1, region2).mean())


def load_real_adc(
    files: list, rng_nfft: int = 1024, n_chirps: int = 256, remove_dc: bool = True
) -> tuple:
    """Load the repository's per-channel uint32 capture files.

    Follows ``init_function/load_adc_data.m``: each file is flattened,
    aligned to ``rng_nfft/2 * n_chirp`` words (truncate surplus from the
    end / zero-pad shortfall), reshaped fast-time-major per chirp, then each
    uint32 word is split into two consecutive signed int16 samples
    (high 16 bits first).  Output ``(adc, n_values)`` with ``adc`` shaped
    ``(rng_nfft, n_chirps, K)`` and ``n_values`` the raw per-file counts.

    Note: the community validator ``run_reference_validation.py`` drops the
    *first* value of a 131073-value file instead of the trailing one; the
    official MATLAB code truncates from the end, which is what we do here.
    The one-sample offset is immaterial for spectral-shape comparison.
    """
    words_per_chirp = rng_nfft // 2
    expected = words_per_chirp * n_chirps
    channels = []
    n_values = []
    for path in files:
        values = np.loadtxt(path, delimiter=",").reshape(-1)
        values = values[np.isfinite(values)]
        n_values.append(int(values.size))
        if values.size > expected:
            values = values[:expected]
        elif values.size < expected:
            values = np.concatenate([values, np.zeros(expected - values.size)])
        words = values.astype(np.uint32)
        high = (words >> 16).astype(np.uint16).view(np.int16).astype(np.float64)
        low = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float64)
        samples = np.empty(expected * 2, dtype=np.float64)
        samples[0::2] = high
        samples[1::2] = low
        channels.append(samples.reshape((rng_nfft, n_chirps), order="F"))
    adc = np.stack(channels, axis=2)
    if remove_dc:
        adc = adc - adc.mean(axis=0, keepdims=True)
    return adc, n_values


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
