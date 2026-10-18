"""Monte-Carlo calibration machinery.

Every experiment reduces to the same primitive: push many synthetic noise
frames through the official chain, and record, for every decision cell,

* ``ratio = power / Z``  (the threshold-only statistic: the cell fires when
  ``ratio > alpha``), and
* the same ratio restricted to cells that are 8-neighbourhood local maxima
  (the *full* community detector, which additionally applies the local-max
  rule and drops the range border).

Because ``alpha`` only enters as a comparison against the recorded ratio,
one Monte-Carlo run yields the whole empirical Pfa(alpha) curve, and the
threshold factor for a target Pfa is a single empirical quantile - no
per-alpha re-simulation and no iterative bisection over simulations is
needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import beta as _beta_dist

from .cfar import (
    TRAINING,
    GUARD,
    alpha_from_pfa_single_look,
    ca_noise_estimate,
    interior_mask,
    kernel_count,
    local_max_mask,
    db_ring_stats,
)
from .pipeline import RadarConfig, range_doppler_power, synth_real_noise, to_db

PFA_TARGETS: Tuple[float, float, float] = (1e-2, 1e-3, 1e-4)
N_TRAIN: int = kernel_count()  # 332 cells for the validator defaults

# alpha grid on which empirical Pfa(alpha) curves are tabulated for plotting
CURVE_ALPHAS = np.geomspace(0.4, 16.0, 60)


def make_rng(base_seed: int, *key: int) -> np.random.Generator:
    """Deterministic per-configuration RNG (stable under reordering)."""
    return np.random.default_rng([int(base_seed), *(int(k) for k in key)])


def cfg_tag(cfg: RadarConfig) -> str:
    return f"{cfg.n_chirps}c{cfg.adc_sample_num}s{cfg.rng_nfft}/{cfg.vel_nfft}-{cfg.win}"


# --------------------------------------------------------------------------
# binomial statistics
# --------------------------------------------------------------------------

def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Exact binomial confidence interval (Clopper-Pearson)."""
    if n <= 0:
        raise ValueError("n must be positive")
    k = int(min(max(k, 0), n))
    lo = 0.0 if k == 0 else float(
        _beta_dist.ppf((1.0 - conf) / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(
        _beta_dist.ppf(1.0 - (1.0 - conf) / 2.0, k + 1, n - k))
    return lo, hi


def count_above(sample: np.ndarray, alpha: float) -> int:
    return int(np.count_nonzero(sample > alpha))


def empirical_pfa(sample: np.ndarray, alpha: float) -> float:
    if sample.size == 0:
        return float("nan")
    return count_above(sample, alpha) / sample.size


def alpha_at_pfa(sample: np.ndarray, pfa: float, n_total: int = None) -> float:
    """Empirical threshold factor whose exceedance rate equals ``pfa``.

    ``n_total`` is the denominator of the exceedance rate.  It defaults to
    ``sample.size`` (threshold statistic); for the full-detector statistic the
    denominator is *all* interior cells while ``sample`` holds only the
    local-maximum subset, so the quantile is taken deeper into that subset.
    """
    n = sample.size if n_total is None else int(n_total)
    frac = sample.size / n
    q = min(max(1.0 - pfa / frac, 0.0), 1.0)
    return float(np.quantile(sample, q))


# --------------------------------------------------------------------------
# Monte-Carlo sampling
# --------------------------------------------------------------------------

@dataclass
class Sample:
    k_channels: int
    rho: float
    window: str
    cfg: str
    n_trials: int
    n_cells: int          # interior decision cells (threshold statistic)
    n_cells_lm: int       # interior cells that are local maxima
    ratios: np.ndarray    # float32, threshold-only statistic
    ratios_lm: np.ndarray # float32, full-detector statistic

    @property
    def stat(self) -> Dict[str, np.ndarray]:
        return {"thresh": self.ratios, "full": self.ratios_lm}


def collect_sample(
    cfg: RadarConfig,
    k_channels: int,
    rho: float,
    n_trials: int,
    base_seed: int,
    window: str = None,
) -> Sample:
    """Run ``n_trials`` noise frames through the chain and record ratios."""
    c = cfg if window is None else RadarConfig(**{**cfg.__dict__, "win": window})
    rng = make_rng(base_seed, 1, k_channels, int(round(1000 * rho)),
                   1 if c.win == "hann" else 0, c.n_chirps, c.vel_nfft,
                   c.adc_sample_num, c.rng_nfft)
    parts: list[np.ndarray] = []
    parts_lm: list[np.ndarray] = []
    for _ in range(n_trials):
        adc = synth_real_noise(rng, c, k_channels, rho)
        power = range_doppler_power(adc, c)
        ratio = power / ca_noise_estimate(power)
        valid = interior_mask(power.shape)
        lm = local_max_mask(power)
        parts.append(ratio[valid].astype(np.float32))
        parts_lm.append(ratio[valid & lm].astype(np.float32))
    ratios = np.concatenate(parts)
    ratios_lm = np.concatenate(parts_lm)
    return Sample(k_channels, rho, c.win, cfg_tag(c), n_trials,
                  int(ratios.size), int(ratios_lm.size), ratios, ratios_lm)


def collect_db_margins(
    cfg: RadarConfig, k_channels: int, n_trials: int, base_seed: int
) -> np.ndarray:
    """Record ``(dB - local mean) / local std`` on interior cells.

    This is the statistic a "mean + k * sigma in the dB domain" threshold
    compares against k, so one run again yields the whole Pfa(k) curve.
    """
    rng = make_rng(base_seed, 2, k_channels, cfg.n_chirps, cfg.vel_nfft,
                   cfg.adc_sample_num, cfg.rng_nfft)
    parts: list[np.ndarray] = []
    for _ in range(n_trials):
        adc = synth_real_noise(rng, cfg, k_channels, 0.0)
        power = range_doppler_power(adc, cfg)
        db = to_db(power)
        mu, sd = db_ring_stats(db)
        margin = (db - mu) / sd
        parts.append(margin[interior_mask(power.shape)].astype(np.float32))
    return np.concatenate(parts)


def inject_sinusoid_target(
    adc: np.ndarray, cfg: RadarConfig, rng_bin: int, vel_bin: int, amp: float
) -> np.ndarray:
    """Add a bin-centred point target (identical in all channels).

    A real cosine at fast-time frequency ``rng_bin / rng_nfft`` and slow-time
    frequency ``(vel_bin - vel_nfft//2) / vel_nfft`` lands, after the chain's
    fftshift, exactly on map cell ``(rng_bin, vel_bin)``.
    """
    n_samples, n_chirps, _ = adc.shape
    n = np.arange(n_samples)
    m = np.arange(n_chirps)
    slow_freq = (vel_bin - cfg.vel_nfft // 2) % cfg.vel_nfft / cfg.vel_nfft
    tone = amp * np.cos(2 * np.pi * (rng_bin / cfg.rng_nfft) * n[:, None]
                        + 2 * np.pi * slow_freq * m[None, :])
    return adc + tone[..., None]


# --------------------------------------------------------------------------
# per-configuration summary
# --------------------------------------------------------------------------

@dataclass
class Summary:
    k_channels: int
    rho: float
    window: str
    cfg: str
    n_trials: int
    n_cells: int
    n_cells_lm: int
    curve_alphas: np.ndarray
    curve_thresh: np.ndarray  # nan where no cell exceeded
    curve_full: np.ndarray
    nominal_rows: list        # dicts, one per nominal Pfa target
    alpha_cal: Dict[Tuple[str, float], float]
    achieved: Dict[Tuple[str, float], Tuple[int, int]]  # (n_cells, count)


def summarize(
    sample: Sample,
    pfa_targets: Sequence[float] = PFA_TARGETS,
    curve_alphas: np.ndarray = CURVE_ALPHAS,
    n_train: int = N_TRAIN,
    conf: float = 0.95,
) -> Summary:
    """Empirical Pfa at nominal single-look alphas + calibrated alphas."""
    nominal_rows = []
    stat_arrays = {"thresh": sample.ratios, "full": sample.ratios_lm}
    for pfa in pfa_targets:
        alpha = alpha_from_pfa_single_look(pfa, n_train)
        row = {"pfa_nominal": pfa, "alpha_nominal": alpha}
        for stat, arr in stat_arrays.items():
            n = sample.n_cells  # denominator: all interior decision cells
            k = count_above(arr, alpha)
            p_emp = k / n
            lo, hi = clopper_pearson(k, n, conf)
            row[f"count_{stat}"] = k
            row[f"n_cells_{stat}"] = n
            row[f"pfa_emp_{stat}"] = p_emp
            row[f"ci_lo_{stat}"] = lo
            row[f"ci_hi_{stat}"] = hi
            row[f"within_ci_{stat}"] = bool(lo <= pfa <= hi)
            row[f"factor_{stat}"] = p_emp / pfa
        nominal_rows.append(row)

    alpha_cal: Dict[Tuple[str, float], float] = {}
    achieved: Dict[Tuple[str, float], Tuple[int, int]] = {}
    for stat, arr in stat_arrays.items():
        for pfa in pfa_targets:
            a = alpha_at_pfa(arr, pfa, sample.n_cells)
            alpha_cal[(stat, pfa)] = a
            achieved[(stat, pfa)] = (sample.n_cells, count_above(arr, a))

    ct = np.array([count_above(sample.ratios, a) / sample.n_cells
                   for a in curve_alphas])
    cf = np.array([count_above(sample.ratios_lm, a) / sample.n_cells
                   for a in curve_alphas])
    ct[ct <= 0] = np.nan
    cf[cf <= 0] = np.nan
    return Summary(
        sample.k_channels, sample.rho, sample.window, sample.cfg,
        sample.n_trials, sample.n_cells, sample.n_cells_lm,
        curve_alphas, ct, cf, nominal_rows, alpha_cal, achieved,
    )
