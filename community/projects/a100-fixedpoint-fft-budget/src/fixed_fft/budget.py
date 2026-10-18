"""Error metrics against a float64 reference and the wordlength budget sweeps.

Every metric compares the fixed-point FFT output of :mod:`fft_fixed` with a
``numpy.fft`` float64 reference computed from the *same* integer input
samples, so the reported error isolates the finite-wordlength effects of the
FFT arithmetic itself.  Only numerical error statistics are produced; no
runtime or memory figures are measured anywhere in this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .fft_fixed import (
    OVERFLOW_SATURATE,
    SCALING_NONE,
    SCALING_SCALE_1BIT,
    FixedFFTResult,
    fft_fixed,
)
from . import inputs as fx_inputs

DEFAULT_N_FFT = 1024
DEFAULT_Q = 16
B_LIST = list(range(8, 33))
Q_LIST = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

METRIC_KEYS = (
    "input",
    "B",
    "Q",
    "scaling",
    "overflow_mode",
    "n_fft",
    "input_shift",
    "input_clipped",
    "rel_l2_power",
    "peak_bin_ref",
    "peak_bin_fix",
    "peak_bin_shift",
    "peak_amp_rel_err",
    "energy_ratio",
    "total_overflow",
    "max_stage_peak",
)


def reference_spectrum(x: Sequence[int], n_fft: int) -> np.ndarray:
    """float64 DFT of the same integer samples the fixed-point core receives."""
    return np.fft.fft(np.asarray(x, dtype=np.float64).ravel(), n=n_fft)


def spectrum_metrics(fixed: FixedFFTResult, x_ref: Sequence[int]) -> Dict:
    """Compare a fixed-point result with its float64 reference.

    Metrics (all dimensionless unless noted):

    * ``rel_l2_power`` - relative L2 error of the one-sided power spectrum
      ``||Pfix - Pref||_2 / ||Pref||_2`` over bins ``0 .. n_fft/2``;
    * ``peak_bin_ref`` / ``peak_bin_fix`` - argmax bin of the one-sided
      magnitude spectrum (reference / fixed), ``peak_bin_shift`` their
      difference in bins;
    * ``peak_amp_rel_err`` - relative magnitude error at the peak bin;
    * ``energy_ratio`` - ratio of total two-sided spectral energies;
    * ``total_overflow`` / ``max_stage_peak`` (LSB) - datapath statistics.
    """
    n_fft = fixed.n_fft
    x_ref_input = np.zeros(n_fft, dtype=np.float64)
    xs = np.asarray(x_ref, dtype=np.float64).ravel()
    x_ref_input[: xs.size] = xs

    xfix = fixed.spectrum() * fixed.effective_gain()
    xref = np.fft.fft(x_ref_input)

    n_side = n_fft // 2 + 1
    pfix = np.abs(xfix[:n_side])
    pref = np.abs(xref[:n_side])

    denom = np.linalg.norm(pref)
    rel_l2 = float(np.linalg.norm(pfix - pref) / denom) if denom > 0 else 0.0

    i_ref = int(np.argmax(pref))
    i_fix = int(np.argmax(pfix))
    peak_amp_rel_err = float(abs(pfix[i_fix] - pref[i_ref]) / pref[i_ref]) if pref[i_ref] > 0 else 0.0

    e_fix = float(np.sum(np.abs(xfix) ** 2))
    e_ref = float(np.sum(np.abs(xref) ** 2))
    energy_ratio = float(e_fix / e_ref) if e_ref > 0 else 0.0

    return {
        "input_shift": int(fixed.input_shift),
        "input_clipped": int(fixed.input_clipped),
        "rel_l2_power": rel_l2,
        "peak_bin_ref": i_ref,
        "peak_bin_fix": i_fix,
        "peak_bin_shift": i_fix - i_ref,
        "peak_amp_rel_err": peak_amp_rel_err,
        "energy_ratio": energy_ratio,
        "total_overflow": int(fixed.total_overflow),
        "max_stage_peak": int(max(fixed.stage_peak)) if fixed.stage_peak else 0,
    }


def run_case(
    name: str,
    x: Sequence[int],
    n_fft: Optional[int] = None,
    B: int = 24,
    Q: int = DEFAULT_Q,
    scaling: str = SCALING_NONE,
    overflow: str = OVERFLOW_SATURATE,
) -> Dict:
    """Run one fixed-point FFT and return the full metric row."""
    x = np.asarray(x, dtype=np.int64).ravel()
    if n_fft is None:
        n_fft = 1 << (x.size - 1).bit_length()
    fixed = fft_fixed(x, n_fft=n_fft, B=B, Q=Q, scaling=scaling, overflow=overflow)
    row = {
        "input": name,
        "B": int(B),
        "Q": int(Q),
        "scaling": scaling,
        "overflow_mode": overflow,
        "n_fft": int(n_fft),
    }
    row.update(spectrum_metrics(fixed, x))
    return row


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def sweep_B(
    signals: Dict[str, np.ndarray],
    B_values: Iterable[int],
    Q: int = DEFAULT_Q,
    scaling: str = SCALING_NONE,
) -> List[Dict]:
    """Experiment 1: error vs internal wordlength B (Q and scaling fixed)."""
    rows: List[Dict] = []
    for name, x in signals.items():
        for B in B_values:
            rows.append(run_case(name, x, B=B, Q=Q, scaling=scaling))
    return rows


def sweep_Q(
    signals: Dict[str, np.ndarray],
    Q_values: Iterable[int],
    B: int = 24,
    scaling: str = SCALING_NONE,
) -> List[Dict]:
    """Experiment 2: error vs twiddle fraction bits Q (B fixed)."""
    rows: List[Dict] = []
    for name, x in signals.items():
        for Q in Q_values:
            rows.append(run_case(name, x, B=B, Q=Q, scaling=scaling))
    return rows


def scaling_compare(
    signals: Dict[str, np.ndarray],
    B_values: Iterable[int],
    Q: int = DEFAULT_Q,
) -> List[Dict]:
    """Experiment 3: no scaling vs 1-bit-per-stage scaling across B."""
    rows: List[Dict] = []
    for name, x in signals.items():
        for B in B_values:
            rows.append(run_case(name, x, B=B, Q=Q, scaling=SCALING_NONE))
            rows.append(run_case(name, x, B=B, Q=Q, scaling=SCALING_SCALE_1BIT))
    return rows


def stage_waterfall(
    signals: Dict[str, np.ndarray],
    B_peak: int = 32,
    B_overflow: int = 16,
    Q: int = DEFAULT_Q,
    scaling: str = SCALING_NONE,
) -> List[Dict]:
    """Experiment 4: per-stage peak magnitude (unsaturated) and overflow counts.

    The peak trajectory is measured at ``B_peak`` (wide enough that no
    saturation masks the growth); the overflow counts are measured at the
    narrower ``B_overflow`` datapath.  ``peak_bits`` is the integer wordlength
    that stage s's peak would require.
    """
    rows: List[Dict] = []
    for name, x in signals.items():
        fixed_peak = fft_fixed(x, B=B_peak, Q=Q, scaling=scaling)
        fixed_ovf = fft_fixed(x, B=B_overflow, Q=Q, scaling=scaling)
        for stage in range(fixed_peak.n_stages):
            peak = int(fixed_peak.stage_peak[stage])
            rows.append(
                {
                    "input": name,
                    "B_peak": int(B_peak),
                    "B_overflow": int(B_overflow),
                    "Q": int(Q),
                    "scaling": scaling,
                    "stage": stage + 1,
                    "peak_lsb": peak,
                    "peak_bits": int(max(1, peak).bit_length()),
                    "overflow_at_B_overflow": int(fixed_ovf.stage_overflow[stage]),
                }
            )
    return rows


def real_data_error(
    adc_path,
    B_values: Iterable[int],
    Q: int = DEFAULT_Q,
    scaling: str = SCALING_NONE,
    rng_nfft: int = fx_inputs.INIT0["rng_nfft"],
    nchirp: int = fx_inputs.INIT0["nchirp"],
    adc_sample_num: int = fx_inputs.INIT0["adc_sample_num"],
) -> List[Dict]:
    """Experiment 5: official ADC range FFT, fixed-point vs float64, vs B.

    Loads one RX dump through the official chain, builds the windowed range
    FFT inputs (``hann(adc_sample_num)``, zero-padded to ``rng_nfft``, rounded
    to the int16 grid) and runs every chirp at each wordlength.  The float64
    reference uses the identical integer input vectors.
    """
    frame = fx_inputs.load_official_range_inputs(
        adc_path, rng_nfft=rng_nfft, nchirp=nchirp, adc_sample_num=adc_sample_num
    )
    refs = np.stack([np.fft.fft(frame[c].astype(np.float64), n=rng_nfft) for c in range(nchirp)])
    n_side = rng_nfft // 2 + 1
    pref_all = np.abs(refs[:, :n_side])
    ref_peak = pref_all.argmax(axis=1)
    ref_peak_amp = pref_all[np.arange(nchirp), ref_peak]

    rows: List[Dict] = []
    for B in B_values:
        rel_errors = np.empty(nchirp)
        peak_rel_errors = np.empty(nchirp)
        matches = 0
        total_overflow = 0
        input_shift = 0
        for c in range(nchirp):
            fixed = fft_fixed(frame[c], n_fft=rng_nfft, B=B, Q=Q, scaling=scaling)
            xfix = fixed.spectrum() * fixed.effective_gain()
            pfix = np.abs(xfix[:n_side])
            rel_errors[c] = np.linalg.norm(pfix - pref_all[c]) / np.linalg.norm(pref_all[c])
            i_fix = int(np.argmax(pfix))
            peak_rel_errors[c] = abs(pfix[i_fix] - ref_peak_amp[c]) / ref_peak_amp[c]
            matches += int(i_fix == ref_peak[c])
            total_overflow += fixed.total_overflow
            input_shift = max(input_shift, fixed.input_shift)
        rows.append(
            {
                "file": Path(adc_path).name,
                "B": int(B),
                "Q": int(Q),
                "scaling": scaling,
                "nchirp": int(nchirp),
                "rng_nfft": int(rng_nfft),
                "adc_sample_num": int(adc_sample_num),
                "input_shift": int(input_shift),
                "mean_rel_l2_power": float(rel_errors.mean()),
                "max_rel_l2_power": float(rel_errors.max()),
                "median_rel_l2_power": float(np.median(rel_errors)),
                "mean_peak_amp_rel_err": float(peak_rel_errors.mean()),
                "peak_bin_match_fraction": float(matches / nchirp),
                "total_overflow_all_chirps": int(total_overflow),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# helpers for the report
# ---------------------------------------------------------------------------

def sha256_of_files(paths: Iterable) -> List[str]:
    """sha256 hex digests of the given files, one line of text per file."""
    lines = []
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {Path(p).name}")
    return lines


def min_b_below(rows: Sequence[Dict], threshold: float, **filters) -> Optional[int]:
    """Smallest B in ``rows`` (matching ``filters``) with rel_l2_power < threshold
    for every distinct input present."""
    wanted = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    names = {r["input"] for r in wanted}
    for B in sorted({r["B"] for r in wanted}):
        at_b = [r for r in wanted if r["B"] == B]
        if len(at_b) == len(names) and all(r["rel_l2_power"] < threshold for r in at_b):
            return int(B)
    return None


def min_q_below(rows: Sequence[Dict], threshold: float) -> Optional[int]:
    """Smallest Q in a sweep_Q result with rel_l2_power < threshold for all inputs."""
    names = {r["input"] for r in rows}
    for Q in sorted({r["Q"] for r in rows}):
        at_q = [r for r in rows if r["Q"] == Q]
        if len(at_q) == len(names) and all(r["rel_l2_power"] < threshold for r in at_q):
            return int(Q)
    return None


def overflow_boundary(rows: Sequence[Dict], scaling: str) -> Dict[str, Optional[int]]:
    """Per input: the largest B that still shows overflow for a given scaling."""
    out: Dict[str, Optional[int]] = {}
    for r in rows:
        if r.get("scaling") != scaling:
            continue
        name = r["input"]
        if r["total_overflow"] > 0:
            out[name] = max(out.get(name) or 0, r["B"])
        else:
            out.setdefault(name, None)
    return out
