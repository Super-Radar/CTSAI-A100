"""Mismatch experiment matrix: synthetic ground truth + real-file crossings.

Experiment design (see README for the honesty red line):

* "correct"  = capture profile and assumed profile agree;
* "mismatch" = the *same* data stream is reshaped/scaled with the *other*
  profile.  Both files have exactly 131072 payload words per channel, so the
  reshape succeeds silently in both directions -- that silence is the trap
  this project documents.

Synthetic captures are generated from the ideal de-chirped IF model

    phase(n, k) = 4*pi*(R0 + v*n*Tp)/lambda
                  + 2*pi*(2*S/c)*(R0 + v*n*Tp)*t_k

with n the chirp index, k the fast-time sample, t_k = k/Fs.  The fast term
puts the beat tone at f_b = 2*S*R/c, the slow term advances the Doppler phase
by 4*pi*v*Tp/lambda per chirp, and samples beyond ``adc_sample_num`` inside
each chirp slot are zero-filled -- which is what the real far-wave capture
firmware does and what the detector keys on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import (load_cube, rv_image, dominant_peak, range_energy_stats,
                       samples_to_words)
from .profiles import Profile, mismatch_ratios, C_LIGHT


# ------------------------------------------------------------- synthetic ADC
def synth_adc_words(profile: Profile,
                    targets: list[tuple[float, float, float]],
                    noise_sigma: float = 0.0,
                    seed: int = 0,
                    zero_dead_slots: bool = True) -> np.ndarray:
    """Ideal single/multi-target capture for ``profile`` as a uint32 stream.

    ``targets`` is a list of ``(range_m, vel_mps, amplitude)`` triples.
    """
    rng = np.random.default_rng(seed)
    n = profile.nchirp
    nfft = profile.rng_nfft
    t_fast = np.arange(nfft) / profile.fs_hz
    t_slow = np.arange(n) * profile.chirp_period_s

    sig = np.zeros((n, nfft))
    for range_m, vel_mps, amp in targets:
        rn = range_m + vel_mps * t_slow                     # [n]
        phase = (4.0 * np.pi * rn / profile.wavelength_m)[:, None] \
            + 2.0 * np.pi * (2.0 * profile.slope_hz_per_s / C_LIGHT) \
            * np.outer(rn, t_fast)
        sig += amp * np.cos(phase)

    if noise_sigma > 0.0:
        sig += rng.normal(0.0, noise_sigma, sig.shape)
    if zero_dead_slots and profile.adc_sample_num < nfft:
        sig[:, profile.adc_sample_num:] = 0.0

    return samples_to_words(sig.reshape(-1))


def synth_cube(profile: Profile,
               targets: list[tuple[float, float, float]],
               **kw) -> np.ndarray:
    return load_cube(synth_adc_words(profile, targets, **kw), profile)


def predict_apparent(cap: Profile, ass: Profile,
                     r_true: float, v_true: float) -> dict:
    """Deterministic apparent peak position of an ideal (R, v) target when
    ``cap`` data is processed with the ``ass`` profile.

    Range:  R_app = R_true * S_cap / S_ass   (rng_nfft cancels, see
    profiles.mismatch_ratios).

    Velocity: the *nominal* mapping is the closed form of
    mismatch_ratios, but when one true chirp spans TWO assumed chirps
    (near data read as far: 2048 -> 2 x 1024) every odd assumed chirp starts
    mid-tone, at beat phase advanced by

        theta = 2*pi * f_b * (rng_nfft_ass / Fs)

    which is a period-2 modulation of the Doppler sequence.  Its spectrum
    splits the tone into a main component (weight |1+e^{i theta}|) and an
    image at bin k +/- vel_nfft/2 (weight |1-e^{i theta}|).  Depending on
    theta one or the other dominates -- so the apparent velocity may jump
    by half the assumed unambiguous span (+-maxV/2).  This function predicts
    which component wins and returns the wrapped apparent velocity.
    """
    ratios = mismatch_ratios(cap, ass)
    f_b = 2.0 * cap.slope_hz_per_s * r_true / C_LIGHT
    r_app = r_true * ratios["range_ratio"]
    v_main = v_true * ratios["vel_ratio"]
    k_main = v_main / ass.v_acu_mps

    jump = False
    k_pred = k_main
    if cap.rng_nfft == 2 * ass.rng_nfft:
        theta = 2.0 * np.pi * f_b * (ass.rng_nfft / ass.fs_hz)
        if abs(1.0 - np.exp(1j * theta)) > abs(1.0 + np.exp(1j * theta)):
            k_pred = k_main + ass.vel_nfft / 2.0
            jump = True
    # wrap to the signed bin range (-N/2, N/2-1]
    n = ass.vel_nfft
    k_wrap = (k_pred + n / 2.0) % n - n / 2.0
    return {
        "range_m": r_app,
        "vel_mps": k_wrap * ass.v_acu_mps,
        "k_range": f_b * ass.rng_nfft / ass.fs_hz,
        "k_doppler": k_wrap,
        "half_frame_jump": jump,
        "vel_nominal_mps": v_main,
    }


# ------------------------------------------------------------ result records
@dataclass
class PeakRecord:
    capture: str            # 'far(init0)' | 'near(init1)' | 'synthetic-...'
    assumed: str
    k_range: int
    k_doppler: int
    range_m: float
    vel_mps: float
    power_db: float

    def as_row(self) -> list:
        return [self.capture, self.assumed, self.k_range, self.k_doppler,
                f"{self.range_m:.3f}", f"{self.vel_mps:.3f}",
                f"{self.power_db:.1f}"]


def _peak_record(cube: np.ndarray, cap_name: str, ass_name: str,
                 ass_profile: Profile) -> PeakRecord:
    img = rv_image(cube, ass_profile)
    pk = dominant_peak(img, ass_profile)
    return PeakRecord(cap_name, ass_name, pk["k_range"], pk["k_doppler"],
                      pk["range_m"], pk["vel_mps"], pk["power_db"])


# ------------------------------------------------------- synthetic experiment
def run_synthetic(profs: dict[str, Profile],
                  target: tuple[float, float] = (8.0, 2.0),
                  amplitude: float = 20000.0,
                  noise_sigma: float = 0.0,
                  seed: int = 20260929) -> list[dict]:
    """Deterministic apparent-position verification on synthetic captures.

    For every (capture, assumed) combination the measured peak position is
    compared with the theoretical prediction
        R_app = R_true * S_cap / S_ass,
        v_app = v_true * (nchirp_cap/nchirp_ass)*(Tp_cap/Tp_ass)*(lam_ass/lam_cap),
    errors are reported in bins.
    """
    rows: list[dict] = []
    for cap_name, cap in profs.items():
        words = synth_adc_words(cap, [(*target, amplitude)],
                                noise_sigma=noise_sigma, seed=seed)
        for ass_name, ass in profs.items():
            cube = load_cube(words, ass)
            rec = _peak_record(cube, f"synthetic-{cap_name}", ass_name, ass)
            pred = predict_apparent(cap, ass, target[0], target[1])
            rows.append({
                "capture": cap_name,
                "assumed": ass_name,
                "correct": cap_name == ass_name,
                "k_range": rec.k_range,
                "k_doppler": rec.k_doppler,
                "R_meas_m": rec.range_m,
                "R_pred_m": pred["range_m"],
                "R_err_bins": (rec.range_m - pred["range_m"]) / ass.rng_res_1dfft_m,
                "v_meas_mps": rec.vel_mps,
                "v_pred_mps": pred["vel_mps"],
                "v_err_bins": (rec.vel_mps - pred["vel_mps"]) / ass.v_acu_mps,
                "v_jump": pred["half_frame_jump"],
                "v_nominal_mps": pred["vel_nominal_mps"],
            })
    return rows


# ------------------------------------------------------------- real crossings
def list_captures(data_dir: Path) -> dict[str, list[Path]]:
    """Group the public capture files by true profile (from their names)."""
    data_dir = Path(data_dir)
    far = sorted(data_dir.glob("adc_远波_*_f0_*.txt"))
    near = sorted(data_dir.glob("adc_近波_*_f1_*.txt"))
    if not far or not near:
        raise FileNotFoundError(
            f"expected adc_远波_*_f0_*.txt and adc_近波_*_f1_*.txt under {data_dir}")
    return {"init0": far, "init1": near}


def run_real_matrix(profs: dict[str, Profile],
                    data_dir: Path,
                    cubes_cache: dict | None = None) -> list[dict]:
    """Cross every public capture with both profiles.

    For each file the dominant peak is located under the *correct* profile
    (ground truth for this file's dominant reflector) and again under the
    *wrong* profile; the table reports the apparent shift next to the
    deterministic ratio prediction.  When the dominant cell moves to a
    leakage bin under the wrong reshape the row is flagged ``peak_jump``
    instead of being quietly compared (see README "局限").
    """
    from .pipeline import read_raw_words

    rows: list[dict] = []
    for cap_name, files in list_captures(data_dir).items():
        cap = profs[cap_name]
        other_name = "init1" if cap_name == "init0" else "init0"
        other = profs[other_name]
        ratios = mismatch_ratios(cap, other)
        for path in files:
            words = read_raw_words(path)
            cube_ok = load_cube(words, cap)
            rec_ok = _peak_record(cube_ok, cap_name, cap_name, cap)
            cube_bad = load_cube(words, other)
            rec_bad = _peak_record(cube_bad, cap_name, other_name, other)

            r_pred = rec_ok.range_m * ratios["range_ratio"]
            v_pred = rec_ok.vel_mps * ratios["vel_ratio"]
            r_err_bins = (rec_bad.range_m - r_pred) / other.rng_res_1dfft_m
            v_err_bins = (rec_bad.vel_mps - v_pred) / other.v_acu_mps
            # A peak "jump" means the wrong reshape changed which reflector
            # dominates (typically into a leakage/DC cell); then the apparent
            # position is not a pure rescaling of the correct one.
            jumped = (abs(r_err_bins) > 3.0 or abs(v_err_bins) > 3.0)
            rows.append({
                "file": path.name,
                "capture": cap_name,
                "assumed_ok": f"{rec_ok.k_range}@{rec_ok.k_doppler}",
                "R_ok_m": rec_ok.range_m,
                "v_ok_mps": rec_ok.vel_mps,
                "assumed_bad": f"{rec_bad.k_range}@{rec_bad.k_doppler}",
                "R_bad_m": rec_bad.range_m,
                "v_bad_mps": rec_bad.vel_mps,
                "R_ratio_pred": ratios["range_ratio"],
                "R_ratio_meas": (rec_bad.range_m / rec_ok.range_m
                                 if rec_ok.range_m > 0 else float("nan")),
                "v_ratio_pred": ratios["vel_ratio"],
                "v_err_bins": v_err_bins,
                "R_err_bins": r_err_bins,
                "peak_jump": jumped,
                "cube_ok": cube_ok,
                "cube_bad": cube_bad,
            })
            if cubes_cache is not None:
                cubes_cache[path.name] = (cube_ok, cube_bad)
    return rows
