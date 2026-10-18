"""Heuristic capture-vs-profile mismatch detector.

WARNING: this is a heuristic, not a guarantee.  It inspects only the raw
sample stream (never the file name) and votes on two time-domain features
plus, for reporting only, spectral descriptors.

Feature A -- dead-slot zero fingerprint (primary, deterministic here)
    The far-wave profile stores only ``adc_sample_num = 1020`` valid samples
    per 1024-sample chirp slot, and the capture chain zero-fills the last 4
    slots of every chirp (verified: all four public far-wave files have
    ``zero fraction == 1.0`` at offsets 1020..1023 mod 1024).  The near-wave
    profile clamps its window to the full 2048-sample slot, so it has no dead
    slots at all.  ``zero_tail_fraction`` measures exactly that.

Feature B -- chirp-boundary click asymmetry (corroborating)
    At every true chirp restart the recorded waveform shows a transient
    (measured jump 3-11x the sample-to-sample median).  For far data,
    boundaries sit at every multiple of 1024, so jumps at *odd* and *even*
    multiples of 1024 are equally large.  For near data, odd multiples of
    1024 fall mid-chirp, so ``J_odd/J_even`` collapses to ~0.1.

Reported-but-not-voted spectral descriptors (README "方法"):
    per-hypothesis range-energy entropy, near-bin energy fraction, and an
    "impossible velocity" flag (dominant peak beyond the assumed unambiguous
    velocity span).  These turned out to be scene-dependent rather than
    profile-dependent on the public data, so they deliberately do NOT enter
    the decision -- see README for the honest discussion.

Failure modes demonstrated with synthetic streams in the CLI run:
    * pure noise with no zero-filled dead slots and no boundary transients
      -> ``uncertain`` (the detector abstains instead of guessing);
    * a hypothetical capture chain that fills dead slots with garbage
      instead of zeros would kill feature A and leave only feature B.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pipeline import load_cube, rv_image, dominant_peak, range_energy_stats
from .profiles import Profile

# Decision thresholds (measured on the 8 public files; see detector_output.csv):
#   zero_tail: far = 1.000, near = 0.001-0.005        -> cut at 0.05 / 0.5
#   click asym: far = 0.26-1.14, near = 0.07-0.12     -> cut at 0.35 / 0.6
Z_LOW, Z_HIGH = 0.05, 0.5
A_LOW, A_HIGH = 0.35, 0.6


@dataclass
class Detection:
    label: str
    n_samples: int
    zero_tail_fraction: float
    click_odd: float
    click_even: float
    click_asymmetry: float
    decision: str                 # 'init0' | 'init1' | 'uncertain'
    confidence: str               # 'high' | 'low' | '-'
    rule: str
    spectral: dict = field(default_factory=dict)

    def as_row(self) -> list:
        return [self.label,
                f"{self.zero_tail_fraction:.4f}",
                f"{self.click_asymmetry:.3f}",
                self.decision, self.confidence, self.rule]


def zero_fraction(samples: np.ndarray, block: int, offsets: range) -> float:
    """Fraction of exact-zero samples at ``offsets`` (mod ``block``)."""
    samples = np.asarray(samples)
    m = samples.size // block
    seg = samples[: m * block].reshape(m, block)[:, offsets]
    return float(np.mean(seg == 0))


def boundary_clicks(samples: np.ndarray, block: int) -> tuple[float, float]:
    """Mean |Δs| crossing into odd / even multiples of ``block``."""
    d = np.abs(np.diff(np.asarray(samples, dtype=np.float64)))
    idx = np.arange(1, samples.size // block) * block
    j_odd = float(d[idx[0::2] - 1].mean())
    j_even = float(d[idx[1::2] - 1].mean())
    return j_odd, j_even


def spectral_descriptors(samples: np.ndarray,
                         profs: dict[str, Profile]) -> dict:
    """Per-hypothesis spectral stats; descriptive only (scene-dependent)."""
    out: dict = {}
    for name, prof in profs.items():
        cube = load_cube(samples, prof)
        stats = range_energy_stats(cube, prof)
        img = rv_image(cube, prof)
        pk = dominant_peak(img, prof)
        impossible = abs(pk["vel_mps"]) > prof.max_v_mps / 2 - 2 * prof.v_acu_mps
        out[name] = {
            "entropy": round(stats["range_entropy"], 2),
            "frac_le50bins": round(stats["frac_energy_first50bins"], 3),
            "peak_v_mps": round(pk["vel_mps"], 2),
            "impossible_v": bool(impossible),
        }
    return out


def detect_profile(samples: np.ndarray,
                   far: Profile,
                   near: Profile) -> Detection:
    """Classify a raw unpacked sample stream as far-wave or near-wave.

    ``far`` must be the profile with the shorter chirp slot (rng_nfft 1024
    and dead slots); ``near`` the one with rng_nfft 2048.
    """
    samples = np.asarray(samples)
    block = far.rng_nfft
    offs = range(far.adc_sample_num, far.rng_nfft)
    z = zero_fraction(samples, block, offs)
    j_odd, j_even = boundary_clicks(samples, block)
    asym = j_odd / j_even if j_even > 0 else float("inf")

    if z >= Z_HIGH:
        decision, conf = "init0", "high"
        rule = "dead-slot zero fingerprint"
    elif z < Z_LOW:
        if asym <= A_LOW:
            decision, conf = "init1", "high"
            rule = "no dead slots + click asymmetry low"
        elif asym <= A_HIGH:
            decision, conf = "init1", "low"
            rule = "no dead slots; click asymmetry ambiguous"
        else:
            decision, conf = "uncertain", "-"
            rule = "no dead slots, click asymmetry ~1 (no boundary structure?)"
    else:
        if asym >= A_HIGH:
            decision, conf = "init0", "low"
            rule = "partial dead slots; click asymmetry high"
        elif asym <= A_LOW:
            decision, conf = "init1", "low"
            rule = "partial dead slots; click asymmetry low"
        else:
            decision, conf = "uncertain", "-"
            rule = "both features in the gray zone"

    return Detection(
        label="",
        n_samples=int(samples.size),
        zero_tail_fraction=z,
        click_odd=j_odd,
        click_even=j_even,
        click_asymmetry=asym,
        decision=decision,
        confidence=conf,
        rule=rule,
    )
