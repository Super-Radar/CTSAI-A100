"""Conservative per-chirp impulse masking; rejected chirps stay explicit."""

import numpy as np


def mitigate(adc, threshold=8.0, max_fraction=0.05, max_run=8, method="linear"):
    x = np.asarray(adc)
    if x.ndim != 2 or not x.size or np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite real [chirp, sample] ADC")
    if (
        not np.isfinite([threshold, max_fraction, max_run]).all()
        or threshold <= 0
        or not 0 < max_fraction < 1
        or max_run < 1
        or int(max_run) != max_run
        or method not in ("linear", "zero")
    ):
        raise ValueError("Invalid mitigation options")
    x = x.astype(float)
    centre = np.median(x, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(x - centre), axis=1, keepdims=True)
    # A one-count floor is explicit for integer ADC; no auto gain normalization.
    mask = np.abs(x - centre) > threshold * np.maximum(scale, 1.0)
    output = x.copy()
    usable = np.ones(len(x), dtype=bool)
    for row, bad in enumerate(mask):
        indices = np.flatnonzero(bad)
        runs = np.diff(np.r_[-2, indices]) != 1
        lengths = (
            np.diff(np.r_[np.flatnonzero(runs), len(indices)]) if len(indices) else []
        )
        too_long = len(lengths) > 0 and max(lengths) > max_run
        # Edge samples have no two-sided interpolation support.
        usable[row] = (
            bad.mean() <= max_fraction and not too_long and not bad[0] and not bad[-1]
        )
        if not usable[row]:
            output[row] = np.nan
            continue
        if method == "zero":
            output[row, bad] = 0
        elif len(indices):
            good = np.flatnonzero(~bad)
            output[row, bad] = np.interp(indices, good, x[row, good])
    return output, mask, usable
