"""Controlled ADC word-stream corruption models and the two mitigation strategies.

Every model operates on the per-chirp word matrix ``[nchirp, words_per_chirp]``
(the payload after the official length normalisation) and returns a pair
``(corrupted_words, missing_mask)``:

* ``corrupted_words`` has the same shape as the input, with missing words
  already zero-filled (this is exactly what the positional-zero-fill strategy
  consumes);
* ``missing_mask`` marks the corrupted positions (oracle knowledge of *where*
  the corruption happened - the study assumes the gap locations are known, as
  they would be from a transport layer with sequence numbers).

Models (severity semantics):

===================  =========================================================
model                severity
===================  =========================================================
``random_drop``      float fraction of words dropped at uniformly random
                     positions, 0-0.2
``zero_segment``     int length in words of one contiguous zeroed run,
                     1-64, positioned at head / middle / tail of the stream
``chirp_zero``       int count of whole chirps zeroed (uniformly random
                     distinct chirps), 1-32
``tail_trunc``       float fraction of the word stream absent from the tail
                     (collection ended early), 0.005-0.1
===================  =========================================================

``severity == 0`` is an exact identity for every model: the returned words
equal the input and the mask is all-False (this is the zero-perturbation
regression guarantee the tests rely on).

Two mitigation strategies are provided for the downstream chain:

* ``zerofill`` (official-style positional zero-fill): keep all chirps, zeros
  stay where words went missing.  For tail truncation this coincides with the
  official ``load_adc_data.m`` behaviour of zero-padding a short file.
* ``reject`` (truncate to complete-chirp boundaries / refuse incomplete
  data): drop every chirp that contains at least one missing word and process
  only the surviving chirps.  For tail truncation this reduces to keeping
  ``floor(n_received / words_per_chirp)`` complete chirps.  Chirps survive in
  their original order; the Doppler FFT still assumes a uniform chirp grid,
  so interior gaps introduce a phase discontinuity for moving targets (an
  honestly measured effect, see README).
"""

from __future__ import annotations

import numpy as np

from .pipeline import unpack_words

MODELS = ("random_drop", "zero_segment", "chirp_zero", "tail_trunc")

STRATEGIES = ("zerofill", "reject")

MODEL_LABEL = {
    "random_drop": "random word drop",
    "zero_segment": "contiguous zero segment",
    "chirp_zero": "whole-chirp zeroing",
    "tail_trunc": "tail truncation",
}

MODEL_AXIS = {
    "random_drop": "dropped word fraction",
    "zero_segment": "zero-segment length (words)",
    "chirp_zero": "zeroed chirp count",
    "tail_trunc": "truncated tail fraction",
}


def corrupt_words(words, model, severity, rng, position: str = "middle"):
    """Apply one corruption model; returns ``(corrupted_words, missing_mask)``.

    ``rng`` must be a ``numpy.random.Generator``.  ``position`` is only used
    by ``zero_segment`` ("head" / "middle" / "tail").  Severity 0 (int or
    float) returns the input unchanged with an all-False mask.
    """
    w = np.array(words, dtype=np.uint32, copy=True)
    mask = np.zeros(w.shape, dtype=bool)
    if severity == 0:
        return w, mask
    nchirp, wpc = w.shape
    n_total = w.size
    if model == "random_drop":
        k = int(round(float(severity) * n_total))
        if k > 0:
            idx = rng.choice(n_total, size=k, replace=False)
            mask.reshape(-1)[idx] = True
    elif model == "zero_segment":
        length = int(severity)
        if length > 0:
            if position == "head":
                start = 0
            elif position == "middle":
                start = (n_total - length) // 2
            elif position == "tail":
                start = n_total - length
            else:
                raise ValueError(f"unknown position {position!r}")
            mask.reshape(-1)[start : start + length] = True
    elif model == "chirp_zero":
        k = int(severity)
        if k > 0:
            rows = rng.choice(nchirp, size=k, replace=False)
            mask[rows, :] = True
    elif model == "tail_trunc":
        k = int(round(float(severity) * n_total))
        if k > 0:
            mask.reshape(-1)[n_total - k :] = True
    else:
        raise ValueError(f"unknown corruption model {model!r}")
    w[mask] = 0
    return w, mask


def chirp_keep_mask(masks) -> np.ndarray:
    """Chirps intact in every channel (intersection of per-channel masks)."""
    masks = [np.asarray(m) for m in masks]
    keep = np.ones(masks[0].shape[0], dtype=bool)
    for m in masks:
        keep &= ~m.any(axis=1)
    return keep


def apply_strategy(words, mask, strategy: str):
    """Build the per-chirp sample frame for one channel under one strategy.

    Returns ``(frame_or_None, info)`` where ``frame_or_None`` is an int16
    matrix ``[nchirp_used, samples_per_chirp]`` (None when the strategy leaves
    no usable chirp) and ``info`` records ``nchirp_used`` and
    ``n_chirps_rejected``.
    """
    words = np.asarray(words)
    mask = np.asarray(mask)
    if strategy == "zerofill":
        frame = unpack_words(words)
        return frame, {"nchirp_used": frame.shape[0], "n_chirps_rejected": 0}
    if strategy == "reject":
        keep = ~mask.any(axis=1)
        n_used = int(keep.sum())
        if n_used == 0:
            return None, {"nchirp_used": 0, "n_chirps_rejected": int(words.shape[0])}
        frame = unpack_words(words[keep])
        return frame, {"nchirp_used": n_used, "n_chirps_rejected": int(words.shape[0] - n_used)}
    raise ValueError(f"unknown strategy {strategy!r}")


__all__ = [
    "MODELS", "STRATEGIES", "MODEL_LABEL", "MODEL_AXIS",
    "corrupt_words", "chirp_keep_mask", "apply_strategy",
]
