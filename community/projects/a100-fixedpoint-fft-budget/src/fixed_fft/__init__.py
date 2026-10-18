"""a100-fixedpoint-fft-budget: fixed-point FFT wordlength and scaling budget study.

Offline, machine-independent Python analysis for CTSAI-A100 radar data.
"""

from .fft_fixed import (
    FixedFFTConfig,
    FixedFFTResult,
    SCALING_MODES,
    SCALING_NONE,
    SCALING_SCALE_1BIT,
    OVERFLOW_MODES,
    OVERFLOW_SATURATE,
    OVERFLOW_WRAP,
    clip_to_word,
    fft_fixed,
    fit_input,
    quantized_twiddles,
    twiddle_quantization_error,
)

__version__ = "1.0.0"

__all__ = [
    "FixedFFTConfig",
    "FixedFFTResult",
    "SCALING_MODES",
    "SCALING_NONE",
    "SCALING_SCALE_1BIT",
    "OVERFLOW_MODES",
    "OVERFLOW_SATURATE",
    "OVERFLOW_WRAP",
    "clip_to_word",
    "fft_fixed",
    "fit_input",
    "quantized_twiddles",
    "twiddle_quantization_error",
    "__version__",
]
