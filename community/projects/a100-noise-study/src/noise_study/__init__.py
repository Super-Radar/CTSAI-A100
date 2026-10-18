"""Noise floor estimation sensitivity study for the CTSAI-A100 radar.

Reproduces the official ``calculate_global_noise.m`` estimator, quantifies its
sensitivity to noise-region choice, and compares robust alternatives
(median / quantiles / trimmed mean / MAD) on the repository's public
ADC captures.
"""

__version__ = "1.0.0"
