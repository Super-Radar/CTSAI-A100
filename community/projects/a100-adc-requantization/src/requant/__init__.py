"""a100-adc-requantization: spectral fidelity budget for lossy ADC requantization.

Sub-package modules
-------------------
config     : radar frame configurations (init0 / init1 / synthetic).
pipeline   : official-equivalent unpack + window + 2D-FFT spectral chain.
requantize : fixed full-scale int16 bit-reduction (round-half-away / trunc).
fidelity   : time / spectral error, band-energy retention, peak drift metrics.
synthetic  : synthetic int16 benchmarks (tones, two-tone, band-limited noise).
figures    : the five report figures (matplotlib, Agg backend).
cli        : one-click reproduction with fixed seed 20261003.
"""

__version__ = "1.0.0"

SEED = 20261003
