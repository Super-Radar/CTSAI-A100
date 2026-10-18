"""profile_sensitivity -- near/far profile mismatch study for CTSAI-A100 ADC data.

The package reproduces the official MATLAB processing chain
(``adc_mem_2_real_fixed.m`` / ``load_adc_data.m`` / ``do_fft_2d.m`` /
``get_radar_paramete.m``) in NumPy, then deliberately crosses the two sensor
profiles (``init0`` far wave / ``init1`` near wave) against both public
captures to quantify what happens when the wrong configuration is used, and
ships a heuristic detector that identifies which profile a capture belongs to.
"""

__version__ = "1.0.0"

from . import profiles, pipeline, mismatch, detector  # noqa: F401
