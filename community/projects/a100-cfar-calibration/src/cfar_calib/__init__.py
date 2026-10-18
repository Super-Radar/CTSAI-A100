"""CA-CFAR false-alarm calibration study for the CTSAI-A100 community pipeline.

Monte-Carlo quantification of how the single-look CA-CFAR threshold factor
deviates from its nominal false-alarm rate once it is applied to a
K-channel non-coherently integrated (power-averaged, dB-displayed)
Range-Doppler spectrum produced by the official window + 2D-FFT chain.
"""

__all__ = ["cfar", "pipeline", "calibration", "cli"]
__version__ = "1.0.0"
