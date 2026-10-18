"""End-to-end: injected point target survives the chain and the detector."""

import numpy as np

from cfar_calib.calibration import inject_sinusoid_target, make_rng
from cfar_calib.cfar import alpha_from_pfa_single_look, cfar_detect, kernel_count
from cfar_calib.pipeline import RadarConfig, range_doppler_power, synth_real_noise

CFG = RadarConfig(n_chirps=64, adc_sample_num=256, rng_nfft=256, vel_nfft=64)


def test_injected_target_detected_and_alarms_bounded():
    rng = make_rng(31, 99)
    adc = synth_real_noise(rng, CFG, n_channels=4, rho=0.0)
    rng_bin, vel_bin = 100, 40          # interior cell (border is 10 rows)
    amp = 25.0 * adc.std()              # ~28 dB above noise floor after FFT gain
    adc_t = inject_sinusoid_target(adc, CFG, rng_bin, vel_bin, amp)
    power = range_doppler_power(adc_t, CFG)
    assert power.shape == (CFG.rng_bins, CFG.vel_nfft)

    alpha = alpha_from_pfa_single_look(1e-4, kernel_count())
    mask, _ = cfar_detect(power, alpha)
    assert mask[rng_bin, vel_bin], "injected target not detected"
    others = int(mask.sum()) - 1
    n_cells = int(np.prod(mask.shape))
    bound = 5e-3 * n_cells               # far above nominal, catches blow-ups
    assert others <= bound, f"{others} extra alarms in {n_cells} cells"

    # same frame without the target must stay near the nominal alarm count
    power_n = range_doppler_power(adc, CFG)
    mask_n, _ = cfar_detect(power_n, alpha)
    assert int(mask_n.sum()) <= bound
