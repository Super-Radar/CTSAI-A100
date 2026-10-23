# CTSAI-A100 MATLAB ADC Signal-Processing Pipeline with Independent Validation

A reproducible MATLAB reference implementation that converts the public
CTSAI-A100 ADC text captures into a range spectrum, raw Range-Doppler map and
CA-CFAR diagnostic detections. It deliberately keeps physical velocity and angle
as `NaN` because the public DDMA phase/offset/channel metadata is incomplete, and
it ships an independent NumPy validator that mirrors the same equations.

## 1. What this project does

- reads all four public RX files and validates their lengths;
- unpacks two signed 16-bit ADC samples from each stored 32-bit word;
- loads the public near/far HXX profile parameters;
- performs DC removal, Hann windowing, Range FFT and raw Doppler FFT;
- integrates the four RX channels noncoherently;
- detects targets with a toolbox-independent 2-D CA-CFAR implementation;
- treats the raw Doppler FFT as periodic during CFAR and applies the range ROI
  only after noise estimation;
- exports raw Doppler bins while DDMA metadata is unavailable;
- keeps physical velocity and angle as `NaN` instead of reporting undecoded data;
- exports PNG figures, a CSV target table, a configuration report and a MAT file;
- provides an independent NumPy validation of the same HXX/ADC/FFT/CFAR math.

## 2. Environment

- Hardware: not required for the committed public-data example;
- Configuration: official `sensor_config_init0.hxx` / `sensor_config_init1.hxx`;
- MATLAB R2020b or later (only base MATLAB functions are used);
- Python 3.10 (NumPy 2.2, Pillow 12.3) for the independent validator.

The reference figures and CSV files were regenerated after correcting ADC sample
alignment, using Python 3.10.9, NumPy 1.26.4 and Pillow 9.5.0. See
`validator/results/adc-alignment-validation.txt` for sample-level checks.
The earlier MATLAB R2026a run in `validator/results/matlab-r2026a-validation.txt`
predates this correction: agreement with the old NumPy loader did not detect
their shared one-word shift. It does not validate the corrected outputs.
Native MATLAB `main('near')` / `main('far')` revalidation is still pending.
No radar hardware test is claimed.

## 3. Data

The pipeline reads the public ADC captures and HXX profiles that already ship
with this repository:

- ADC text files: `ADC数据采集/示例adc数据和结果/`
- HXX profiles: `ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/cfg/CTASI-A100配置/`

This project does not bundle or redistribute any private data. Use your own
captures in place of the public ones if you do not have access to them.

The `*_f0_*` / `*_f1_*` captures above are headerless comma-separated records:
131072 packed ADC words followed by one extra zero. The loader also accepts the
exact payload without that extra word. It preserves the first word and zeros
inside the payload; only a single zero beyond the expected length is removed.
Every field must be a finite unsigned 32-bit integer. Truncated records, invalid
fields and other trailers are rejected instead of shifting or padding samples.
This loader does not infer headers: the separate `adc_test_*_Pf0_Rx*.txt`
three-field-header format is not supported. Custom captures must follow the
documented headerless layout and selected profile; length alone cannot prove
capture completeness or distinguish the two profiles.

No slow-time mean subtraction or center-bin suppression is applied to the raw
Doppler spectrum. With unresolved DDMA phase coding, a stationary target's TX
components need not occupy the center bin, so suppressing that bin could remove
valid signal.

## 4. Project layout

```text
community/projects/a100-matlab-adc-pipeline-validation/
├── README.md
├── main.m
├── src/
│   ├── ctsai_config.m
│   ├── load_hxx_profile.m
│   ├── load_ctsai_adc.m
│   ├── process_radar_cube.m
│   ├── ca_cfar_2d.m
│   ├── estimate_angle_fft.m
│   ├── build_detection_table.m
│   └── plot_results.m
├── validator/
│   ├── README.md
│   ├── run_reference_validation.py
│   └── results/{near,far}/
└── docs/
    └── CTSAI-A100_MATLAB_ADC_processing.md
```

## 5. How to run

Make this directory the current directory (or add it to the MATLAB path), then
run:

```matlab
main('near');
main('far');
```

Results are written to `results/near/` and `results/far/`:

- `01_range_spectrum.png`
- `02_raw_range_doppler_map.png`
- `03_raw_cfar_detections.png`
- `04_processing_status.png`
- `detections.csv`
- `configuration_report.txt`
- `processing_result.mat`

To re-run the independent numerical validation:

```powershell
python "community/projects/a100-matlab-adc-pipeline-validation/validator/run_reference_validation.py"
```

Check exact ADC sample order, signed conversion, chirp/channel boundaries and
rejection of malformed input (run from the repository root):

```powershell
python -m unittest discover -s "community/projects/a100-matlab-adc-pipeline-validation/validator" -p test_adc_loading.py -v
```

In MATLAB, from the project directory:

```matlab
addpath('tests');
run_adc_loading_tests();
```

Both test entries check all eight public files against independent arithmetic
unpacking, in addition to small hand-decoded fixtures.

## 6. Results

Committed reference figures for both the near and far profiles are under
`validator/results/`. The `detections.csv` tables retain raw Doppler bins and
leave `velocity_mps` / `angle_deg` empty with `kinematics_valid=False`.

## 7. Notes

- The public capture is processed as one frame; multi-frame tracking is not
  implemented.
- DDMA decoding and calibrated virtual-array angle output are disabled until the
  required metadata is publicly available.
- Static-clutter suppression is disabled while the DDMA zero-velocity bin
  locations are unresolved.
- This example does not implement proprietary interference cancellation, phase
  calibration, velocity ambiguity resolution or product firmware logic.

See `docs/CTSAI-A100_MATLAB_ADC_processing.md` for algorithm details.
