# Independent numerical validation

`run_reference_validation.py` loads the repository HXX files and mirrors the
guarded MATLAB ADC unpacking, Range FFT, raw Doppler FFT, four-RX integration
and CA-CFAR equations with NumPy.

The validation deliberately does not turn unresolved DDMA bins into physical
velocity or angle. Generated CSV files retain raw Doppler bins and leave
`velocity_mps` and `angle_deg` empty with `kinematics_valid=False`.
It also performs no slow-time mean subtraction or center-bin suppression,
because unresolved DDMA coding means the combined spectrum center is not a
known physical zero-velocity location.

The validator includes a regression check for a target on the periodic Doppler
boundary. Range-of-interest filtering is applied only after the CFAR noise map
has been estimated.

Run from the repository root:

```powershell
python "community/projects/a100-matlab-adc-pipeline-validation/validator/run_reference_validation.py"
```

Validated environment for the committed results:

- Python 3.10.9
- NumPy 1.26.4
- Pillow 9.5.0

Expected cube dimensions:

| Profile | HXX source | ADC cube | Raw Range-Doppler cube |
|---|---|---:|---:|
| near | `sensor_config_init1.hxx` | 2048 x 128 x 4 | 1024 x 128 x 4 |
| far | `sensor_config_init0.hxx` | 1024 x 256 x 4 | 512 x 256 x 4 |

## ADC alignment regression

The public files contain 131072 payload words plus one trailing zero, not a
leading metadata word. Both loaders now retain the first word and reject
invalid fields instead of deleting them. Run `test_adc_loading.py` with unittest
from the repository root as documented in the project README. It checks signed
samples, chirp/channel order, optional trailing zero and all eight public files
against independent scalar integer unpacking. The MATLAB test entry is
`tests/run_adc_loading_tests.m`.

The regenerated near/far tables still contain 32/3 detections at the same cells,
but their powers change by up to 0.012981/0.010186 dB. Unchanged detection cells
do not establish sample alignment. See `results/adc-alignment-validation.txt`.

## Historical native MATLAB cross-validation

MATLAB R2026a completed `main('near')` and `main('far')` on a GitHub-hosted
Ubuntu runner. The resulting detection rows were compared against these NumPy
reference CSV files: raw Doppler bins matched exactly, while maximum range and
power differences were below `5e-14 m` and `5e-13 dB`. Run metadata and the
public workflow URL are recorded in `results/matlab-r2026a-validation.txt`.
That run predates the ADC alignment correction and compared two loaders with
the same shift. The log is retained as historical evidence; it is not a native
MATLAB validation of the corrected outputs. Full MATLAB main revalidation is
pending.
