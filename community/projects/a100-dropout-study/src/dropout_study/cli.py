"""One-click reproduction entry point.

Usage (from the project directory ``community/projects/a100-dropout-study``)::

    PYTHONPATH=src python3 -m dropout_study.cli            # full run (~1.5-2 min)
    PYTHONPATH=src python3 -m dropout_study.cli --quick    # trimmed grids (~1/4 of the work)

Writes ``results/*.csv``, ``results/summary.md`` and ``figures/*.png`` next to
this package, with fixed random seeds so every run produces identical output.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

from .study import run_all  # noqa: E402  (import after backend selection)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dropout-study",
        description="ADC dropout spectral-impact study for CTSAI-A100",
    )
    parser.add_argument("--project-dir", default=None,
                        help="project root that holds results/ and figures/ "
                             "(default: the directory containing this package)")
    parser.add_argument("--data-dir", default=None,
                        help="directory with the official ADC sample txt files "
                             "(default: <repo>/ADC数据采集/示例adc数据和结果)")
    parser.add_argument("--quick", action="store_true",
                        help="trimmed severity grids and fewer repetitions")
    args = parser.parse_args(argv)

    summary = run_all(project_dir=args.project_dir, quick=args.quick,
                      data_dir=args.data_dir)

    kn = summary["key_numbers"]
    print()
    print("== key numbers (real far wave, 4-channel average, medians) ==")
    print(f"  random word drop 0.5% / 1% -> peak power error: "
          f"{kn['far_random_05pct_power_db']:.2f} / {kn['far_random_1pct_power_db']:.2f} dB")
    print(f"  8 chirps zeroed            -> peak power error: "
          f"{kn['far_chirp8_power_db']:.2f} dB")
    print(f"  tail 5% zero-fill          -> {kn['far_tail5_zerofill_power_db']:.2f} dB, "
          f"drift {kn['far_tail5_zerofill_drift']:.2f} bin")
    print(f"  tail 5% reject-incomplete  -> {kn['far_tail5_reject_power_db']:.2f} dB, "
          f"drift {kn['far_tail5_reject_drift']:.2f} bin")
    print(f"  exp1 zero-perturbation: {summary['exp1_passed']} checks passed")
    print(f"  total wall time: {summary['elapsed_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
