"""One-command reproduction of the whole noise-floor sensitivity study.

Usage (from the project directory ``community/projects/a100-noise-study``):

    PYTHONPATH=src python3 -m noise_study.cli [--data-dir DIR] [--out-dir DIR]

Defaults resolve to the repository that contains this project, so inside a
checkout of CTSAI-A100 no arguments are needed.  Outputs land in
``figures/*.png`` and ``results/*.{csv,md}`` under the project directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .study import run_all


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="noise_study.cli",
        description="CTSAI-A100 noise floor estimation sensitivity study "
                    "(official calculate_global_noise vs robust alternatives)",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="directory with the adc_*.txt example captures "
                             "(default: <repo>/ADC数据采集/示例adc数据和结果)")
    parser.add_argument("--hxx-dir", type=Path, default=None,
                        help="directory with sensor_config_init{0,1}.hxx "
                             "(default: <repo>/ADC数据采集/.../CTASI-A100配置)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: this project directory)")
    parser.add_argument("--seed", type=int, default=20260929,
                        help="random seed for bootstrap / injections (fixed default)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap resample count (default: 2000)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    print("== a100-noise-study: official noise floor sensitivity study ==")
    print(f"data-dir : {args.data_dir}")
    print(f"hxx-dir  : {args.hxx_dir}")
    print(f"out-dir  : {args.out_dir}")
    print(f"seed     : {args.seed}  n_boot: {args.n_boot}")
    result = run_all(
        data_dir=args.data_dir,
        hxx_dir=args.hxx_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        n_boot=args.n_boot,
    )

    print("\n-- reproduction (official noiseValue, dB) --")
    for row in result["reproduction"]:
        print(f"  {row[0]:5s} {row[1]:>20s}  official_mean={row[3]:8.3f} dB "
              f"median={row[4]:8.3f} dB")
    print("\n-- region scan spread (max-min over scanned regions, dB) --")
    for row in result["region_scan"]:
        print(f"  {row[0]:5s} {row[1]:>13s}  spread={row[5]:6.3f} dB "
              f"(std {row[6]:.3f}, n={row[2]})")
    print("\n-- contamination demo (far, 4ch, synthetic peaks) --")
    for row in result["contamination"]:
        if row[4] in ("official_mean", "median", "mad_floor"):
            print(f"  {row[0]:>20s} {row[4]:>13s}  value={row[5]:8.3f} dB "
                  f"deviation={row[6]:+6.3f} dB")
    print(f"\nsummary written to: {result['summary_path']}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
