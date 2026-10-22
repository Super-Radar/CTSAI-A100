"""Injected interference on real ADC baselines; no hardware interference claim."""

import json
from pathlib import Path

import numpy as np

from adc_io import load, sources
from mitigate import mitigate

ROOT = Path(__file__).resolve().parent


def error(reference, value, usable):
    if not usable.any():
        return None
    x = reference[usable]
    y = value[usable]
    window = np.hanning(x.shape[1])
    a = np.fft.rfft(x * window)
    b = np.fft.rfft(y * window)
    return {
        "relative_time_l2": float(
            np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-12)
        ),
        "relative_complex_spectrum_l2": float(
            np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)
        ),
    }


def main():
    reports = []
    for path, row in sources(ROOT):
        original, meta = load(path, row["sha256"])
        clean, clean_mask, clean_usable = mitigate(original)
        runs = []
        for run in range(20):
            seed = run % 5
            amplitude = (200, 2000, 8000, 20000)[run // 5]
            rng = np.random.default_rng(20261022 + seed)
            damaged = original.copy()
            truth = np.zeros_like(original, dtype=bool)
            chosen = rng.choice(len(original), 32, replace=False)
            for chirp in chosen:
                start = int(rng.integers(20, original.shape[1] - 23))
                truth[chirp, start : start + 3] = True
                damaged[chirp, start : start + 3] = np.clip(
                    damaged[chirp, start : start + 3]
                    + rng.choice([-amplitude, amplitude]),
                    -32768,
                    32767,
                )
            linear, mask, usable = mitigate(damaged)
            zero, _, zero_usable = mitigate(damaged, method="zero")
            assert np.array_equal(usable, zero_usable)
            # Every method uses the same retained chirps, not different denominators.
            target_rows = usable & np.isin(np.arange(len(original)), chosen)
            detected = int((mask & truth).sum())
            flagged = int(mask.sum())
            runs.append(
                {
                    "seed": 20261022 + seed,
                    "injection_counts": amplitude,
                    "injected_samples": int(truth.sum()),
                    "detected_injected_samples": detected,
                    "flagged_outside_injection": int((mask & ~truth).sum()),
                    "flagged_samples": flagged,
                    "total_chirps": len(original),
                    "usable_chirps": int(usable.sum()),
                    "compared_injected_chirps": int(target_rows.sum()),
                    "damaged": error(original, damaged, target_rows),
                    "zero": error(original, zero, target_rows),
                    "linear": error(original, linear, target_rows),
                }
            )
        reports.append(
            {
                "input": meta,
                "clean_flagged_samples": int(clean_mask.sum()),
                "clean_usable_chirps": int(clean_usable.sum()),
                "clean_effect": error(original, clean, clean_usable),
                "runs": runs,
            }
        )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary = {
        "parameters": {
            "threshold_mad": 8,
            "max_fraction": 0.05,
            "max_run": 8,
            "injection_counts": [200, 2000, 8000, 20000],
            "injected_run": 3,
            "chirps_per_run": 32,
        },
        "captures": reports,
        "caveat": "Original ADC is a numeric reference, not interference-free ground truth. Injected pulses are synthetic. Rejected chirps are NaN and must not be silently FFT processed; long bursts and edge pulses are not interpolated.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for method in ("damaged", "zero", "linear"):
        values = [
            np.median(
                [
                    r[method]["relative_complex_spectrum_l2"]
                    for r in p["runs"]
                    if r[method] and r["injection_counts"] == 20000
                ]
            )
            for p in reports
        ]
        ax.plot(range(1, len(reports) + 1), values, "o-", label=method)
    ax.set(
        xlabel="Manifest ADC file order",
        ylabel="Relative complex range-spectrum error",
        yscale="log",
        title="20000-count injection; matched chirps, median of five seeds",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(out / "mitigation.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {
                "captures": len(reports),
                "clean_flagged": sum(p["clean_flagged_samples"] for p in reports),
                "median_errors": {
                    m: float(
                        np.median(
                            [
                                r[m]["relative_complex_spectrum_l2"]
                                for p in reports
                                for r in p["runs"]
                                if r[m]
                            ]
                        )
                    )
                    for m in ("damaged", "zero", "linear")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
