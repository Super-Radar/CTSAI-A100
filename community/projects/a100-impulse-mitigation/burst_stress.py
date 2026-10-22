"""Burst geometry and rejected-chirp coverage over the same public ADC files."""

import csv
import json
from pathlib import Path

import numpy as np

from adc_io import load, sources
from benchmark import error
from mitigate import mitigate

ROOT = Path(__file__).resolve().parent
SEED = 20261022
CASES = [(f"interior_{width}", width, 20000) for width in (1, 3, 8, 9, 16)] + [
    ("left_edge", 3, 20000),
    ("right_edge", 3, 20000),
    ("distributed_6pct", 0, 20000),
    ("half_chirp", 0, 20000),
    ("weak_interior_3", 3, 200),
]


def positions(name, width, samples):
    if name == "left_edge":
        return np.arange(width)
    if name == "right_edge":
        return np.arange(samples - width, samples)
    if name == "distributed_6pct":
        return np.linspace(1, samples - 2, int(np.ceil(samples * 0.06)), dtype=int)
    if name == "half_chirp":
        return np.arange(samples // 4, samples // 4 + samples // 2)
    start = (samples - width) // 2
    return np.arange(start, start + width)


def inject(original, chosen, indices, amplitude):
    damaged = original.copy()
    truth = np.zeros_like(original, dtype=bool)
    truth[np.ix_(chosen, indices)] = True
    damaged[truth] = np.clip(damaged[truth] + amplitude, -32768, 32767)
    return damaged, truth


def main():
    reports, details, inputs = [], [], []
    for path, entry in sources(ROOT):
        original, meta = load(path, entry["sha256"])
        inputs.append(meta)
        chosen = np.sort(
            np.random.default_rng(SEED).choice(len(original), 32, replace=False)
        )
        for name, width, amplitude in CASES:
            damaged, truth = inject(
                original, chosen, positions(name, width, original.shape[1]), amplitude
            )
            linear, mask, usable = mitigate(damaged)
            zero, zero_mask, zero_usable = mitigate(damaged, method="zero")
            if not np.array_equal(mask, zero_mask) or not np.array_equal(
                usable, zero_usable
            ):
                raise RuntimeError("Methods changed the comparison population")
            target = np.isin(np.arange(len(original)), chosen)
            common = target & usable
            if not np.isnan(linear[~usable]).all():
                raise RuntimeError("Rejected chirps lost their explicit NaN marker")
            reports.append(
                {
                    "source": meta["source"],
                    "case": name,
                    "amplitude": amplitude,
                    "injected_chirps": int(target.sum()),
                    "injected_samples": int(truth.sum()),
                    "changed_samples": int(np.count_nonzero(damaged != original)),
                    "detected_injected_samples": int((mask & truth).sum()),
                    "flagged_outside_injection": int((mask & ~truth).sum()),
                    "retained_injected_chirps": int(common.sum()),
                    "rejected_other_chirps": int((~usable & ~target).sum()),
                    "damaged": error(original, damaged, common),
                    "zero": error(original, zero, common),
                    "linear": error(original, linear, common),
                }
            )
            for index in chosen:
                details.append(
                    {
                        "source": meta["source"],
                        "case": name,
                        "chirp_order": int(index),
                        "usable": bool(usable[index]),
                        "injected_samples": int(truth[index].sum()),
                        "detected_samples": int((mask[index] & truth[index]).sum()),
                        "flagged_outside": int((mask[index] & ~truth[index]).sum()),
                    }
                )
    aggregate = []
    for name, _, _ in CASES:
        runs = [r for r in reports if r["case"] == name]
        aggregate.append(
            dict(
                case=name,
                files=len(runs),
                **{
                    key: sum(r[key] for r in runs)
                    for key in (
                        "injected_chirps",
                        "retained_injected_chirps",
                        "injected_samples",
                        "detected_injected_samples",
                        "flagged_outside_injection",
                        "rejected_other_chirps",
                    )
                },
                median_spectrum_error={
                    method: float(
                        np.median(
                            [
                                r[method]["relative_complex_spectrum_l2"]
                                for r in runs
                                if r[method] is not None
                            ]
                        )
                    )
                    if any(r[method] is not None for r in runs)
                    else None
                    for method in ("damaged", "zero", "linear")
                },
            )
        )
    out = ROOT / "results/burst-stress"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": SEED,
        "inputs": inputs,
        "cases": aggregate,
        "runs": reports,
        "limits": "Synthetic positive additive bursts clipped to int16 over public recorded ADC. Original samples are a numeric reference, not interference-free truth. Error methods share retained injected chirps; no retained chirps means null, never zero error. Rejected chirps retain original indices; coverage is reported separately. No real RF interference or detection performance claim.",
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (out / "chirps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    x = np.arange(len(aggregate))
    axes[0].bar(
        x, [r["retained_injected_chirps"] / r["injected_chirps"] for r in aggregate]
    )
    axes[0].set(
        ylabel="Retained injected chirps / all injected",
        ylim=(0, 1.05),
        title="Coverage and injected-sample detection are separate outcomes",
    )
    axes[1].bar(
        x, [r["detected_injected_samples"] / r["injected_samples"] for r in aggregate]
    )
    axes[1].set(ylabel="Flagged injected samples / injected", ylim=(0, 1.05))
    for ax in axes:
        ax.set_xticks(x, [r["case"].replace("_", "\n") for r in aggregate], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "coverage.png", dpi=140)
    plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
