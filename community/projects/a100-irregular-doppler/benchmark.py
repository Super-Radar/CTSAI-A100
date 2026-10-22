"""Timestamp-aware projection versus compressed-index FFT and zero-fill oracle."""

import json
from pathlib import Path

import numpy as np

from adc_io import load, sources
from spectrum import sampling_window, spectrum

ROOT = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(20261022)
    n = 256
    f = np.arange(-128, 128) / 256
    synthetic = []
    first = None
    for jitter in (0.0, 0.3):
        errors = []
        compressed = []
        for _ in range(40):
            keep = np.sort(rng.choice(n, 192, replace=False))
            t = np.arange(n, dtype=float) + rng.uniform(-jitter, jitter, n)
            truth = 37 / 256
            x = np.exp(2j * np.pi * truth * t[keep])
            aware = spectrum(x, t[keep], f)
            naive = np.fft.fftshift(np.fft.fft(x, n)) / len(x)
            errors.append(abs(f[np.argmax(abs(aware))] - truth))
            compressed.append(abs(f[np.argmax(abs(naive))] - truth))
            if jitter and first is None:
                first = (aware, naive)
        synthetic.append(
            {
                "jitter_periods": jitter,
                "runs": 40,
                "truth_cycles_per_period": 37 / 256,
                "timestamp_peak_error_median": float(np.median(errors)),
                "compressed_fft_peak_error_median": float(np.median(compressed)),
                "timestamp_max_error": float(max(errors)),
            }
        )
    real = []
    for path, row in sources(ROOT):
        adc, meta = load(path, row["sha256"])
        count = len(adc)
        times = np.arange(count)
        range_fft = np.fft.rfft(adc * np.hanning(adc.shape[1]), axis=1)
        # Choose the strongest demeaned non-DC range bin without scenario labels.
        centred = range_fft - range_fft.mean(axis=0)
        energies = np.mean(abs(centred) ** 2, axis=0)
        energies[0] = -1
        bin_index = int(np.argmax(energies))
        x = centred[:, bin_index]
        freq = np.fft.fftshift(np.fft.fftfreq(count))
        keep = np.sort(rng.choice(count, 3 * count // 4, replace=False))
        window = np.hanning(count)[keep]
        aware = spectrum(x[keep], times[keep], freq, window)
        zero = np.zeros(count, dtype=complex)
        zero[keep] = x[keep] * window
        oracle = np.fft.fftshift(np.fft.fft(zero)) / window.sum()
        oracle *= np.exp(
            2j * np.pi * freq * times[keep[0]]
        )  # Same relative time origin as spectrum().
        relative = float(np.max(abs(aware - oracle)) / max(np.max(abs(oracle)), 1e-12))
        assert relative < 1e-10
        ref = (
            np.fft.fftshift(np.fft.fft(x * np.hanning(count))) / np.hanning(count).sum()
        )
        real.append(
            {
                "input": meta,
                "range_bin": bin_index,
                "kept_chirps": len(keep),
                "zero_fill_oracle_relative_error": relative,
                "full_peak_cycles_per_period": float(freq[np.argmax(abs(ref))]),
                "missing_peak_cycles_per_period": float(freq[np.argmax(abs(aware))]),
            }
        )
    alias = sampling_window(np.arange(n)[::2], [-0.5, 0, 0.5])
    report = {
        "seed": 20261022,
        "synthetic": synthetic,
        "real_integer_index_replays": real,
        "regular_decimation_window_power": alias.tolist(),
        "caveat": "Real capture indices are assumed nominal uniform periods; masks/jitter are injected, not measured timing. No physical velocity/DDMA inference. Integer-grid missing samples equal correctly normalized zero-fill FFT; NUDFT is useful for genuinely nonuniform timestamps, not an accuracy upgrade for uniform data.",
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for y, label in zip(first, ["Timestamp-aware", "Compressed-index FFT"]):
        axes[0].plot(f, abs(y) ** 2, label=label)
    axes[0].axvline(37 / 256, color="gray", ls="--")
    axes[0].set(
        xlabel="Cycles / nominal period",
        ylabel="Projection power",
        title="Synthetic known tone, missing + jitter",
    )
    axes[0].legend()
    grid = np.linspace(-0.5, 0.5, 1025)
    axes[1].plot(grid, sampling_window(np.arange(n)[::2], grid))
    axes[1].set(
        xlabel="Frequency offset",
        ylabel="Sampling-window power",
        title="Every-other-chirp sampling retains exact aliases",
    )
    fig.savefig(out / "spectrum.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {
                "synthetic": synthetic,
                "real_files": len(real),
                "max_oracle_error": max(
                    r["zero_fill_oracle_relative_error"] for r in real
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
