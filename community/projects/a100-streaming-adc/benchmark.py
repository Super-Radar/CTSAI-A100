"""Correctness on ten captures and fresh-process RSS/time measurements."""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stream_adc import range_profile, sha256


def batch(path):
    values = np.fromstring(
        Path(path).read_text().strip().rstrip(","), sep=",", dtype=np.uint32
    )
    _, samples, chirps = map(int, values[:3])
    # Independent high-half-first signed view, not the streaming unpacker.
    adc = (
        values[3 : 3 + samples * chirps // 2]
        .astype(">u4")
        .view(">i2")
        .astype(float)
        .reshape(chirps, samples)
    )
    return np.mean(np.abs(np.fft.rfft(adc * np.hanning(samples), axis=1)) ** 2, axis=0)


def worker(method, path):
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    done = threading.Event()

    def sample():
        while not done.wait(0.001):
            samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = range_profile(path)[0] if method == "stream" else batch(path)
    seconds = time.perf_counter() - start
    samples.append(process.memory_info().rss)
    done.set()
    sampler.join()
    return {
        "seconds": seconds,
        "baseline_rss_bytes": baseline,
        "sampled_peak_rss_bytes": max(samples),
        "result_sum": float(result.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["batch", "stream"])
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.capture)))
        return
    data = ROOT.parent / "a100-indoor-cart-mti"
    rows = list(csv.DictReader((data / "data/manifest.csv").open()))
    verified = []
    for row in rows:
        if row["data_type"] != "adc_txt":
            continue
        path = data / row["relative_path"]
        assert sha256(path) == row["sha256"]
        streamed, _ = range_profile(path)
        expected = batch(path)
        error = float(np.max(np.abs(streamed - expected)) / max(np.max(expected), 1))
        assert error < 1e-12
        verified.append(
            {
                "file": row["relative_path"],
                "sha256": row["sha256"],
                "relative_error": error,
            }
        )
    measurements = []
    source = data / verified[0]["file"]
    raw = np.fromstring(
        source.read_text().strip().rstrip(","), sep=",", dtype=np.uint32
    )
    channel, samples, chirps = map(int, raw[:3])
    text = ",".join(map(str, raw[3 : 3 + samples * chirps // 2]))
    with tempfile.TemporaryDirectory() as tmp:
        for factor in (1, 4, 16):
            capture = Path(tmp) / f"repeated-{factor}.txt"
            with capture.open("w") as stream:
                stream.write(f"{channel},{samples},{chirps * factor},")
                for _ in range(factor):
                    stream.write(text + ",")
                stream.write("0")
            for method in ("batch", "stream"):
                runs = []
                for _ in range(3):
                    output = subprocess.check_output(
                        [
                            sys.executable,
                            __file__,
                            "--worker",
                            method,
                            "--capture",
                            str(capture),
                        ],
                        text=True,
                    )
                    runs.append(json.loads(output))
                measurements.append(
                    {
                        "repetition_factor": factor,
                        "file_bytes": capture.stat().st_size,
                        "method": method,
                        "median_seconds": statistics.median(r["seconds"] for r in runs),
                        "max_sampled_peak_rss_bytes": max(
                            r["sampled_peak_rss_bytes"] for r in runs
                        ),
                        "runs": runs,
                    }
                )
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "numpy": np.__version__,
        "verified_captures": verified,
        "benchmark": measurements,
        "method": "Fresh process per run; RSS sampled at 1ms, includes interpreter/NumPy and may miss short peaks; 3 runs. Repeated payloads are synthetic scaling inputs, not new captures.",
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method in ("batch", "stream"):
        selected = [r for r in measurements if r["method"] == method]
        x = [r["file_bytes"] / 1e6 for r in selected]
        axes[0].plot(
            x,
            [r["max_sampled_peak_rss_bytes"] / 1e6 for r in selected],
            "o-",
            label=method,
        )
        axes[1].plot(x, [r["median_seconds"] for r in selected], "o-", label=method)
    axes[0].set(xlabel="Repeated capture size (MB)", ylabel="Sampled total RSS (MB)")
    axes[1].set(xlabel="Repeated capture size (MB)", ylabel="Median elapsed (s)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(out / "memory-time.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {
                "real_captures": len(verified),
                "measurements": [
                    {
                        k: r[k]
                        for k in (
                            "repetition_factor",
                            "method",
                            "median_seconds",
                            "max_sampled_peak_rss_bytes",
                        )
                    }
                    for r in measurements
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
