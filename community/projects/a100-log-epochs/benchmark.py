"""Byte-for-byte round trips and monotonic epoch checks on all public sessions."""

import csv
import hashlib
import itertools
import json
import tempfile
import tracemalloc
from pathlib import Path

from epochs import export, index_frames, reconstruct, restore

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT.parent / "a100-indoor-cart-mti"
    rows = list(csv.DictReader((source / "data/manifest.csv").open(encoding="utf-8")))
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            if row["data_type"] != "raw_target_csv":
                continue
            path = source / row["relative_path"]
            original = path.read_bytes()
            assert hashlib.sha256(original).hexdigest() == row["sha256"]
            dest = Path(tmp) / row["scenario"]
            m = export(path, dest)
            tracemalloc.start()
            restored_bytes = reconstruct(dest)
            _, bytes_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert restored_bytes == original
            del restored_bytes
            restored_path = Path(tmp) / (row["scenario"] + "-restored.csv")
            tracemalloc.start()
            restore_report = restore(dest, restored_path)
            _, stream_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            with restored_path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
            assert restore_report["source_bytes"] == len(original)
            checked = 0
            for epoch in sorted(dest.glob("epoch_*.csv")):
                _, frames = index_frames(epoch.read_bytes())
                times = [frame["time"] for frame in frames]
                assert all(f["complete"] for f in frames) and all(
                    t is not None for t in times
                )
                assert all(
                    0 < (b - a).total_seconds() <= 1
                    for a, b in itertools.pairwise(times)
                )
                checked += len(frames)
            quarantine = sum(r["file"] == "quarantine.bin" for r in m["frames"])
            assert checked + quarantine == len(m["frames"])
            reports.append(
                {
                    "scene": row["scenario"],
                    "source": row["relative_path"],
                    "sha256": row["sha256"],
                    "source_bytes": len(original),
                    "frames": len(m["frames"]),
                    "epochs": m["epochs"],
                    "quarantined_frames": quarantine,
                    "byte_exact_roundtrip": True,
                    "streaming_file_roundtrip": True,
                    "bytes_api_python_peak_bytes": bytes_peak,
                    "streaming_restore_python_peak_bytes": stream_peak,
                    "monotonic_frames_checked": checked,
                }
            )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report = {
        "sessions": reports,
        "total_frames": sum(r["frames"] for r in reports),
        "total_bytes": sum(r["source_bytes"] for r in reports),
        "memory_method": "tracemalloc peak Python allocations during reconstruction or file restore, after export. Includes parsed JSON frame index; excludes previously allocated source bytes and most native memory. Single run, not process RSS. Streaming copy buffer is 65536 bytes.",
        "caveat": "Original wall-clock timestamps and bytes are preserved, not repaired or synchronized. Device FrameNb is not an epoch boundary. Missing/bad/multiple timestamps and incomplete frames are quarantined; no interpolation.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
