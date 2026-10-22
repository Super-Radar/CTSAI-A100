"""Lossless byte-indexed splitting of target logs into monotonic time epochs."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def index_frames(data):
    lines = data.splitlines(keepends=True)
    offset = 0
    frames = []
    current = None
    prefix_end = len(data)
    for line in lines:
        first = line.decode("utf-8-sig").split(",")[0].strip()
        if first == "START":
            if current is not None:
                current["end"] = offset
                frames.append(current)
            else:
                prefix_end = offset
            current = {
                "start": offset,
                "end": None,
                "complete": False,
                "times": [],
                "invalid_time": False,
            }
        elif current is not None:
            if first == "END":
                current["complete"] = True
            if "---->" in first:
                try:
                    current["times"].append(
                        datetime.strptime(
                            first.replace("---->", "").strip(), "%Y/%m/%d %H:%M:%S:%f"
                        )
                    )
                except ValueError:
                    current["invalid_time"] = True
        offset += len(line)
    if current is not None:
        current["end"] = len(data)
        frames.append(current)
    if not frames:
        raise ValueError("No START-framed input")
    for frame in frames:
        times = frame.pop("times")
        frame["time"] = (
            times[0] if len(times) == 1 and not frame.pop("invalid_time") else None
        )
        frame["length"] = frame["end"] - frame["start"]
    return prefix_end, frames


def export(source, destination, max_gap=1.0):
    if not 0 < max_gap < float("inf"):
        raise ValueError("Finite positive max_gap required")
    source = Path(source)
    data = source.read_bytes()
    prefix_end, frames = index_frames(data)
    prefix = data[:prefix_end]
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "prefix.bin").write_bytes(prefix)
    previous = None
    epoch = -1
    active = None
    records = []
    for order, frame in enumerate(frames):
        stamp = frame["time"]
        reason = None
        if not frame["complete"]:
            reason = "incomplete_frame"
        elif stamp is None:
            reason = "missing_or_invalid_or_multiple_timestamp"
        if reason:
            name = "quarantine.bin"
            previous = None
            active = None
        else:
            if previous is None:
                reason = "start_after_boundary"
            elif stamp <= previous:
                reason = "non_increasing_time"
            elif (stamp - previous).total_seconds() > max_gap:
                reason = "long_gap"
            if reason or active is None:
                epoch += 1
                active = f"epoch_{epoch:04d}.csv"
                (dest / active).write_bytes(prefix)
            name = active
            previous = stamp
        chunk = data[frame["start"] : frame["end"]]
        path = dest / name
        with path.open("ab") as stream:
            out_offset = stream.tell()
            stream.write(chunk)
        records.append(
            {
                "order": order,
                "source_start": frame["start"],
                "length": len(chunk),
                "file": name,
                "offset": out_offset,
                "sha256": digest(chunk),
                "time": stamp.isoformat() if stamp is not None else None,
                "complete": frame["complete"],
                "boundary_reason": reason,
            }
        )
    manifest = {
        "version": 1,
        "source": source.name,
        "source_sha256": digest(data),
        "source_bytes": len(data),
        "prefix_bytes": len(prefix),
        "prefix_sha256": digest(prefix),
        "max_gap_s": max_gap,
        "epochs": epoch + 1,
        "frames": records,
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if reconstruct(dest) != data:
        raise RuntimeError("Export failed byte-exact reconstruction")
    return manifest


def _range_chunks(path, offset, length, expected_hash, chunk_bytes):
    checksum = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(remaining, chunk_bytes))
            if not chunk:
                raise ValueError("Truncated artifact")
            remaining -= len(chunk)
            checksum.update(chunk)
            yield chunk
    if checksum.hexdigest() != expected_hash:
        raise ValueError("Artifact integrity failure")


def _verified_chunks(directory, chunk_bytes=65536):
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("Positive integer chunk_bytes required")
    root = Path(directory)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if m.get("version") != 1:
        raise ValueError("Unsupported manifest")
    if (
        type(m["prefix_bytes"]) is not int
        or m["prefix_bytes"] < 0
        or (root / "prefix.bin").stat().st_size != m["prefix_bytes"]
    ):
        raise ValueError("Prefix integrity failure")
    whole_hash = hashlib.sha256()
    for chunk in _range_chunks(
        root / "prefix.bin", 0, m["prefix_bytes"], m["prefix_sha256"], chunk_bytes
    ):
        whole_hash.update(chunk)
        yield chunk
    cursor = m["prefix_bytes"]
    for order, row in enumerate(m["frames"]):
        name = row["file"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid artifact filename")
        if (
            any(
                type(row[key]) is not int
                for key in ("order", "source_start", "length", "offset")
            )
            or row["order"] != order
            or row["source_start"] != cursor
            or row["length"] <= 0
            or row["offset"] < 0
        ):
            raise ValueError("Non-contiguous source index")
        for chunk in _range_chunks(
            root / name, row["offset"], row["length"], row["sha256"], chunk_bytes
        ):
            whole_hash.update(chunk)
            yield chunk
        cursor += row["length"]
    if cursor != m["source_bytes"] or whole_hash.hexdigest() != m["source_sha256"]:
        raise ValueError("Source integrity failure")


def reconstruct(directory):
    """Compatibility API returning all source bytes in memory."""
    return b"".join(_verified_chunks(directory))


def restore(directory, destination, chunk_bytes=65536):
    """Stream verified bytes to a new file; never publish a partial restore.

    The output filesystem must support hard links (e.g. NTFS/ext4). A link
    publishes the fully verified temporary file atomically without overwriting.
    Memory includes the JSON frame index plus a bounded byte-copy buffer.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = None
    checksum, size = hashlib.sha256(), 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".a100-restore-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for chunk in _verified_chunks(directory, chunk_bytes):
                stream.write(chunk)
                checksum.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink()
    return {"source_bytes": size, "source_sha256": checksum.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-gap", type=float, default=1.0)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore source export directory to a new destination file",
    )
    args = parser.parse_args()
    if args.restore:
        print(json.dumps(restore(args.source, args.destination)))
    else:
        result = export(args.source, args.destination, args.max_gap)
        print(
            json.dumps(
                {
                    "frames": len(result["frames"]),
                    "epochs": result["epochs"],
                    "source_sha256": result["source_sha256"],
                }
            )
        )
