"""Strict, bounded-memory range-power accumulation for headed A100 captures."""

import argparse
import hashlib
import json
import re
from itertools import islice
from pathlib import Path

import numpy as np


def words(path, chunk_bytes=65536):
    """Comma-separated decimal integers; allow one final trailing comma."""
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    pending = ""
    with Path(path).open(encoding="ascii") as stream:
        while chunk := stream.read(chunk_bytes):
            fields = (pending + chunk).split(",")
            pending = fields.pop()
            for field in fields:
                field = field.strip()
                if len(field) > 10 or not re.fullmatch(r"[0-9]+", field):
                    raise ValueError("Expected a decimal uint32 token")
                value = int(field)
                if value > 2**32 - 1:
                    raise ValueError("Token outside uint32 range")
                yield value
            if len(pending) > 64:
                raise ValueError("Oversized numeric token")
        if pending.strip():
            field = pending.strip()
            if len(field) > 10 or not re.fullmatch(r"[0-9]+", field):
                raise ValueError("Invalid final token")
            value = int(field)
            if value > 2**32 - 1:
                raise ValueError("Token outside uint32 range")
            yield value


def unpack(packed):
    packed = np.asarray(packed, dtype=np.uint32)
    samples = np.empty(packed.size * 2, dtype=np.int32)
    samples[::2] = packed >> 16
    samples[1::2] = packed & 65535
    samples[samples >= 32768] -= 65536
    return samples


def header(iterator):
    values = list(islice(iterator, 3))
    if len(values) != 3:
        raise ValueError("Missing channel/sample/chirp header")
    channel, samples, chirps = values
    if channel > 3 or samples < 2 or samples % 2 or samples > 65536 or chirps < 1:
        raise ValueError(
            "Invalid header; expected Rx0..3, even sample count 2..65536, positive chirps"
        )
    return channel, samples, chirps


def range_profile(path, block_chirps=8):
    if not isinstance(block_chirps, int) or not 1 <= block_chirps <= 256:
        raise ValueError("block_chirps must be an integer in 1..256")
    iterator = words(path)
    try:
        channel, samples, chirps = header(iterator)
        window = np.hanning(samples)
        if not np.any(window):
            raise ValueError(
                "At least four samples required for a nonzero symmetric Hann window"
            )
        total = np.zeros(samples // 2 + 1)
        for start in range(0, chirps, block_chirps):
            count = min(block_chirps, chirps - start)
            packed = list(islice(iterator, count * samples // 2))
            if len(packed) != count * samples // 2:
                raise ValueError("Truncated capture; no implicit zero padding")
            adc = unpack(packed).reshape(count, samples)
            total += np.sum(np.abs(np.fft.rfft(adc * window, axis=1)) ** 2, axis=0)
        trailer = list(islice(iterator, 2))
        if trailer not in ([], [0]):
            raise ValueError("Only an optional single zero trailer is supported")
    finally:
        iterator.close()
    return total / chirps, {
        "channel": channel,
        "samples": samples,
        "chirps": chirps,
        "padding_words": len(trailer),
        "block_chirps": block_chirps,
    }


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-chirps", type=int, default=8)
    args = parser.parse_args()
    power, meta = range_profile(args.capture, args.block_chirps)
    meta.update(
        source=args.capture.name,
        sha256=sha256(args.capture),
        units="ADC count squared, uncalibrated",
        window="symmetric Hann; unnormalized rFFT; mean power over all chirps",
        range_power=power.tolist(),
    )
    args.output.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
