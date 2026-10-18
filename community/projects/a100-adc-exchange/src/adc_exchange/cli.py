"""Command line interface: pack / export / rebuild / verify.

Run from the repository root (or anywhere, using absolute paths):

    python -m adc_exchange pack ADC数据采集/示例adc数据和结果/adc_近波_...txt -o /tmp/x.npz
    python -m adc_exchange export /tmp/x.npz -o /tmp/x.csv --max-rows 1000 --mat /tmp/x.mat
    python -m adc_exchange rebuild /tmp/x.npz -o /tmp/x.txt
    python -m adc_exchange verify --npz /tmp/x.npz --txt ADC数据采集/示例adc数据和结果/adc_近波_...txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .export import export_csv, export_mat
from .ingest import IngestError, PROFILE_PARAMS, ingest
from .pack import PackageError, check_package, load_package, pack_file
from .rebuild import rebuild_bytes, rebuild_file


def _parse_profile(text: str | None):
    if text is None:
        return {}
    if text in PROFILE_PARAMS:
        params = PROFILE_PARAMS[text]
        return {"rng_nfft": params["rng_nfft"], "nchirp": params["nchirp"]}
    raise SystemExit(f"error: unknown profile {text!r} (expected init0 or init1)")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_pack(args: argparse.Namespace) -> int:
    kwargs = _parse_profile(args.profile)
    if args.rng_nfft is not None or args.nchirp is not None:
        if args.rng_nfft is None or args.nchirp is None:
            raise SystemExit("error: --rng-nfft and --nchirp must be passed together")
        kwargs = {"rng_nfft": args.rng_nfft, "nchirp": args.nchirp}
    meta = pack_file(
        args.txt,
        args.npz,
        limit_chirps=args.limit_chirps,
        compact=args.compact,
        notes=args.notes,
        **kwargs,
    )
    size = Path(args.npz).stat().st_size
    state = "complete" if meta["is_complete"] else f"partial ({meta['chirps_included']} chirps)"
    print(f"[OK] packed {args.npz} ({size} bytes, {state})")
    print(f"     source sha256 : {meta['source']['sha256']}")
    print(f"     variant       : {meta['format']['variant']}, header={meta['format']['header_fields']}")
    print(
        f"     payload       : {meta['payload_words']} words, trailing={meta['trailing']['count']}"
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    stats = export_csv(args.npz, args.csv, max_rows=args.max_rows)
    suffix = " (truncated)" if stats["truncated"] else ""
    print(f"[OK] wrote {args.csv}: {stats['rows']} rows{suffix}")
    if args.mat:
        export_mat(args.npz, args.mat)
        print(f"[OK] wrote {args.mat} (MATLAB v5)")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    sha = rebuild_file(args.npz, args.txt)
    print(f"[OK] rebuilt {args.txt}")
    print(f"     rebuilt sha256: {sha}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    failures: list[str] = []
    pkg = load_package(args.npz)
    try:
        meta = check_package(pkg)
    except PackageError as exc:
        print(f"[FAIL] package consistency: {exc}")
        return 1
    print("[PASS] package consistency (keys, dtypes, shapes, unpack view, frames)")
    if args.txt is None:
        print(f"[PASS] package {args.npz} is internally consistent")
        print("       (pass --txt to additionally prove byte-identical rebuild)")
        return 0
    src = Path(args.txt)
    if not src.is_file():
        print(f"[FAIL] source text not found: {src}")
        return 1
    actual_sha = _sha256_file(src)
    if actual_sha != meta["source"]["sha256"]:
        failures.append(
            f"source sha256 mismatch: file is {actual_sha}, package records "
            f"{meta['source']['sha256']}"
        )
    else:
        print(f"[PASS] source sha256 matches package metadata ({actual_sha[:16]}...)")
    try:
        rebuilt = rebuild_bytes(meta, pkg["raw_words"], pkg["trailing_words"])
    except PackageError as exc:
        failures.append(f"rebuild refused: {exc}")
    else:
        original = src.read_bytes()
        if rebuilt == original:
            print(f"[PASS] byte-identical rebuild ({len(original)} bytes, {len(original) * 8} bits)")
        else:
            diff = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(rebuilt, original))
                    if a != b
                ),
                min(len(rebuilt), len(original)),
            )
            failures.append(
                f"rebuilt bytes differ from source (first difference at byte {diff}; "
                f"rebuilt {len(rebuilt)} bytes vs source {len(original)} bytes)"
            )
        # Re-ingest the rebuilt text to prove the values survive a full cycle.
        tmp = Path(args.npz).with_suffix(".rebuild-check.tmp")
        tmp.write_bytes(rebuilt)
        try:
            re_ingested = ingest(
                tmp,
                rng_nfft=meta["profile"]["rng_nfft"],
                nchirp=meta["profile"]["nchirp"],
            )
            if not np.array_equal(re_ingested.words, pkg["raw_words"]):
                failures.append("re-ingesting the rebuilt text yielded different payload words")
            elif not np.array_equal(re_ingested.trailing, pkg["trailing_words"]):
                failures.append("re-ingesting the rebuilt text yielded different trailing values")
            else:
                print("[PASS] re-ingest of rebuilt text reproduces identical payload and trailing")
        finally:
            tmp.unlink(missing_ok=True)
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        return 1
    print(f"[PASS] verify complete for {args.npz}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adc_exchange",
        description="Reversible CTSAI-A100 ADC data exchange (txt <-> npz).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pack = sub.add_parser("pack", help="pack an ADC text capture into an .npz package")
    p_pack.add_argument("txt", help="source capture (.txt)")
    p_pack.add_argument("-o", "--npz", required=True, help="output .npz path")
    p_pack.add_argument("--profile", choices=sorted(PROFILE_PARAMS), help="init0 (f0) or init1 (f1)")
    p_pack.add_argument("--rng-nfft", type=int, help="explicit range FFT length")
    p_pack.add_argument("--nchirp", type=int, help="explicit chirp count")
    p_pack.add_argument(
        "--limit-chirps",
        type=int,
        help="keep only the first N chirps (creates a partial package)",
    )
    p_pack.add_argument(
        "--compact",
        action="store_true",
        help="store only raw_words/trailing/metadata; views are derived on load "
        "(smaller archive, still exactly rebuildable)",
    )
    p_pack.add_argument("--notes", help="free-form note stored in metadata")
    p_pack.set_defaults(func=cmd_pack)

    p_export = sub.add_parser("export", help="export package contents to CSV (and .mat)")
    p_export.add_argument("npz", help="input .npz package")
    p_export.add_argument("-o", "--csv", required=True, help="output long-table CSV path")
    p_export.add_argument(
        "--max-rows", type=int, help="truncate the CSV after N data rows"
    )
    p_export.add_argument("--mat", help="optional MATLAB v5 .mat output path")
    p_export.set_defaults(func=cmd_export)

    p_rebuild = sub.add_parser("rebuild", help="restore the original .txt from a package")
    p_rebuild.add_argument("npz", help="input .npz package (must be complete)")
    p_rebuild.add_argument("-o", "--txt", required=True, help="output .txt path")
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_verify = sub.add_parser("verify", help="verify package consistency and exact roundtrip")
    p_verify.add_argument("--npz", required=True, help="input .npz package")
    p_verify.add_argument(
        "--txt", help="original source text; enables the byte-identical rebuild proof"
    )
    p_verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (IngestError, PackageError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
