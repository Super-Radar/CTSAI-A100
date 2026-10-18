#!/usr/bin/env python
"""Generate the results/ artifacts for the a100-adc-exchange project.

Runs the full pack -> rebuild -> byte-compare -> verify cycle over every
public A-group capture (8 files) and every B-group Pf0 capture (4 files),
measures txt/npz/csv sizes, and builds the small sample package. Writes:

* results/roundtrip_verification.txt
* results/size_comparison.csv
* results/sample_near_f1_rx0_first16chirps.npz

Exit code 0 only if every file roundtrips byte-for-byte and the sample
package stays below 1 MB.

Usage (from the repository root):

    python community/projects/a100-adc-exchange/scripts/run_verification.py
"""

from __future__ import annotations

import csv
import io
import platform
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from adc_exchange import __version__ as tool_version  # noqa: E402
from adc_exchange.cli import main as cli_main  # noqa: E402
from adc_exchange.export import export_csv  # noqa: E402
from adc_exchange.pack import pack_file  # noqa: E402

DATA_A = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果"
DATA_B = (
    REPO_ROOT
    / "ADC数据采集"
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "data"
)
SAMPLE_SOURCE_NAME = "adc_近波_20260512141857_f1_0_14_19_05_965.txt"
SAMPLE_CHIRPS = 16
SAMPLE_LIMIT_BYTES = 1024 * 1024


def cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(argv)
    return int(rc), buf.getvalue().rstrip("\n")


def main() -> int:
    started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    a_files = sorted(DATA_A.glob("adc_*_f?_*.txt"))
    b_files = sorted(DATA_B.glob("adc_test_*_Pf0_*.txt"))
    missing = [str(p) for p in (DATA_A, DATA_B) if not p.is_dir()]
    if missing:
        print(f"[FAIL] missing input locations: {missing}", file=sys.stderr)
        return 2
    expected = {"A": 8, "B": 4}
    counts = {"A": len(a_files), "B": len(b_files)}
    if counts != expected:
        print(f"[FAIL] expected {expected} files, found {counts}", file=sys.stderr)
        return 2
    files = [("A", p) for p in a_files] + [("B", p) for p in b_files]

    work = Path(tempfile.mkdtemp(prefix="a100-adc-exchange-verify-"))
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log: list[str] = []
    size_rows: list[dict] = []
    all_pass = True

    log.append("CTSAI-A100 ADC exchange - full roundtrip verification")
    log.append("=====================================================")
    log.append(f"generated (local time notation, UTC): {started}")
    log.append(f"python : {platform.python_version()}  numpy {np.__version__}")
    log.append(f"tool   : adc_exchange {tool_version}")
    log.append(f"repo   : {REPO_ROOT}")
    log.append("command: python community/projects/a100-adc-exchange/scripts/run_verification.py")
    log.append("")
    log.append("cycle per file: pack -> verify --txt (sha256 + byte-identical rebuild + re-ingest)")
    log.append("")

    for group, src in files:
        stem = src.stem
        npz = work / f"{stem}.npz"
        npz_compact = work / f"{stem}.compact.npz"
        rebuilt = work / f"{stem}.rebuilt.txt"
        rebuilt_compact = work / f"{stem}.compact.rebuilt.txt"
        csv_path = work / f"{stem}.csv"

        meta = pack_file(src, npz)
        pack_file(src, npz_compact, compact=True)
        rc_pack_log = f"pack ok: {npz.name}"

        rc, out = cli(["verify", "--npz", str(npz), "--txt", str(src)])
        rc_rebuild, _ = cli(["rebuild", str(npz), "-o", str(rebuilt)])
        rc_compact_rebuild, _ = cli(
            ["rebuild", str(npz_compact), "-o", str(rebuilt_compact)]
        )
        byte_equal = rebuilt.read_bytes() == src.read_bytes()
        compact_byte_equal = rebuilt_compact.read_bytes() == src.read_bytes()
        export_csv(npz, csv_path)

        ok = rc == 0 and rc_rebuild == 0 and rc_compact_rebuild == 0 and byte_equal and compact_byte_equal
        all_pass &= ok
        status = "PASS" if ok else "FAIL"

        header = meta["format"]["header_fields"]
        log.append(f"[{status}] group {group}  {src.name}")
        log.append(f"      variant={meta['format']['variant']} header={header} "
                   f"trailing={meta['trailing']['values']} total_values="
                   f"{meta['total_values_in_source']} payload_words={meta['payload_words']}")
        log.append(f"      profile={meta['profile']['name']} frames_shape={meta['frames_shape']}")
        log.append(f"      sha256={meta['source']['sha256']}")
        log.append(f"      {rc_pack_log}; rebuild rc={rc_rebuild}; verify rc={rc}; "
                   f"bytes identical: {byte_equal}; compact rebuild rc={rc_compact_rebuild}, "
                   f"bytes identical: {compact_byte_equal}")
        for line in out.splitlines():
            log.append(f"      | {line}")
        log.append("")

        size_rows.append(
            {
                "group": group,
                "file": src.name,
                "variant": meta["format"]["variant"],
                "total_values": meta["total_values_in_source"],
                "payload_words": meta["payload_words"],
                "trailing_values": ";".join(str(v) for v in meta["trailing"]["values"]),
                "txt_bytes": src.stat().st_size,
                "npz_bytes": npz.stat().st_size,
                "npz_compact_bytes": npz_compact.stat().st_size,
                "csv_bytes": csv_path.stat().st_size,
                "payload_binary_bytes": meta["payload_words"] * 4,
            }
        )

    # --- small sample package (near wave, Rx0, first 16 chirps) ---
    sample_src = DATA_A / SAMPLE_SOURCE_NAME
    sample_npz = results_dir / "sample_near_f1_rx0_first16chirps.npz"
    meta_s = pack_file(
        sample_src,
        sample_npz,
        profile="init1",
        limit_chirps=SAMPLE_CHIRPS,
        notes="示例小样本：近波 f1 Rx0 前 16 chirp，便于无全量数据时快速体验（partial，不可逆向重建全量 txt）",
    )
    sample_size = sample_npz.stat().st_size
    sample_ok = sample_size < SAMPLE_LIMIT_BYTES
    all_pass &= sample_ok
    log.append(
        f"[{'PASS' if sample_ok else 'FAIL'}] sample package {sample_npz.name}: "
        f"{sample_size} bytes (< 1 MiB), chirps_included={meta_s['chirps_included']}, "
        f"frames_shape={meta_s['frames_shape']}, source sha256={meta_s['source']['sha256']}"
    )
    log.append("")

    n_pass = sum(1 for line in log if line.startswith("[PASS]"))
    n_fail = sum(1 for line in log if line.startswith("[FAIL]"))
    log.append(f"summary: {n_pass} PASS / {n_fail} FAIL over {len(files)} captures + 1 sample")
    log.append(f"roundtrip: {'ALL BYTE-IDENTICAL' if all_pass else 'FAILURES PRESENT'}")
    log.append(f"exit_code: {0 if all_pass else 1}")
    (results_dir / "roundtrip_verification.txt").write_text(
        "\n".join(log) + "\n", encoding="utf-8"
    )

    csv_path_out = results_dir / "size_comparison.csv"
    with csv_path_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(size_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(size_rows)

    tot_txt = sum(r["txt_bytes"] for r in size_rows)
    tot_npz = sum(r["npz_bytes"] for r in size_rows)
    tot_npz_c = sum(r["npz_compact_bytes"] for r in size_rows)
    tot_csv = sum(r["csv_bytes"] for r in size_rows)
    tot_bin = sum(r["payload_binary_bytes"] for r in size_rows)
    print("\n".join(log[-4:]))
    print(
        f"sizes (12 files): txt {tot_txt} B | npz {tot_npz} B "
        f"({tot_npz / tot_txt:.1%} of txt) | npz compact {tot_npz_c} B "
        f"({tot_npz_c / tot_txt:.1%}) | csv {tot_csv} B "
        f"({tot_csv / tot_txt:.1%}) | uint32 binary {tot_bin} B ({tot_bin / tot_txt:.1%})"
    )
    print(f"wrote {results_dir / 'roundtrip_verification.txt'}")
    print(f"wrote {csv_path_out}")
    print(f"wrote {sample_npz} ({sample_size} bytes)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
