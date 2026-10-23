"""Independent reference validation for guarded CTSAI-A100 processing."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

C = 299_792_458.0
CANVAS = (1100, 650)
PROFILE_FILES = {
    "near": ("f1", "sensor_config_init1.hxx"),
    "far": ("f0", "sensor_config_init0.hxx"),
}


def parse_hxx(path: Path, marker: str) -> dict:
    text = path.read_text(encoding="utf-8")

    def scalar(name: str) -> float:
        match = re.search(rf"\.{name}\s*=\s*([-+0-9.eE]+)\s*,", text)
        if not match:
            raise ValueError(f"{path.name}: missing {name}")
        return float(match.group(1))

    tx_match = re.search(r"\.tx_groups\s*=\s*\{([^}]*)\}", text)
    tx_groups = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)", tx_match.group(1))]
    return {
        "marker": marker,
        "config": path,
        "fc": scalar("fmcw_startfreq") * 1e9,
        "bandwidth": scalar("fmcw_bandwidth") * 1e6,
        "ramp": scalar("fmcw_chirp_rampup") * 1e-6,
        "period": scalar("fmcw_chirp_period") * 1e-6,
        "chirps": int(scalar("nchirp")),
        "sample_rate": scalar("adc_freq") * 1e6,
        "nfft": int(scalar("rng_nfft")),
        "doppler_nfft": int(scalar("vel_nfft")),
        "tx_groups": tx_groups,
    }


def load_adc(files: list[Path], cfg: dict) -> np.ndarray:
    packed = cfg["nfft"] // 2
    expected = packed * cfg["chirps"]
    channels = []
    for path in files:
        # Public exports are a comma-separated record, not a leading header
        # followed by samples. Preserve every field until validation completes.
        values = np.array([float(token) for token in
                           path.read_text(encoding="utf-8").strip().split(",")])
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > np.iinfo(np.uint32).max)
                or np.any(values != np.floor(values))):
            raise ValueError(f"{path.name}: expected finite uint32 ADC words")
        if values.size == expected + 1:
            if values[-1] != 0:
                raise ValueError(f"{path.name}: only one extra trailing zero is supported")
            values = values[:expected]
        if values.size != expected:
            raise ValueError(f"{path.name}: {values.size} values, expected {expected} "
                             "headerless words (+ optional trailing zero)")
        words = values.astype(np.uint32)
        high = (words >> 16).astype(np.uint16).view(np.int16)
        low = (words & 65535).astype(np.uint16).view(np.int16)
        unpacked = np.empty(expected * 2, dtype=np.float64)
        unpacked[0::2] = high
        unpacked[1::2] = low
        channels.append(unpacked.reshape((cfg["nfft"], cfg["chirps"]), order="F"))
    adc = np.stack(channels, axis=2)
    return adc - adc.mean(axis=0, keepdims=True)


def cfar(power: np.ndarray, training=(8, 6), guard=(2, 2), pfa=1e-4) -> np.ndarray:
    tr, td = training
    gr, gd = guard
    kernel = np.ones((2*(tr+gr)+1, 2*(td+gd)+1))
    kernel[tr:tr+2*gr+1, td:td+2*gd+1] = 0
    count = kernel.sum()
    # Range is bounded, while the Doppler FFT axis is periodic. Padding the
    # dimensions separately preserves that topology at +/-N/2.
    padded = np.pad(power, ((tr+gr, tr+gr), (0, 0)), mode="constant")
    padded = np.pad(padded, ((0, 0), (td+gd, td+gd)), mode="wrap")
    noise = np.zeros_like(power)
    for r, d in np.argwhere(kernel):
        noise += padded[r:r+power.shape[0], d:d+power.shape[1]]
    noise /= count
    alpha = count * (pfa ** (-1/count) - 1)
    mask = power > alpha * noise
    local = np.ones_like(mask)
    for dr in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if dr or dd:
                local &= power >= np.roll(power, (dr, dd), axis=(0, 1))
    mask &= local
    border_r, border_d = tr + gr, td + gd
    mask[:border_r] = mask[-border_r:] = False
    return mask


def validate_cfar_topology() -> None:
    """Regression-check detection at the periodic Doppler boundary."""
    probe = np.ones((64, 64), dtype=np.float64)
    probe[32, 0] = 1_000.0
    mask = cfar(probe, training=(4, 4), guard=(1, 1), pfa=1e-3)
    if not mask[32, 0]:
        raise AssertionError("CFAR lost a target at the wrapped Doppler boundary")


def heatmap_image(values: np.ndarray, title: str, cells=None) -> Image.Image:
    lo, hi = np.percentile(values, (1, 99.5))
    norm = np.clip((values-lo)/(hi-lo+1e-12), 0, 1)
    red = np.clip(1.5-abs(4*norm-3), 0, 1)
    green = np.clip(1.5-abs(4*norm-2), 0, 1)
    blue = np.clip(1.5-abs(4*norm-1), 0, 1)
    rgb = (255*np.stack((red, green, blue), axis=2)).astype(np.uint8)
    plot = Image.fromarray(np.flipud(rgb)).resize((1000, 540), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", CANVAS, "white")
    canvas.paste(plot, (70, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 25), title, fill="black")
    draw.text((450, 625), "Raw Doppler bins", fill="black")
    draw.text((5, 320), "Range bins", fill="black")
    if cells:
        rows, cols = values.shape
        for r, d in cells:
            x = 70 + int(d/(cols-1)*1000)
            y = 70 + int((rows-1-r)/(rows-1)*540)
            draw.ellipse((x-5, y-5, x+5, y+5), outline="red", width=2)
    return canvas


def line_image(y: np.ndarray, title: str) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 25), title, fill="black")
    draw.rectangle((70, 70, 1070, 610), outline="black")
    ymin, ymax = np.percentile(y, (1, 99.5))
    points = [(70+int(i/(len(y)-1)*1000), 610-int(np.clip((v-ymin)/(ymax-ymin+1e-12),0,1)*540))
              for i, v in enumerate(y)]
    draw.line(points, fill=(20, 80, 180), width=2)
    draw.text((470, 625), "Range bins", fill="black")
    return canvas


def status_image(name: str, count: int) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(canvas)
    lines = [
        f"CTSAI-A100 {name} processing status",
        "Range and raw Doppler diagnostics completed.",
        f"Raw CFAR detections exported: {count}",
        "Physical velocity withheld: DDMA coding/offset metadata unavailable.",
        "Angle withheld: DDMA separation/channel order/TX calibration unavailable.",
        "See validation.txt and project documentation.",
    ]
    for i, line in enumerate(lines):
        draw.text((80, 70+i*80), line, fill="black")
    return canvas


def save_results(name: str, cfg: dict, files: list[Path], output: Path) -> None:
    adc = load_adc(files, cfg)
    range_fft = np.fft.fft(adc*np.hanning(adc.shape[0])[:, None, None],
                           cfg["nfft"], axis=0)[:cfg["nfft"]//2]
    # Unresolved DDMA coding can move a stationary TX response away from the
    # center bin. Preserve the complete combined raw spectrum rather than
    # applying slow-time mean or center-bin suppression.
    rd = np.fft.fftshift(np.fft.fft(
        range_fft*np.hanning(cfg["chirps"])[None, :, None],
        cfg["doppler_nfft"], axis=1), axes=1)
    power = np.mean(np.abs(rd)**2, axis=2)
    power_db = 10*np.log10(power + np.finfo(float).eps)
    slope = cfg["bandwidth"] / cfg["ramp"]
    range_bin = C*cfg["sample_rate"]/(2*slope*cfg["nfft"])
    ranges = np.arange(power.shape[0])*range_bin
    doppler_bins = np.arange(cfg["doppler_nfft"])-cfg["doppler_nfft"]//2
    mask = cfar(power)
    # Filter the range ROI after CFAR so excluded bins cannot depress the
    # training-cell average at the ROI boundaries.
    mask[(ranges < 0.5) | (ranges > 0.9*ranges[-1])] = False
    cells = sorted(np.argwhere(mask), key=lambda x: power_db[tuple(x)], reverse=True)[:32]

    output.mkdir(parents=True, exist_ok=True)
    rows = [(ranges[r], int(doppler_bins[d]), "", "", power_db[r, d], False,
             "raw_ddma_not_decoded") for r, d in cells]
    with (output/"detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("range_m", "doppler_bin", "velocity_mps", "angle_deg",
                         "power_db", "kinematics_valid", "processing_status"))
        writer.writerows(rows)

    line_image(20*np.log10(np.abs(range_fft[:, 0, 0])+1e-12),
               f"CTSAI-A100 {name} range spectrum").save(output/"01_range_spectrum.png")
    heatmap_image(power_db, f"CTSAI-A100 {name} raw Range-Doppler map").save(
        output/"02_raw_range_doppler_map.png")
    heatmap_image(power_db, f"CTSAI-A100 {name} raw CA-CFAR detections", cells).save(
        output/"03_raw_cfar_detections.png")
    status_image(name, len(rows)).save(output/"04_processing_status.png")

    tx = ",".join(f"0x{x:04x}" for x in cfg["tx_groups"])
    summary = (f"profile={name}\nconfig={cfg['config'].name}\ntx_groups={tx}\n"
               f"adc_shape={adc.shape}\nrd_shape={rd.shape}\n"
               f"finite={np.isfinite(power_db).all()}\ndetections={len(rows)}\n"
               "mimo_mode=unresolved_ddma\nmetadata_complete=False\n"
               "doppler_cfar_wrap=True\nrange_roi_after_cfar=True\n"
               "slow_time_mean_removal=False\ncenter_bin_suppression=False\n"
               "physical_velocity_valid=False\nangle_valid=False\n")
    (output/"validation.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


def main() -> None:
    validate_cfar_topology()
    validation_dir = Path(__file__).resolve().parent
    repo = validation_dir.parents[3]
    data_dir = repo/"ADC数据采集"/"示例adc数据和结果"
    config_dir = (repo/"ADC数据采集"/
                  "matlab_signal_processing_platform_231023_for_txt_A100"/
                  "cfg"/"CTASI-A100配置")
    all_files = list(data_dir.glob("*.txt"))
    for name, (marker, config_name) in PROFILE_FILES.items():
        cfg = parse_hxx(config_dir/config_name, marker)
        files = sorted(p for p in all_files if f"_{marker}_" in p.name)
        if len(files) != 4:
            raise FileNotFoundError(f"{name}: expected four files, found {len(files)}")
        save_results(name, cfg, files, validation_dir/"results"/name)


if __name__ == "__main__":
    main()
