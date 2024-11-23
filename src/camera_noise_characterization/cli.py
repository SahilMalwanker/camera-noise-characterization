"""Command-line interface: ``camera-noise-characterization <command> --help``."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from camera_noise_characterization import edges, filters, noise, protocols
from camera_noise_characterization.io import load_gray, save_gray
from camera_noise_characterization.optics import BAUMER_TXG08, Lens, OpticalSetup, Sensor
from camera_noise_characterization.sensor import SimulatedCamera

CSV_FIELDS = ["exposure_us", "gain", "mean_dn", "sigma_diff_dn"]


def _table(rows: Sequence[Sequence[object]], header: Sequence[str]) -> str:
    cells = [[str(c) for c in header]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = {"mean_dn", "sigma_diff_dn"} - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")

    def column(name: str) -> np.ndarray:
        return np.array(
            [float(r[name]) if r.get(name) not in (None, "") else math.nan for r in rows]
        )

    data = {name: column(name) for name in CSV_FIELDS if name in rows[0]}
    data["include"] = np.array(
        [(r.get("include") or "1").strip().lower() not in ("0", "no", "false") for r in rows]
    )
    return data


def cmd_optics(args: argparse.Namespace) -> None:
    sensor = BAUMER_TXG08
    if args.pixel_um or args.width_px or args.height_px:
        sensor = Sensor(
            "custom",
            args.width_px or sensor.width_px,
            args.height_px or sensor.height_px,
            args.pixel_um or sensor.pixel_um,
        )
    setup = OpticalSetup(sensor, Lens(args.focal, args.f_number), args.distance, args.coc_um)
    rows = [(k, f"{v:.5g}") for k, v in setup.summary().items()]
    print(f"{sensor.name}, f = {args.focal} mm, f/{args.f_number}, object at {args.distance} mm\n")
    print(_table(rows, ["quantity", "value"]))


def cmd_pair(args: argparse.Namespace) -> None:
    s = noise.pair_stats(load_gray(args.a), load_gray(args.b), args.black_level)
    rows = [
        ("mean signal [DN]", f"{s.mean:.3f}"),
        ("std(A - B) [DN]", f"{s.sigma_diff:.4f}"),
        ("temporal noise per frame [DN]", f"{s.sigma:.4f}"),
        ("relative noise std(A-B)/mean", f"{s.relative_noise:.5f}"),
        ("SNR [dB]", f"{20 * math.log10(s.snr):.2f}"),
    ]
    print(_table(rows, ["quantity", "value"]))


def cmd_fit(args: argparse.Namespace) -> None:
    data = _read_csv(Path(args.csv))
    keep = data["include"] & np.isfinite(data["mean_dn"]) & np.isfinite(data["sigma_diff_dn"])
    mean, var = data["mean_dn"][keep], data["sigma_diff_dn"][keep] ** 2 / 2.0
    gains = data.get("gain", np.ones_like(data["mean_dn"]))[keep]
    model = args.model
    if model == "auto":
        model = "gain" if np.unique(gains[np.isfinite(gains)]).size >= 3 else "ptc"
    if model == "gain":
        fit = noise.fit_gain_sweep(gains, mean, var)
        rows = [
            ("model", "var = K1 g mean + (K1 sr)^2 g^2 + 1/12"),
            ("points used", fit.n_points),
            (
                "conversion gain K1 [DN/e-]",
                f"{fit.conversion_gain:.4f} +- {fit.conversion_gain_se:.4f}",
            ),
            ("electrons per DN at gain 1", f"{1 / fit.conversion_gain:.1f}"),
            ("read noise [e-]", f"{fit.read_noise_e:.1f}"),
            ("read noise at gain 1 [DN]", f"{fit.read_noise_dn:.2f}"),
            ("R^2", f"{fit.r_squared:.4f}"),
        ]
    else:
        fit = noise.fit_photon_transfer(mean, var, args.max_mean)
        rows = [
            ("model", "var = K mean + var0"),
            ("points used", fit.n_points),
            (
                "conversion gain K [DN/e-]",
                f"{fit.conversion_gain:.4f} +- {fit.conversion_gain_se:.4f}",
            ),
            ("electrons per DN", f"{fit.electrons_per_dn:.1f}"),
            ("noise floor [DN]", f"{math.sqrt(max(fit.noise_floor_var, 0)):.2f}"),
            ("electrons at full scale (255 DN)", f"{fit.saturation_e():.0f}"),
            ("R^2", f"{fit.r_squared:.4f}"),
        ]
    print(_table(rows, ["quantity", "value"]))


def cmd_simulate(args: argparse.Namespace) -> None:
    cam = SimulatedCamera(photon_rate=args.photon_rate, seed=args.seed)
    if args.protocol == "ptc":
        points = protocols.photon_transfer_sweep(cam, n_points=args.steps)
    elif args.protocol == "exposure":
        cam.gain = 1.0
        t0 = protocols.auto_expose(cam, 30.0)
        t1 = protocols.auto_expose(cam, 220.0)
        points = protocols.exposure_sweep(cam, t0, t1, args.steps)
    else:
        points = protocols.gain_sweep(cam, range(1, 11), target_mean=128.0)
    rows = [
        (f"{p.exposure_us:.1f}", f"{p.gain:g}", f"{p.stats.mean:.3f}", f"{p.stats.sigma_diff:.4f}")
        for p in points
    ]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_FIELDS)
            writer.writerows(rows)
        print(f"wrote {len(rows)} points to {args.out}")
    else:
        print(_table(rows, CSV_FIELDS))


def cmd_denoise(args: argparse.Namespace) -> None:
    results = filters.denoise_benchmark(load_gray(args.a), load_gray(args.b))
    rows = [
        (r.name, f"{r.sigma_after:.4f}", f"{r.measured_gain:.3f}", f"{r.predicted_gain:.3f}")
        for r in results
    ]
    print(_table(rows, ["filter", "noise [DN]", "measured gain", "white-noise theory"]))


def cmd_edges(args: argparse.Namespace) -> None:
    img = load_gray(args.image).astype(float)
    if args.method == "canny":
        out = edges.canny(img, sigma=args.sigma, non_edge_fraction=args.non_edge_fraction)
    elif args.method == "log":
        out = edges.log_edges(img, sigma=args.sigma)
    else:
        gx, gy = edges.gradients(img, args.method)
        out = edges.threshold_magnitude(np.hypot(gx, gy))
    save_gray(args.out, out.astype(np.uint8) * 255)
    print(f"{args.method}: {int(out.sum())} edge pixels -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="camera-noise-characterization", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    o = sub.add_parser("optics", help="field of view, pixel footprint, depth of field")
    o.add_argument("--focal", type=float, default=12.0, help="focal length [mm]")
    o.add_argument("--f-number", type=float, default=1.4)
    o.add_argument("--distance", type=float, default=342.0, help="object distance [mm]")
    o.add_argument("--coc-um", type=float, default=None, help="circle of confusion [um]")
    o.add_argument("--width-px", type=int)
    o.add_argument("--height-px", type=int)
    o.add_argument("--pixel-um", type=float)
    o.set_defaults(func=cmd_optics)

    pr = sub.add_parser("pair", help="mean and temporal noise of two flat-field frames")
    pr.add_argument("a")
    pr.add_argument("b")
    pr.add_argument("--black-level", type=float, default=0.0)
    pr.set_defaults(func=cmd_pair)

    f = sub.add_parser("fit", help="photon-transfer or gain-sweep fit of a measurement CSV")
    f.add_argument("csv", help=f"columns: {', '.join(CSV_FIELDS)} (optional: include)")
    f.add_argument("--model", choices=["auto", "ptc", "gain"], default="auto")
    f.add_argument("--max-mean", type=float, default=240.0, help="ignore points above [DN]")
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("simulate", help="run a lab protocol on the sensor twin")
    s.add_argument("protocol", choices=["ptc", "exposure", "gain"])
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--photon-rate", type=float, default=3.0, help="photons / pixel / us")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="write CSV here (same format `fit` reads)")
    s.set_defaults(func=cmd_simulate)

    d = sub.add_parser("denoise", help="binomial and median filters on a flat-field pair")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(func=cmd_denoise)

    e = sub.add_parser("edges", help="edge map of an image")
    e.add_argument("image")
    e.add_argument("--method", choices=["sobel", "prewitt", "canny", "log"], default="canny")
    e.add_argument("--sigma", type=float, default=1.4)
    e.add_argument(
        "--non-edge-fraction",
        type=float,
        default=0.7,
        help="Canny: share of pixels assumed to be non-edges (raise for textured scenes)",
    )
    e.add_argument("--out", default="edges.png")
    e.set_defaults(func=cmd_edges)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
