"""Regenerate every figure in docs/figures from the package and the data in data/.

python examples/reproduce_figures.py           # full resolution
python examples/reproduce_figures.py --quick   # small frames (CI smoke test)
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from camera_noise_characterization import edges, filters, noise, protocols  # noqa: E402
from camera_noise_characterization.io import load_gray  # noqa: E402
from camera_noise_characterization.optics import (  # noqa: E402
    BAUMER_TXG08,
    Lens,
    OpticalSetup,
    airy_diameter_um,
    diffraction_limited_f_number,
)
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera  # noqa: E402
from camera_noise_characterization.targets import siemens_star  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NAVY, TEAL, ORANGE, GREY, RED = "#1c3d5a", "#0b7a75", "#e8590c", "#8a8f98", "#c92a2a"
LAB_MEAN = 129.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4a4f57",
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    im = im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, optimize=True)
    print(f"  {path.name:28s} {im.size[0]}x{im.size[1]}  {path.stat().st_size / 1024:.0f} KB")


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lab_exposure_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray, noise.PTCFit]:
    rows = read_csv("lab_exposure_sweep.csv")
    mu = np.array([float(r["mean_dn"]) for r in rows])
    var = np.array([float(r["sigma_diff_dn"]) for r in rows]) ** 2 / 2
    used = np.array([r["include"] == "1" for r in rows])
    return mu, var, used, noise.fit_photon_transfer(mu[used], var[used])


def lab_gain_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray, noise.GainSweepFit]:
    rows = read_csv("lab_gain_sweep.csv")
    g = np.array([float(r["gain"]) for r in rows])
    s = np.array([float(r["sigma_diff_dn"]) for r in rows])
    used = np.array([r["include"] == "1" for r in rows])
    fit = noise.fit_gain_sweep(g[used], np.full(used.sum(), LAB_MEAN), s[used] ** 2 / 2)
    return g, s, used, fit


def draw_lab_ptc(ax: plt.Axes, compact: bool = False) -> None:
    mu, var, used, fit = lab_exposure_sweep()
    x = np.linspace(0, 255, 50)
    ax.plot(x, fit.conversion_gain * x + fit.noise_floor_var, color=TEAL, lw=2.2, zorder=2)
    ax.scatter(mu[used], var[used], s=55, color=NAVY, zorder=3, label="logged, used in fit")
    shown = ~used & (mu <= 255)
    ax.scatter(
        mu[shown],
        var[shown],
        s=55,
        facecolors="white",
        edgecolors=NAVY,
        lw=1.5,
        zorder=3,
        label="near full scale, excluded",
    )
    ax.axvline(255, color=GREY, ls=":", lw=1)
    ax.set(
        xlim=(0, 270),
        ylim=(0, 7),
        xlabel=r"mean signal $\mu$ [DN]",
        ylabel=r"temporal variance $\sigma^2$ [DN$^2$]",
    )
    label = (
        f"K = {fit.conversion_gain:.4f} $\\pm$ {fit.conversion_gain_se:.4f} DN/e$^-$\n"
        f"= {fit.electrons_per_dn:.0f} e$^-$ per DN"
    )
    ax.text(
        0.04,
        0.95 if compact else 0.72,
        label,
        transform=ax.transAxes,
        va="top",
        color=TEAL,
        fontweight="bold",
    )
    if not compact:
        ax.legend(loc="upper left")


def draw_lab_gain(ax: plt.Axes, compact: bool = False) -> None:
    g, s, used, fit = lab_gain_sweep()
    gg = np.linspace(1, 10, 200)
    mean = np.full_like(gg, LAB_MEAN)
    ax.plot(
        gg,
        np.sqrt(2) * fit.predict_sigma(gg, mean),
        color=TEAL,
        lw=2.2,
        label="model: shot and read noise in quadrature",
    )
    if not compact:
        ax.plot(
            gg,
            np.sqrt(2 * fit.conversion_gain * gg * LAB_MEAN),
            color=ORANGE,
            ls="--",
            lw=1.4,
            label=r"shot noise only, $\propto\sqrt{g}$",
        )
        ax.plot(
            gg,
            np.sqrt(2) * fit.read_noise_dn * gg,
            color=GREY,
            ls="--",
            lw=1.4,
            label=r"read noise only, $\propto g$",
        )
    ax.scatter(g[used], s[used], s=55, color=NAVY, zorder=3, label="measured, used in fit")
    ax.scatter(
        g[~used],
        s[~used],
        s=55,
        facecolors="white",
        edgecolors=NAVY,
        lw=1.5,
        zorder=3,
        label="shorter exposure (less signal), excluded",
    )
    ax.set(xlim=(0.5, 10.5), ylim=(0, 12), xlabel="analogue gain g", ylabel="std(A - B) [DN]")
    label = (
        f"K$_1$ = {fit.conversion_gain:.4f} $\\pm$ {fit.conversion_gain_se:.4f} DN/e$^-$\n"
        f"read noise {fit.read_noise_e:.0f} e$^-$ = {fit.read_noise_dn:.2f} DN"
    )
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", color=TEAL, fontweight="bold")
    if not compact:
        ax.legend(loc="lower right")


def fig_lab_photon_transfer(path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    draw_lab_ptc(ax1)
    ax1.set_title("Exposure sweep at gain 1")
    draw_lab_gain(ax2)
    ax2.set_title(f"Gain sweep at constant {LAB_MEAN:.0f} DN")
    fig.suptitle(
        "Real Baumer TXG08 data - two independent experiments, one conversion gain",
        fontweight="bold",
        fontsize=12.5,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save(fig, path)


def fig_twin_validation(path: Path, quick: bool) -> None:
    w, h = (256, 192) if quick else (1032, 776)
    sensor = SensorModel(width=w, height=h, seed=1)
    cam = SimulatedCamera(sensor, seed=2)
    points = protocols.photon_transfer_sweep(
        cam, n_points=12 if quick else 36, overshoot=1.3, black_level=sensor.black_level_dn
    )
    mu = np.array([p.stats.mean for p in points])
    var = np.array([p.stats.variance for p in points])
    full = sensor.max_dn - sensor.black_level_dn
    used = mu < 0.9 * full
    fit = noise.fit_photon_transfer(mu[used], var[used])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.linspace(0, full, 50)
    ax1.plot(x, fit.conversion_gain * x + fit.noise_floor_var, color=TEAL, lw=2.2, label="fit")
    ax1.scatter(mu[used], var[used], s=28, color=NAVY, zorder=3, label="twin, linear range")
    ax1.scatter(
        mu[~used],
        var[~used],
        s=28,
        facecolors="white",
        edgecolors=RED,
        zorder=3,
        label="clipping squeezes the variance",
    )
    ax1.set(
        xlabel=r"mean signal $\mu$ [DN]",
        ylabel=r"temporal variance $\sigma^2$ [DN$^2$]",
        title="Photon transfer curve of the sensor twin",
    )
    k_true, r_true = sensor.conversion_gain, sensor.read_noise_e
    truth = (
        f"ground truth  K = {k_true:.4f},  read noise = {r_true:.0f} e$^-$\n"
        f"recovered     K = {fit.conversion_gain:.4f},  read noise = {fit.read_noise_e:.1f} e$^-$"
    )
    ax1.text(
        0.03,
        0.97,
        truth,
        transform=ax1.transAxes,
        va="top",
        family="monospace",
        fontsize=8.5,
        bbox={"facecolor": "#f4f6f8", "edgecolor": "none", "pad": 6},
    )
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82))

    k, r = sensor.conversion_gain, sensor.read_noise_e * sensor.conversion_gain
    xs = np.logspace(np.log10(0.5), np.log10(full), 200)
    snr = xs / np.sqrt(k * xs + r**2 + noise.QUANTISATION_VAR)
    ax2.loglog(xs, snr, color=TEAL, lw=2.2, label="twin model")
    ax2.loglog(
        xs,
        xs / np.sqrt(r**2 + noise.QUANTISATION_VAR),
        color=GREY,
        ls="--",
        lw=1.2,
        label="read-noise limit, slope 1",
    )
    ax2.loglog(
        xs, np.sqrt(xs / k), color=ORANGE, ls="--", lw=1.2, label="shot-noise limit, slope 1/2"
    )
    ax2.scatter(
        mu[used], mu[used] / np.sqrt(var[used]), s=22, color=NAVY, zorder=3, label="twin pairs"
    )
    lab_mu, lab_var, lab_used, _ = lab_exposure_sweep()
    ax2.scatter(
        lab_mu[lab_used],
        lab_mu[lab_used] / np.sqrt(lab_var[lab_used]),
        s=90,
        marker="*",
        color=ORANGE,
        edgecolors="black",
        lw=0.6,
        zorder=4,
        label="real camera (lab log)",
    )
    ax2.set(
        xlim=(0.5, 300),
        ylim=(0.5, 200),
        xlabel=r"mean signal $\mu$ [DN]",
        ylabel=r"SNR = $\mu / \sigma$",
        title="Signal-to-noise ratio: twin vs real camera",
    )
    ax2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, path)


def fig_optics(path: Path) -> None:
    lens = Lens(12.0, 1.4)
    lab = OpticalSetup(BAUMER_TXG08, lens, 342.0)
    d = np.linspace(100, 1000, 200)
    fov = np.array([OpticalSetup(BAUMER_TXG08, lens, x).field_of_view_mm() for x in d])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax1.plot(d, fov[:, 0], color=NAVY, lw=2.2, label="width (1032 px)")
    ax1.plot(d, fov[:, 1], color=TEAL, lw=2.2, label="height (776 px)")
    ax1.set(
        xlabel="object distance [mm]",
        ylabel="field of view [mm]",
        title="12 mm lens on the TXG08: what one frame covers",
    )
    w, hgt = lab.field_of_view_mm()
    ax1.axvline(342, color=GREY, ls=":", lw=1)
    ax1.scatter([342, 342], [w, hgt], color=ORANGE, zorder=3, s=40)
    ax1.annotate(
        f"lab set-up, 342 mm:\n{w:.0f} x {hgt:.0f} mm field\n"
        f"{lab.object_pixel_mm() * 1000:.0f} $\\mu$m per pixel\n"
        f"{lab.depth_of_field_mm()[2]:.1f} mm depth of field at f/1.4",
        xy=(342, w),
        xytext=(560, 40),
        color=NAVY,
        arrowprops={"arrowstyle": "->", "color": GREY},
    )
    ax1.legend(loc="upper left", title="field of view")

    ns = np.array([1.4, 2, 2.8, 4, 5.6, 8, 11, 16])
    dof = np.array(
        [OpticalSetup(BAUMER_TXG08, Lens(12, n), 342.0).depth_of_field_mm()[2] for n in ns]
    )
    airy = airy_diameter_um(ns)
    n_lim = diffraction_limited_f_number(BAUMER_TXG08.pixel_um)
    ax2.bar(np.arange(len(ns)), dof, color=TEAL, width=0.62, label="depth of field at 342 mm")
    ax2.set_xticks(np.arange(len(ns)), [f"f/{n:g}\nx{(n / 1.4) ** 2:.0f}" for n in ns])
    ax2.set(
        ylabel="depth of field [mm]",
        ylim=(0, dof.max() * 1.18),
        xlabel="f-number  /  exposure time needed relative to f/1.4",
        title="Stopping down: more depth, less light, more diffraction",
    )
    ax2.axvspan(
        np.interp(n_lim, ns, np.arange(len(ns))),
        len(ns) - 0.5,
        color=RED,
        alpha=0.07,
        label=f"diffraction-limited (beyond f/{n_lim:.1f})",
    )
    tw = ax2.twinx()
    tw.plot(np.arange(len(ns)), airy, color=RED, marker="o", lw=1.8, label="Airy disk diameter")
    tw.axhline(2 * BAUMER_TXG08.pixel_um, color=RED, ls=":", lw=1)
    tw.text(0.1, 2 * BAUMER_TXG08.pixel_um + 0.4, "2 pixels", color=RED, fontsize=8)
    tw.set_ylabel(r"Airy disk diameter [$\mu$m]", color=RED)
    tw.tick_params(axis="y", colors=RED)
    tw.grid(False)
    tw.spines["right"].set_visible(True)
    handles = ax2.get_legend_handles_labels()[0] + tw.get_legend_handles_labels()[0]
    ax2.legend(handles, [h.get_label() for h in handles], loc="upper left")
    fig.tight_layout()
    save(fig, path)


def star_camera(size: int, seed: int = 4) -> SimulatedCamera:
    star = siemens_star(size=size, spokes=24)[0]
    return SimulatedCamera(
        SensorModel(width=size, height=size, seed=seed), photon_rate=9.7, scene=star, seed=seed
    )


def fig_ladder(path: Path, quick: bool) -> None:
    size = 160 if quick else 360
    settings = [
        ("Under-exposed", 300.0, 1.0),
        ("Well exposed", 2500.0, 1.0),
        ("Over-exposed", 9000.0, 1.0),
        ("Same brightness via gain 8", 2500.0 / 8, 8.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), gridspec_kw={"height_ratios": [3, 1.1]})
    for col, (title, t, g) in enumerate(settings):
        cam = star_camera(size)
        cam.exposure_us, cam.gain = t, g
        a, b = cam.grab(), cam.grab()
        st = noise.pair_stats(a, b)
        clipped = 100.0 * np.mean(a == 255)
        ax = axes[0, col]
        ax.imshow(a, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{title}\n{t:.0f} $\\mu$s, gain {g:g}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        snr = f"SNR {20 * np.log10(st.snr):.0f} dB" if clipped < 1 else "SNR invalid"
        ax.set_xlabel(f"mean {st.mean:.0f} DN  |  {snr}  |  clipped {clipped:.0f} %", fontsize=8.5)
        hx = axes[1, col]
        hx.hist(a.ravel(), bins=64, range=(0, 256), color=NAVY if clipped < 1 else RED)
        hx.set_xlim(0, 256)
        hx.set_yticks([])
        hx.set_xticks([0, 128, 255])
    fig.suptitle(
        "Exposure vs gain on the sensor twin: gain brightens the image, "
        "but SNR is set by the photons collected",
        fontweight="bold",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save(fig, path)


def fig_denoise(path: Path, quick: bool) -> None:
    w, h = (256, 192) if quick else (1032, 776)
    sensor = SensorModel(width=w, height=h, seed=5)
    cam = SimulatedCamera(sensor, seed=6)
    protocols.auto_expose(cam, 128.0)
    a, b = cam.grab(), cam.grab()
    results = filters.denoise_benchmark(a, b)[1:]

    fig = plt.figure(figsize=(13, 4.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.2, 0.15, 1, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(results))
    meas = [r.measured_gain for r in results]
    pred = [r.predicted_gain for r in results]
    ax.bar(xs - 0.19, pred, width=0.38, color="#c5d3e0", label="theory for white noise")
    ax.bar(xs + 0.19, meas, width=0.38, color=TEAL, label="measured on 8-bit flat-field pair")
    for x, m in zip(xs, meas, strict=True):
        ax.text(x + 0.19, m + 0.012, f"{m:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs, [r.name for r in results])
    ax.set(
        ylabel="noise after / noise before",
        ylim=(0, 0.6),
        title=f"Gaussian-like noise ({results[0].sigma_before:.2f} DN): linear filters win",
    )
    ax.legend(loc="upper right")

    rng = np.random.default_rng(8)
    star = 255 * siemens_star(size=96 if quick else 160, spokes=12)[0]
    noisy = filters.add_impulse_noise(star + rng.normal(0, 3, star.shape), 0.06, rng)
    panels = [
        ("6 % salt and pepper", noisy),
        ("binomial 5x5", filters.convolve(noisy, filters.binomial_kernel(5))),
        ("median 3x3", filters.median(noisy, 3)),
    ]
    for i, (name, im) in enumerate(panels):
        axi = fig.add_subplot(gs[0, 2 + i])
        axi.imshow(np.clip(im, 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        mae = np.abs(np.clip(im, 0, 255) - star).mean()
        axi.set_title(name, fontsize=10)
        axi.set_xlabel(f"mean abs. error {mae:.1f} DN", fontsize=8.5)
        axi.set_xticks([])
        axi.set_yticks([])
        axi.grid(False)
    fig.text(0.62, 0.95, "Impulse noise: the median wins", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def coins_crop() -> np.ndarray:
    img = load_gray(DATA / "coins_txg08.png").astype(float)
    return img[120:600, 180:820]


def fig_edges(path: Path) -> None:
    img = coins_crop()
    sx, sy = edges.gradients(img, "sobel")
    px, py = edges.gradients(img, "prewitt")
    panels = [
        ("Real frame (TXG08)", img, "img"),
        ("Sobel x", sx, "signed"),
        ("Sobel y", sy, "signed"),
        ("Laplacian (alpha 0.2)", edges.laplacian(img, 0.2), "signed"),
        ("Prewitt x", px, "signed"),
        ("Prewitt y", py, "signed"),
        ("LoG zero crossings", edges.log_edges(img, 2.0), "binary"),
        ("Canny, 90 % non-edge rule", edges.canny(img, 2.0, non_edge_fraction=0.9), "binary"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.9))
    for ax, (title, im, kind) in zip(axes.ravel(), panels, strict=True):
        if kind == "signed":
            m = np.percentile(np.abs(im), 99.5)
            ax.imshow(im, cmap="gray", vmin=-m, vmax=m, interpolation="nearest")
        elif kind == "binary":
            ax.imshow(im, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, path, dpi=120)


def fig_edge_f1(path: Path, quick: bool) -> None:
    image, truth, care = siemens_star(size=192 if quick else 256, spokes=16, low=0.38, high=0.62)
    clean = 255 * image
    contrast = 255 * (0.62 - 0.38)
    sigmas = np.array([0, 15, 30]) if quick else np.arange(0, 37, 3)
    seeds = [0] if quick else [0, 1, 2]
    methods = {
        "Sobel + threshold": lambda x: edges.threshold_magnitude(np.hypot(*edges.gradients(x))),
        "Prewitt + threshold": lambda x: edges.threshold_magnitude(
            np.hypot(*edges.gradients(x, "prewitt"))
        ),
        "LoG zero crossings, sigma 2": lambda x: edges.log_edges(x, 2.0),
        "Canny, sigma 1.4": lambda x: edges.canny(x, 1.4),
        "Canny, sigma 2.5": lambda x: edges.canny(x, 2.5),
    }
    colors = [GREY, "#b8bec7", ORANGE, "#7fc4bf", TEAL]
    fig, (ax, axi) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.9, 1]})
    for (name, fn), color in zip(methods.items(), colors, strict=True):
        f1 = []
        for s in sigmas:
            vals = []
            for seed in seeds:
                noisy = clean + np.random.default_rng(seed).normal(0, s, clean.shape)
                vals.append(edges.edge_scores(fn(noisy), truth, tolerance=1.5, care=care)[2])
            f1.append(np.mean(vals))
        ax.plot(sigmas, f1, marker="o", lw=2.2, color=color, label=name)
    ax.set(
        xlabel=f"noise standard deviation [DN]  (edge contrast {contrast:.0f} DN)",
        ylabel="F1 score (1.5 px tolerance)",
        ylim=(0, 1.02),
        title="Edge detectors vs noise on a Siemens star with exact ground truth",
    )
    ax.legend(loc="lower left")
    noisy = clean + np.random.default_rng(5).normal(0, 30, clean.shape)
    rgb = np.repeat(np.clip(noisy, 0, 255)[..., None] / 255, 3, axis=2)
    rgb[edges.canny(noisy, 2.5)] = (0.91, 0.35, 0.05)
    axi.imshow(rgb, interpolation="nearest")
    axi.set_title("Canny sigma 2.5 at 30 DN noise (CNR 2)", fontsize=10.5)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.grid(False)
    fig.tight_layout()
    save(fig, path)


def fig_hero(path: Path, quick: bool) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(17, 4.3), gridspec_kw={"width_ratios": [1, 1.25, 1.25, 1.3]}
    )
    cam = star_camera(160 if quick else 300, seed=9)
    cam.exposure_us = 2500.0
    axes[0].imshow(cam.grab(), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("1  Sensor twin capture")
    draw_lab_ptc(axes[1], compact=True)
    axes[1].set_title("2  Photon transfer (real camera)")
    draw_lab_gain(axes[2], compact=True)
    axes[2].set_title("3  Noise vs gain (real camera)")
    img = coins_crop()
    rgb = np.repeat(img[..., None] / 255, 3, axis=2) * 0.85
    rgb[edges.canny(img, 2.0, non_edge_fraction=0.9)] = (0.91, 0.35, 0.05)
    axes[3].imshow(rgb, interpolation="nearest")
    axes[3].set_title("4  Canny edges (real frame)")
    for ax in (axes[0], axes[3]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small frames, for CI")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig_hero(args.out / "hero.png", args.quick)
    fig_lab_photon_transfer(args.out / "lab_photon_transfer.png")
    fig_twin_validation(args.out / "twin_validation.png", args.quick)
    fig_optics(args.out / "optics_tradeoffs.png")
    fig_ladder(args.out / "exposure_vs_gain.png", args.quick)
    fig_denoise(args.out / "denoise_benchmark.png", args.quick)
    fig_edges(args.out / "edges_gallery.png")
    fig_edge_f1(args.out / "edge_f1_vs_noise.png", args.quick)


if __name__ == "__main__":
    main()
