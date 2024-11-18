"""Thin-lens optics for machine-vision setups.

All lengths are millimetres unless the name says otherwise (``_um`` = micrometres).
``distance_mm`` is the object distance measured from the lens' front principal plane,
which for compact C-mount lenses is close to the usual "working distance".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    """An area-scan image sensor."""

    name: str
    width_px: int
    height_px: int
    pixel_um: float

    @property
    def width_mm(self) -> float:
        return self.width_px * self.pixel_um / 1000.0

    @property
    def height_mm(self) -> float:
        return self.height_px * self.pixel_um / 1000.0

    @property
    def diagonal_mm(self) -> float:
        return math.hypot(self.width_mm, self.height_mm)


@dataclass(frozen=True)
class Lens:
    """A fixed-focal-length lens."""

    focal_mm: float
    f_number: float


#: 1/3" interline progressive-scan CCD used for the lab measurements (datasheet values).
BAUMER_TXG08 = Sensor("Baumer TXG08 (1/3in CCD)", 1032, 776, 4.65)


def magnification(focal_mm: float, distance_mm: float) -> float:
    """Lateral magnification ``|m| = f / (a - f)`` of a thin lens focused at distance ``a``."""
    if distance_mm <= focal_mm:
        raise ValueError("the object must be farther away than one focal length")
    return focal_mm / (distance_mm - focal_mm)


def angle_of_view_deg(size_mm: float, focal_mm: float) -> float:
    """Full angle of view ``2 atan(d / 2f)`` subtended by a sensor dimension ``d``."""
    return math.degrees(2.0 * math.atan(size_mm / (2.0 * focal_mm)))


def field_of_view_mm(
    size_mm: float, focal_mm: float, distance_mm: float, model: str = "thin-lens"
) -> float:
    """Object-side extent imaged onto a sensor dimension ``size_mm``.

    ``model="thin-lens"`` uses the exact magnification ``f / (a - f)``;
    ``model="pinhole"`` uses the common far-field shortcut ``d * a / f``.
    """
    if model == "thin-lens":
        return size_mm / magnification(focal_mm, distance_mm)
    if model == "pinhole":
        return size_mm * distance_mm / focal_mm
    raise ValueError(f"unknown model {model!r}; use 'thin-lens' or 'pinhole'")


def depth_of_field_mm(
    focal_mm: float, f_number: float, distance_mm: float, coc_mm: float
) -> tuple[float, float]:
    """Near and far limit of acceptable sharpness.

    ``a_{n,f} = a / (1 +- N u (a - f) / f^2)`` with circle of confusion ``u``.
    The far limit is ``math.inf`` at or beyond the hyperfocal distance.
    """
    x = f_number * coc_mm * (distance_mm - focal_mm) / focal_mm**2
    near = distance_mm / (1.0 + x)
    far = distance_mm / (1.0 - x) if x < 1.0 else math.inf
    return near, far


def hyperfocal_mm(focal_mm: float, f_number: float, coc_mm: float) -> float:
    """Closest focus distance whose depth of field extends to infinity."""
    return focal_mm**2 / (f_number * coc_mm) + focal_mm


def airy_diameter_um(f_number: float, wavelength_nm: float = 550.0) -> float:
    """Diameter of the Airy disk up to its first zero, ``2.44 lambda N``."""
    return 2.44 * wavelength_nm * 1e-3 * f_number


def diffraction_limited_f_number(
    pixel_um: float, wavelength_nm: float = 550.0, pixels_per_airy: float = 2.0
) -> float:
    """f-number at which the Airy disk spans ``pixels_per_airy`` pixels.

    Stopping down further buys depth of field but blurs detail that the pixel grid
    could otherwise resolve.
    """
    return pixels_per_airy * pixel_um / (2.44 * wavelength_nm * 1e-3)


@dataclass(frozen=True)
class OpticalSetup:
    """A sensor, a lens and an object distance - everything needed for a field calculation."""

    sensor: Sensor
    lens: Lens
    distance_mm: float
    coc_um: float | None = None

    @property
    def coc_mm(self) -> float:
        """Circle of confusion; defaults to one pixel pitch."""
        coc_um = self.sensor.pixel_um if self.coc_um is None else self.coc_um
        return coc_um / 1000.0

    @property
    def magnification(self) -> float:
        return magnification(self.lens.focal_mm, self.distance_mm)

    def field_of_view_mm(self, model: str = "thin-lens") -> tuple[float, float]:
        f, a = self.lens.focal_mm, self.distance_mm
        return (
            field_of_view_mm(self.sensor.width_mm, f, a, model),
            field_of_view_mm(self.sensor.height_mm, f, a, model),
        )

    def angle_of_view_deg(self) -> tuple[float, float, float]:
        f = self.lens.focal_mm
        return (
            angle_of_view_deg(self.sensor.width_mm, f),
            angle_of_view_deg(self.sensor.height_mm, f),
            angle_of_view_deg(self.sensor.diagonal_mm, f),
        )

    def object_pixel_mm(self, model: str = "thin-lens") -> float:
        """Footprint of one pixel in the object plane (ground sampling distance)."""
        return field_of_view_mm(
            self.sensor.pixel_um / 1000.0, self.lens.focal_mm, self.distance_mm, model
        )

    def depth_of_field_mm(self) -> tuple[float, float, float]:
        near, far = depth_of_field_mm(
            self.lens.focal_mm, self.lens.f_number, self.distance_mm, self.coc_mm
        )
        return near, far, far - near

    def hyperfocal_mm(self) -> float:
        return hyperfocal_mm(self.lens.focal_mm, self.lens.f_number, self.coc_mm)

    def summary(self) -> dict[str, float]:
        """Every derived quantity as a flat, printable dictionary."""
        fov_w, fov_h = self.field_of_view_mm()
        pin_w, pin_h = self.field_of_view_mm("pinhole")
        aov_w, aov_h, aov_d = self.angle_of_view_deg()
        near, far, dof = self.depth_of_field_mm()
        return {
            "sensor width [mm]": self.sensor.width_mm,
            "sensor height [mm]": self.sensor.height_mm,
            "magnification [-]": self.magnification,
            "field of view width [mm]": fov_w,
            "field of view height [mm]": fov_h,
            "field of view width, pinhole [mm]": pin_w,
            "field of view height, pinhole [mm]": pin_h,
            "angle of view horizontal [deg]": aov_w,
            "angle of view vertical [deg]": aov_h,
            "angle of view diagonal [deg]": aov_d,
            "object-plane pixel [mm]": self.object_pixel_mm(),
            "depth of field near [mm]": near,
            "depth of field far [mm]": far,
            "depth of field total [mm]": dof,
            "hyperfocal distance [mm]": self.hyperfocal_mm(),
            "Airy disk diameter [um]": airy_diameter_um(self.lens.f_number),
            "diffraction-limited f-number [-]": diffraction_limited_f_number(self.sensor.pixel_um),
        }
