"""A configurable high-z galaxy source: morphology, kinematics, multi-line, dust.

Upgrades the single-line toy galaxy into an astrophysical *proxy* with a Sersic
light profile, an inclined rotating disc (arctangent rotation curve), an intrinsic
velocity-dispersion broadening, several redshifted emission lines and an optional
dust attenuation law.

Everything is a controlled synthetic source for *recovery* experiments, **not**
measured galaxy physics:
    * rest wavelengths are ``literature_derived`` atomic constants
      (``source_cube.REST_FRAME_LINES_UM``);
    * line ratios are ``engineering_estimate`` typical HII-region values;
    * morphology / kinematics / dust are ``synthetic_proxy`` constructions.

Rotation-curve vocabulary follows the arctangent form (e.g. Courteau 1997,
1997AJ....114.2402C); the Sersic b_n uses Ciotti & Bertin 1999.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import SPEED_OF_LIGHT_KM_S
from .source_cube import REST_FRAME_LINES_UM, observed_wavelength_um

# Typical HII-region line ratios relative to H-alpha (engineering_estimate; not a
# cited measurement).  Used when the caller does not supply ratios.
DEFAULT_LINE_RATIOS: dict[str, float] = {
    "Halpha": 1.0, "NII6548": 0.10, "NII6583": 0.30,
    "Hbeta": 0.35, "OIII4959": 0.23, "OIII5007": 0.70,
    "OII3726": 0.25, "OII3729": 0.25,
}


@dataclass
class GalaxySourceConfig:
    """Morphology + kinematics + dust of a synthetic high-z galaxy."""

    nx: int = 24
    ny: int = 24
    sersic_index: float = 1.0
    half_light_radius_spaxel: float = 4.0
    position_angle_deg: float = 30.0
    inclination_deg: float = 60.0           # 0 = face-on, 90 = edge-on
    rotation_amplitude_kms: float = 180.0
    turnover_radius_spaxel: float = 2.0
    velocity_dispersion_kms: float = 60.0
    dust_attenuation_av_mag: float = 0.0    # A_V-like scalar (0 = no dust)
    dust_wavelength_law: bool = False       # if True, bluer rest lines attenuated more

    @property
    def axis_ratio(self) -> float:
        """Projected minor/major axis ratio b/a = cos(inclination) (thin disc)."""
        return float(max(np.cos(np.deg2rad(self.inclination_deg)), 0.05))


def arctan_rotation_curve(r, v_max: float, r_turnover: float) -> np.ndarray:
    """Arctangent rotation curve ``v(r) = (2/pi) v_max arctan(r / r_turnover)``.

    Rises through the turnover radius and flattens to ``v_max`` at large radius.
    """
    r_turnover = max(float(r_turnover), 1e-6)
    return (2.0 / np.pi) * float(v_max) * np.arctan(np.asarray(r, dtype=float) / r_turnover)


def _disk_grids(cfg: GalaxySourceConfig):
    """Return ``(radius_disk, cos_azimuth, ellipse_radius)`` arrays, shape (ny, nx)."""
    x = np.arange(cfg.nx) - (cfg.nx - 1) / 2.0
    y = np.arange(cfg.ny) - (cfg.ny - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    pa = np.deg2rad(cfg.position_angle_deg)
    # Rotate into the disc major/minor frame.
    x_maj = xx * np.cos(pa) + yy * np.sin(pa)
    y_min = -xx * np.sin(pa) + yy * np.cos(pa)
    cosi = cfg.axis_ratio
    y_disk = y_min / cosi                      # deproject the minor axis
    r_disk = np.sqrt(x_maj**2 + y_disk**2)
    cos_az = np.where(r_disk > 0, x_maj / np.maximum(r_disk, 1e-9), 0.0)
    ellipse_r = np.sqrt(x_maj**2 + (y_min / cosi) ** 2)  # for the Sersic profile
    return r_disk, cos_az, ellipse_r


def projected_velocity_field(cfg: GalaxySourceConfig) -> np.ndarray:
    """Line-of-sight velocity field [km/s] of an inclined rotating disc, (ny, nx).

    ``v_los = v_rot(r) * sin(inclination) * cos(azimuth)`` -- zero for a face-on
    disc, maximal edge-on.
    """
    r_disk, cos_az, _ = _disk_grids(cfg)
    inc = np.deg2rad(cfg.inclination_deg)
    v_rot = arctan_rotation_curve(r_disk, cfg.rotation_amplitude_kms, cfg.turnover_radius_spaxel)
    return v_rot * np.sin(inc) * cos_az


def sersic_light_profile(cfg: GalaxySourceConfig) -> np.ndarray:
    """Projected Sersic surface-brightness profile, normalised to unit peak."""
    _, _, ellipse_r = _disk_grids(cfg)
    n = cfg.sersic_index
    b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    re = max(cfg.half_light_radius_spaxel, 1e-3)
    profile = np.exp(-b_n * ((ellipse_r / re) ** (1.0 / n) - 1.0))
    return profile / np.max(profile)


def dust_attenuation_factor(cfg: GalaxySourceConfig, rest_wavelength_um: float) -> float:
    """Multiplicative dust attenuation for a line at a given *rest* wavelength.

    ``factor = 10^(-0.4 A)`` with ``A = A_V`` (grey) or, if the wavelength law is
    enabled, ``A = A_V * (0.55 um / rest_lambda)`` so bluer lines are attenuated
    more (a simple, illustrative law -- not a calibrated extinction curve).
    """
    if cfg.dust_attenuation_av_mag <= 0:
        return 1.0
    a = cfg.dust_attenuation_av_mag
    if cfg.dust_wavelength_law:
        a = a * (0.55 / float(rest_wavelength_um))
    return float(10.0 ** (-0.4 * a))


def build_galaxy_source_cube(
    cfg: GalaxySourceConfig,
    wavelength_grid_um: np.ndarray,
    redshift: float,
    lines: list[str] | None = None,
    line_ratios: dict[str, float] | None = None,
    band_margin_um: float = 0.0,
) -> np.ndarray:
    """Build the multi-line, rotating, dust-attenuated galaxy datacube.

    Each requested line is placed at ``(1 + z) * rest_lambda``, Doppler-shifted
    per spaxel by the projected rotation, broadened by the velocity dispersion,
    scaled by its line ratio and dust factor, and weighted by the Sersic profile.
    Lines whose observed centre falls outside the wavelength grid contribute
    nothing (they are simply off-band).

    Raises ``ValueError`` if a line name is unknown or if *no* requested line
    lands inside the wavelength grid (an out-of-band redshift/line choice).
    """
    wl = np.asarray(wavelength_grid_um, dtype=float)
    lines = list(lines) if lines is not None else ["Halpha", "NII6548", "NII6583"]
    ratios = line_ratios if line_ratios is not None else DEFAULT_LINE_RATIOS
    for ln in lines:
        if ln not in REST_FRAME_LINES_UM:
            raise ValueError(f"unknown line {ln!r}; choose from {sorted(REST_FRAME_LINES_UM)}")

    profile = sersic_light_profile(cfg)
    v_los = projected_velocity_field(cfg)
    lo, hi = wl.min() - band_margin_um, wl.max() + band_margin_um

    cube = np.zeros((cfg.ny, cfg.nx, wl.size), dtype=float)
    in_band = []
    for ln in lines:
        obs = observed_wavelength_um(REST_FRAME_LINES_UM[ln], redshift)
        if not (lo <= obs <= hi):
            continue
        in_band.append(ln)
        sigma_lam = max(obs * cfg.velocity_dispersion_kms / SPEED_OF_LIGHT_KM_S, 1e-6)
        centre = obs * (1.0 + v_los / SPEED_OF_LIGHT_KM_S)              # (ny, nx)
        line_cube = np.exp(-0.5 * ((wl[None, None, :] - centre[:, :, None]) / sigma_lam) ** 2)
        weight = float(ratios.get(ln, 1.0)) * dust_attenuation_factor(cfg, REST_FRAME_LINES_UM[ln])
        cube += profile[:, :, None] * weight * line_cube

    if not in_band:
        raise ValueError(
            f"no requested line {lines} lands in the wavelength grid "
            f"[{wl.min():.3f}-{wl.max():.3f}] um at z={redshift}; "
            "choose a redshift / line set that places a line in band."
        )
    return cube


def lines_in_band(lines: list[str], wavelength_grid_um: np.ndarray, redshift: float) -> list[str]:
    """Subset of ``lines`` whose observed wavelength lies in the grid."""
    wl = np.asarray(wavelength_grid_um, dtype=float)
    return [ln for ln in lines if ln in REST_FRAME_LINES_UM
            and wl.min() <= observed_wavelength_um(REST_FRAME_LINES_UM[ln], redshift) <= wl.max()]
