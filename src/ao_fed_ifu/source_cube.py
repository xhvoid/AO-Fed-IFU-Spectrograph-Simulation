"""Toy science datacubes ``I(x, y, lambda)`` and the photon budget.

A *datacube* is the object an IFU is ultimately trying to recover: at every
point on the sky ``(x, y)`` it stores a spectrum over wavelength ``lambda``.
Here we build small synthetic cubes whose *truth* is known, so that after the
full forward+inverse pipeline we can measure how well the input was recovered.

Array convention (see package docstring): a cube has shape
``(ny, nx, nlambda)`` and is indexed ``cube[iy, ix, il]``.  Values are in
**photo-electrons per spaxel per wavelength bin per exposure** once the photon
budget has been applied; before that the helper builders return *relative*
intensities that the photon budget then scales.

Physical content of a cube
    * a smooth continuum (from a source spectrum / blackbody),
    * optional emission line(s) (Gaussian in wavelength),
    * an optional line-of-sight velocity field, which Doppler-shifts the line
      centre spaxel-by-spaxel -- this is what the velocity diagnostic recovers,
    * a spatial brightness profile (point source, Gaussian, or Sersic-like).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SPEED_OF_LIGHT_KM_S
from .data_sources import blackbody_relative

# Rest-frame vacuum wavelengths of common nebular emission lines, in micrometres.
# Textbook atomic constants (literature_derived) used by the high-z science case
# to place a redshifted line into the simulated near-IR band.
REST_FRAME_LINES_UM: dict[str, float] = {
    "Halpha": 0.656281,   # H-alpha
    "Hbeta": 0.486135,    # H-beta
    "OIII": 0.500824,     # [O III] 5008
    "OII": 0.372710,      # [O II] 3727 (doublet, treated as one line here)
}


def observed_wavelength_um(rest_wavelength_um: float, redshift: float) -> float:
    """Redshifted (observed) wavelength: ``lambda_obs = lambda_rest (1 + z)``."""
    return rest_wavelength_um * (1.0 + redshift)


@dataclass
class SourceCubeConfig:
    """Geometry, sampling and photometry of a synthetic datacube.

    Parameters
    ----------
    nx, ny
        Number of spaxels (spatial samples) along x and y.
    nlambda
        Number of wavelength bins.
    spaxel_scale_mas
        Angular size of one spaxel on the sky, in milli-arcseconds.
        HARMONI/MUSE-inspired fine scales are a few mas to tens of mas.
    lambda_min_um, lambda_max_um
        Wavelength range covered by the cube, in micrometres.
    exposure_s
        Exposure time in seconds (sets the photon count via the budget).
    telescope_diameter_m
        Primary-mirror diameter in metres (sets the collecting area).
    throughput_scalar
        End-to-end dimensionless throughput (0..1): the fraction of photons
        above the atmosphere that become detected electrons.  A single scalar
        here -- a real instrument has a wavelength-dependent throughput curve.
    """

    nx: int = 16
    ny: int = 16
    nlambda: int = 64
    spaxel_scale_mas: float = 10.0
    lambda_min_um: float = 1.95
    lambda_max_um: float = 2.45
    exposure_s: float = 600.0
    telescope_diameter_m: float = 8.2
    throughput_scalar: float = 0.25

    @property
    def wavelength_grid_um(self) -> np.ndarray:
        """The wavelength bin centres in micrometres, shape ``(nlambda,)``."""
        return np.linspace(self.lambda_min_um, self.lambda_max_um, self.nlambda)

    @property
    def collecting_area_m2(self) -> float:
        """Geometric collecting area of a circular aperture, ``pi (D/2)^2``.

        This ignores the central obstruction; for a toy photon budget that is a
        deliberately small simplification (engineering_estimate level).
        """
        return np.pi * (self.telescope_diameter_m / 2.0) ** 2

    @property
    def delta_lambda_um(self) -> float:
        """Width of one wavelength bin in micrometres."""
        return (self.lambda_max_um - self.lambda_min_um) / max(self.nlambda - 1, 1)


def _gaussian_line(wavelength_um: np.ndarray, centre_um: float, sigma_um: float) -> np.ndarray:
    """Unit-amplitude Gaussian emission-line profile over a wavelength grid."""
    return np.exp(-0.5 * ((wavelength_um - centre_um) / sigma_um) ** 2)


def _spatial_grid(cfg: SourceCubeConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xx, yy)`` spaxel-index grids centred on the field centre.

    Coordinates are in spaxel units (not mas) with the origin at the geometric
    centre of the field, which is convenient for radial/elliptical profiles and
    for a linear velocity gradient.
    """
    x = np.arange(cfg.nx) - (cfg.nx - 1) / 2.0
    y = np.arange(cfg.ny) - (cfg.ny - 1) / 2.0
    xx, yy = np.meshgrid(x, y)  # shape (ny, nx)
    return xx, yy


def _velocity_field(
    cfg: SourceCubeConfig,
    velocity_gradient_kms_per_spaxel: float,
    position_angle_deg: float = 0.0,
) -> np.ndarray:
    """A simple linear line-of-sight velocity field, shape ``(ny, nx)``.

    Models ordered rotation as a velocity that increases linearly along a chosen
    position angle.  Positive values are red-shifts (receding).  The diagnostic
    module recovers this field from the reconstructed cube as a key science test.
    """
    xx, yy = _spatial_grid(cfg)
    pa = np.deg2rad(position_angle_deg)
    projected = xx * np.cos(pa) + yy * np.sin(pa)
    return velocity_gradient_kms_per_spaxel * projected


def _apply_continuum_and_line(
    cfg: SourceCubeConfig,
    spatial_profile: np.ndarray,
    continuum_spectrum: np.ndarray,
    line_center_um: float | None,
    line_flux_rel: float,
    line_sigma_um: float,
    velocity_field_kms: np.ndarray | None,
) -> np.ndarray:
    """Assemble a cube from a spatial profile, a continuum, and an optional line.

    Parameters
    ----------
    spatial_profile
        Shape ``(ny, nx)`` relative brightness at each spaxel.
    continuum_spectrum
        Shape ``(nlambda,)`` relative continuum shape (1-D, same everywhere).
    line_center_um
        Rest-frame emission-line centre, or ``None`` for continuum only.
    line_flux_rel
        Peak amplitude of the line relative to the continuum normalisation.
    line_sigma_um
        Intrinsic Gaussian sigma of the line in micrometres.
    velocity_field_kms
        Optional ``(ny, nx)`` velocity field that Doppler-shifts the line centre
        per spaxel: ``lambda_obs = lambda_rest (1 + v/c)``.

    Returns
    -------
    cube
        Shape ``(ny, nx, nlambda)`` of relative intensities.
    """
    wl = cfg.wavelength_grid_um
    # Start from continuum replicated over the field, weighted by the spatial profile.
    cube = spatial_profile[:, :, None] * continuum_spectrum[None, None, :]

    if line_center_um is not None:
        if velocity_field_kms is None:
            line = _gaussian_line(wl, line_center_um, line_sigma_um)
            cube = cube + line_flux_rel * spatial_profile[:, :, None] * line[None, None, :]
        else:
            # Per-spaxel Doppler-shifted line centre.  Vectorised over the field.
            shifted_centres = line_center_um * (1.0 + velocity_field_kms / SPEED_OF_LIGHT_KM_S)
            # Broadcast: (ny, nx, 1) centres against (1, 1, nlambda) wavelengths.
            arg = (wl[None, None, :] - shifted_centres[:, :, None]) / line_sigma_um
            line_cube = np.exp(-0.5 * arg**2)
            cube = cube + line_flux_rel * spatial_profile[:, :, None] * line_cube
    return cube


def create_point_source_cube(
    cfg: SourceCubeConfig,
    source_temperature_k: float = 6000.0,
    line_center_um: float | None = None,
    line_flux_rel: float = 5.0,
    line_sigma_um: float = 0.002,
) -> np.ndarray:
    """A single bright spaxel (delta-like point source) with continuum + line.

    Useful for PSF / encircled-energy work where all the flux starts in one
    spaxel.  The spatial profile is a single non-zero central spaxel.
    """
    profile = np.zeros((cfg.ny, cfg.nx))
    profile[cfg.ny // 2, cfg.nx // 2] = 1.0
    continuum = blackbody_relative(cfg.wavelength_grid_um, source_temperature_k)
    return _apply_continuum_and_line(
        cfg, profile, continuum, line_center_um, line_flux_rel, line_sigma_um, None
    )


def create_emission_line_cube(
    cfg: SourceCubeConfig,
    line_center_um: float | None = None,
    line_flux_rel: float = 5.0,
    line_sigma_um: float = 0.002,
    velocity_gradient_kms_per_spaxel: float = 0.0,
    position_angle_deg: float = 0.0,
    continuum_level: float = 1.0,
) -> np.ndarray:
    """Uniformly-illuminated field with a continuum, an emission line and a
    (optional) velocity gradient.

    This is the work-horse cube for the velocity and line-flux diagnostics: the
    whole field shares a flat continuum so the only spatial structure is the
    Doppler shift of the line centre.

    If ``line_center_um`` is ``None`` it defaults to the middle of the band.
    """
    if line_center_um is None:
        line_center_um = 0.5 * (cfg.lambda_min_um + cfg.lambda_max_um)
    profile = np.full((cfg.ny, cfg.nx), 1.0)
    continuum = np.full(cfg.nlambda, continuum_level)
    vfield = _velocity_field(cfg, velocity_gradient_kms_per_spaxel, position_angle_deg)
    return _apply_continuum_and_line(
        cfg, profile, continuum, line_center_um, line_flux_rel, line_sigma_um, vfield
    )


def create_extended_galaxy_cube(
    cfg: SourceCubeConfig,
    half_light_radius_spaxel: float = 3.0,
    sersic_index: float = 1.0,
    axis_ratio: float = 0.6,
    position_angle_deg: float = 30.0,
    source_temperature_k: float = 5000.0,
    line_center_um: float | None = None,
    line_flux_rel: float = 3.0,
    line_sigma_um: float = 0.002,
    velocity_gradient_kms_per_spaxel: float = 40.0,
) -> np.ndarray:
    """An extended source with a Sersic-like profile and a rotation field.

    The spatial brightness follows a Sersic profile

        I(r) ~ exp(-b_n * ((r / r_e)^(1/n) - 1))

    where ``r`` is the elliptical radius (set by ``axis_ratio`` and
    ``position_angle_deg``) and ``b_n`` is the usual Sersic constant.  For
    ``n = 1`` this is an exponential disc; for ``n = 4`` a de Vaucouleurs bulge.
    A linear velocity gradient along the major axis mimics ordered rotation.
    """
    if line_center_um is None:
        line_center_um = 0.5 * (cfg.lambda_min_um + cfg.lambda_max_um)

    xx, yy = _spatial_grid(cfg)
    pa = np.deg2rad(position_angle_deg)
    # Rotate into the galaxy frame, then apply the minor-axis compression.
    x_rot = xx * np.cos(pa) + yy * np.sin(pa)
    y_rot = -xx * np.sin(pa) + yy * np.cos(pa)
    r_ell = np.sqrt(x_rot**2 + (y_rot / max(axis_ratio, 1e-3)) ** 2)

    # Sersic b_n approximation (Ciotti & Bertin 1999) valid for n >~ 0.36.
    n = sersic_index
    b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    profile = np.exp(-b_n * ((r_ell / max(half_light_radius_spaxel, 1e-3)) ** (1.0 / n) - 1.0))
    profile = profile / np.max(profile)

    continuum = blackbody_relative(cfg.wavelength_grid_um, source_temperature_k)
    # Velocity gradient acts along the galaxy major axis (x_rot direction).
    vfield = velocity_gradient_kms_per_spaxel * x_rot
    return _apply_continuum_and_line(
        cfg, profile, continuum, line_center_um, line_flux_rel, line_sigma_um, vfield
    )


def photon_budget_from_abmag(
    cfg: SourceCubeConfig,
    ab_magnitude: float,
    band_throughput: float | None = None,
) -> float:
    """Convert an AB magnitude into detected electrons **per wavelength bin**.

    The AB system defines a flux density per unit frequency:

        f_nu = 10^(-0.4 (m_AB + 48.60))   erg / s / cm^2 / Hz

    We convert that to a photon rate, multiply by the collecting area, the
    exposure time, the throughput, and the *frequency* width of one wavelength
    bin, to get the number of detected electrons in that bin.

    Steps (all SI unless noted):
        1.  f_nu [erg/s/cm^2/Hz] -> [J/s/m^2/Hz]      (x 1e-7 erg->J, x 1e4 cm^2->m^2)
        2.  photon energy at the band centre  E = h nu
        3.  photons/s/m^2/Hz = f_nu / E
        4.  x area x exposure x throughput x delta_nu(bin)  -> electrons/bin

    Parameters
    ----------
    ab_magnitude
        Source AB magnitude (a brightness; smaller = brighter).
    band_throughput
        Optional throughput override; defaults to ``cfg.throughput_scalar``.

    Returns
    -------
    electrons_per_bin
        Detected electrons in a single wavelength bin for the whole source.

    Notes
    -----
    This is a deliberately simplified, *engineering-estimate* photon budget --
    it uses the band-centre frequency for every bin and a single scalar
    throughput.  It is good enough to set realistic count *levels* so that the
    detector-noise regime (read-noise- vs photon-limited) is meaningful.
    """
    h = 6.626_070_15e-34  # J s
    c = 2.997_924_58e8  # m / s
    throughput = cfg.throughput_scalar if band_throughput is None else band_throughput

    # AB flux density in SI: erg/s/cm^2/Hz -> J/s/m^2/Hz is a factor 1e-7 * 1e4 = 1e-3.
    f_nu_cgs = 10.0 ** (-0.4 * (ab_magnitude + 48.60))  # erg/s/cm^2/Hz
    f_nu_si = f_nu_cgs * 1e-3  # J/s/m^2/Hz

    lam_centre_m = 0.5 * (cfg.lambda_min_um + cfg.lambda_max_um) * 1e-6
    nu_centre = c / lam_centre_m
    photon_energy = h * nu_centre  # J per photon at band centre

    # Frequency width of one wavelength bin: |dnu| = c / lambda^2 * dlambda.
    delta_lambda_m = cfg.delta_lambda_um * 1e-6
    delta_nu = c / lam_centre_m**2 * delta_lambda_m

    photon_rate_density = f_nu_si / photon_energy  # photons/s/m^2/Hz
    electrons_per_bin = (
        photon_rate_density
        * cfg.collecting_area_m2
        * cfg.exposure_s
        * throughput
        * delta_nu
    )
    return float(electrons_per_bin)


def scale_cube_to_total_electrons(cube: np.ndarray, total_electrons: float) -> np.ndarray:
    """Rescale a relative-intensity cube so it sums to ``total_electrons``.

    Convenience used by the pipeline: take a unit-less cube from one of the
    ``create_*`` builders and put it on an absolute electron scale derived from
    the photon budget, preserving the cube's spatial/spectral shape.
    """
    total = float(np.sum(cube))
    if total <= 0:
        raise ValueError("Cube has non-positive total flux; cannot rescale.")
    return cube * (total_electrons / total)
