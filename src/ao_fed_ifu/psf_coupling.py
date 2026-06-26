"""Point-spread-function (PSF) proxies and coupling of light into spaxels.

The PSF is the image an instrument forms of a point source.  Its size sets how
sharply the IFU samples the sky, and therefore how much of a source's light
falls into a single spaxel (the *encircled energy*).  This module provides
three deliberately simple, **synthetic** PSF proxies:

    * ``airy_psf_proxy``         -- diffraction-limited (lambda / D) core,
    * ``seeing_gaussian_psf``    -- atmospheric seeing (a wide Gaussian),
    * ``ao_corrected_psf_proxy`` -- a Strehl-weighted core+halo combination
                                    that mimics partial AO correction.

None of these reproduce a real AO system; they are physically-motivated
stand-ins (``synthetic_proxy``) that capture the trends an AO-fed IFU cares
about: higher Strehl or smaller seeing concentrates more light into the central
spaxel.

All PSFs are returned as 2-D arrays normalised to unit total flux, sampled on a
square grid whose pixel scale is given in milli-arcseconds (mas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1  # Bessel function J1, for the Airy pattern

# Conversion factor: 1 radian in milli-arcseconds.
_RAD_TO_MAS = 180.0 / np.pi * 3600.0 * 1000.0


@dataclass
class PSFMetrics:
    """Summary metrics describing how a PSF couples into spaxels.

    Attributes
    ----------
    fwhm_mas
        Full width at half maximum of the PSF core, in mas.
    centroid_shift_mas
        Distance of the flux-weighted centroid from the grid centre, in mas
        (a measure of asymmetry / mis-centring; ~0 for a centred PSF).
    ee1_spaxel
        Encircled energy in the single central spaxel (0..1).
    ee3x3_spaxel
        Encircled energy in the central 3x3 block of spaxels (0..1).
    undersampled
        True if the PSF FWHM spans fewer than two spaxels (Nyquist), i.e. the
        spaxel scale is too coarse to sample the PSF.
    """

    fwhm_mas: float
    centroid_shift_mas: float
    ee1_spaxel: float
    ee3x3_spaxel: float
    undersampled: bool


def _coordinate_grid(pixel_scale_mas: float, n_pix: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xx, yy, rr)`` coordinate arrays in mas, centred on the grid.

    ``n_pix`` is forced odd so there is a well-defined central pixel, which
    keeps the diffraction core and the central spaxel concentric.
    """
    if n_pix % 2 == 0:
        n_pix += 1
    half = (n_pix - 1) / 2.0
    axis = (np.arange(n_pix) - half) * pixel_scale_mas
    xx, yy = np.meshgrid(axis, axis)
    rr = np.hypot(xx, yy)
    return xx, yy, rr


def diffraction_fwhm_mas(lambda_um: float, telescope_diameter_m: float) -> float:
    """Diffraction-limited FWHM ~ 1.025 lambda / D, returned in mas.

    The factor 1.025 is the standard FWHM of an Airy pattern (the first dark
    ring is at 1.22 lambda/D; the half-power radius is smaller).
    """
    lam_over_d_rad = (lambda_um * 1e-6) / telescope_diameter_m
    return 1.025 * lam_over_d_rad * _RAD_TO_MAS


def airy_psf_proxy(
    lambda_um: float,
    telescope_diameter_m: float,
    grid_mas: float,
    n_pix: int = 129,
) -> np.ndarray:
    """Diffraction-limited Airy PSF for a circular aperture.

    The Airy intensity pattern of an unobstructed circular aperture is

        I(theta) = [ 2 J1(x) / x ]^2 ,   x = pi D theta / lambda

    where ``theta`` is the angular distance from the centre and ``J1`` is the
    first-order Bessel function.  The result is normalised to unit total flux.

    Parameters
    ----------
    lambda_um
        Wavelength in micrometres (the diffraction scale is lambda/D).
    telescope_diameter_m
        Aperture diameter in metres.
    grid_mas
        Pixel scale of the returned PSF image, in mas.
    n_pix
        Grid size (forced odd).  Should be large enough to contain the core.
    """
    _, _, rr = _coordinate_grid(grid_mas, n_pix)
    theta_rad = rr / _RAD_TO_MAS
    x = np.pi * telescope_diameter_m * theta_rad / (lambda_um * 1e-6)
    # Handle the removable singularity at x = 0 where 2 J1(x)/x -> 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
    psf = amp**2
    return psf / np.sum(psf)


def seeing_gaussian_psf(
    fwhm_mas: float,
    grid_mas: float,
    n_pix: int = 129,
) -> np.ndarray:
    """Seeing-limited Gaussian PSF with the given FWHM.

    Atmospheric seeing (without AO) blurs point sources into an approximately
    Gaussian core.  ``sigma = FWHM / (2 sqrt(2 ln 2))``.  Normalised to unit flux.
    """
    sigma = fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    _, _, rr = _coordinate_grid(grid_mas, n_pix)
    psf = np.exp(-0.5 * (rr / sigma) ** 2)
    return psf / np.sum(psf)


def ao_corrected_psf_proxy(
    strehl: float,
    lambda_um: float,
    telescope_diameter_m: float,
    seeing_fwhm_mas: float,
    grid_mas: float,
    n_pix: int = 129,
) -> np.ndarray:
    """Strehl-weighted core+halo PSF proxy for partial AO correction.

    A partially-corrected AO PSF is well described by two components:

        * a sharp diffraction-limited **core** (FWHM ~ lambda/D) that carries a
          fraction of the energy roughly equal to the Strehl ratio, and
        * a broad seeing-limited **halo** that carries the remaining energy.

    We model the core as a Gaussian at the diffraction FWHM (a smooth, robust
    proxy for the Airy core) and the halo as a Gaussian at the seeing FWHM:

        PSF = S * core + (1 - S) * halo        (then renormalised to unit flux)

    The Strehl ratio ``S`` in [0, 1] is the only AO "quality" knob: S -> 1 gives
    a near-diffraction-limited PSF, S -> 0 gives the seeing PSF.

    This is a ``synthetic_proxy`` -- it captures the energy-concentration trend
    of AO correction without modelling residual wavefront error in detail.
    """
    strehl = float(np.clip(strehl, 0.0, 1.0))
    core_fwhm = diffraction_fwhm_mas(lambda_um, telescope_diameter_m)

    sigma_core = core_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_halo = seeing_fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    _, _, rr = _coordinate_grid(grid_mas, n_pix)

    core = np.exp(-0.5 * (rr / sigma_core) ** 2)
    core /= np.sum(core)
    halo = np.exp(-0.5 * (rr / sigma_halo) ** 2)
    halo /= np.sum(halo)

    psf = strehl * core + (1.0 - strehl) * halo
    return psf / np.sum(psf)


def convolve_cube_with_psf(
    cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    psf_builder,
    spaxel_scale_mas: float,
    kernel_n_pix: int = 25,
    conserve_flux: bool = False,
) -> np.ndarray:
    """Convolve every wavelength plane of a cube with a (possibly wavelength-
    dependent) PSF sampled at the spaxel scale.

    Boundary handling and flux
    --------------------------
    A plain ``mode="same"`` FFT convolution uses a **zero** boundary, so light
    from a source near the field edge is blurred *off* the field and lost.  This
    is physically real (a finite field-of-view does lose light to the PSF wings),
    so it is the **default** here -- but the pipeline always reports the realised
    field throughput (``psf_field_throughput``) so the loss is never silent.

    With ``conserve_flux=True`` the blurred plane is divided by the kernel
    convolved with a field of ones (**edge-normalisation**).  This restores the
    flux a real, larger surrounding scene would have scattered back in, conserving
    flux *exactly for a locally-uniform scene*.  Use it **only for an extended
    source treated as a sub-field of a larger scene** -- for an isolated point
    source it would invent flux (amplify the wings that physically leave the
    field), so leave it ``False`` in that case.

    Parameters
    ----------
    cube
        Input datacube, shape ``(ny, nx, nlambda)``.
    wavelength_grid_um
        Wavelength of each plane, shape ``(nlambda,)``.  Passed to the builder
        so diffraction-limited PSFs can scale as lambda/D.
    psf_builder
        Callable ``psf_builder(lambda_um, grid_mas, n_pix) -> 2D kernel``.  Use
        ``functools.partial`` to bind telescope diameter, Strehl, seeing, etc.
    spaxel_scale_mas
        Spaxel scale of the cube (the PSF kernel is sampled at this scale so the
        convolution happens directly on the spaxel grid).
    kernel_n_pix
        Size of the convolution kernel in spaxels (forced odd by the grid).
    conserve_flux
        If True, edge-normalise so the field flux is conserved (see above).

    Returns
    -------
    convolved_cube
        Same shape as ``cube``; each plane convolved with its PSF.
    """
    from scipy.signal import fftconvolve

    ny, nx, nlam = cube.shape
    out = np.empty_like(cube, dtype=float)
    ones = np.ones((ny, nx), dtype=float)
    for il in range(nlam):
        kernel = psf_builder(
            lambda_um=float(wavelength_grid_um[il]),
            grid_mas=spaxel_scale_mas,
            n_pix=kernel_n_pix,
        )
        blurred = fftconvolve(cube[:, :, il], kernel, mode="same")
        if conserve_flux:
            # Divide by the in-field fraction of the kernel at each pixel, which
            # compensates for the flux a zero boundary would have lost off-edge.
            norm = fftconvolve(ones, kernel, mode="same")
            blurred = np.divide(
                blurred, norm, out=np.zeros_like(blurred), where=norm > 1e-12
            )
        out[:, :, il] = blurred
    return out


def spaxelize_psf(
    psf_fine: np.ndarray,
    fine_scale_mas: float,
    spaxel_scale_mas: float,
    n_spaxel: int = 7,
) -> np.ndarray:
    """Bin a finely-sampled PSF onto a coarse spaxel grid (flux-conserving).

    The PSF is computed on a fine grid for accuracy; to ask "how much light
    lands in each spaxel" we integrate the fine PSF over spaxel-sized cells.
    A flux-weighted 2-D histogram does this exactly even when the spaxel scale
    is not an integer multiple of the fine scale.

    Returns
    -------
    spaxel_psf
        Array of shape ``(n_spaxel, n_spaxel)`` (n_spaxel forced odd) giving the
        fraction of total PSF flux in each spaxel, centred on the PSF peak.
    """
    if n_spaxel % 2 == 0:
        n_spaxel += 1
    n_fine = psf_fine.shape[0]
    half = (n_fine - 1) / 2.0
    axis = (np.arange(n_fine) - half) * fine_scale_mas
    xx, yy = np.meshgrid(axis, axis)

    half_extent = n_spaxel * spaxel_scale_mas / 2.0
    edges = np.linspace(-half_extent, half_extent, n_spaxel + 1)
    spaxel_psf, _, _ = np.histogram2d(
        yy.ravel(), xx.ravel(), bins=[edges, edges], weights=psf_fine.ravel()
    )
    return spaxel_psf


def encircled_energy_per_spaxel(
    psf_fine: np.ndarray,
    fine_scale_mas: float,
    spaxel_scale_mas: float,
) -> tuple[float, float, np.ndarray]:
    """Return ``(EE1, EE3x3, spaxel_psf)`` encircled-energy metrics.

    * ``EE1``   -- fraction of PSF flux in the single central spaxel,
    * ``EE3x3`` -- fraction in the central 3x3 block of spaxels.

    These are the headline numbers for AO-fed IFU coupling: a good AO correction
    drives EE1 and EE3x3 up by concentrating light into few spaxels.
    """
    spaxel_psf = spaxelize_psf(psf_fine, fine_scale_mas, spaxel_scale_mas, n_spaxel=7)
    total = float(np.sum(spaxel_psf))
    if total <= 0:
        return 0.0, 0.0, spaxel_psf
    c = spaxel_psf.shape[0] // 2  # central index
    ee1 = float(spaxel_psf[c, c]) / total
    ee3x3 = float(np.sum(spaxel_psf[c - 1 : c + 2, c - 1 : c + 2])) / total
    return ee1, ee3x3, spaxel_psf


def psf_fwhm_mas(psf_fine: np.ndarray, fine_scale_mas: float) -> float:
    """Estimate the PSF FWHM from its azimuthally-averaged radial profile (mas).

    We bin the 2-D PSF into radial annuli, find the half-maximum crossing of the
    mean profile, and return twice that radius.  Robust for both the Gaussian
    and Airy proxies used here.
    """
    n = psf_fine.shape[0]
    half = (n - 1) / 2.0
    axis = (np.arange(n) - half) * fine_scale_mas
    xx, yy = np.meshgrid(axis, axis)
    rr = np.hypot(xx, yy).ravel()
    vals = psf_fine.ravel()

    order = np.argsort(rr)
    rr_sorted = rr[order]
    vals_sorted = vals[order]
    # Smooth the radial profile by averaging within thin radial bins.
    nbin = max(n // 2, 8)
    r_bins = np.linspace(0, rr_sorted.max(), nbin)
    prof = np.array(
        [
            vals_sorted[(rr_sorted >= r_bins[i]) & (rr_sorted < r_bins[i + 1])].mean()
            if np.any((rr_sorted >= r_bins[i]) & (rr_sorted < r_bins[i + 1]))
            else np.nan
            for i in range(nbin - 1)
        ]
    )
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    good = ~np.isnan(prof)
    prof, r_centres = prof[good], r_centres[good]
    if prof.size < 2:
        return float("nan")
    peak = prof[0]
    half_level = peak / 2.0
    # First radius where the profile falls below half the peak.
    below = np.where(prof <= half_level)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(2.0 * r_centres[0])
    # Linear interpolation between the bracketing radial bins.
    r0, r1 = r_centres[i - 1], r_centres[i]
    p0, p1 = prof[i - 1], prof[i]
    r_half = r0 + (half_level - p0) * (r1 - r0) / (p1 - p0)
    return float(2.0 * r_half)


def psf_metrics(
    psf_fine: np.ndarray,
    fine_scale_mas: float,
    spaxel_scale_mas: float,
) -> PSFMetrics:
    """Compute the full :class:`PSFMetrics` summary for a PSF image."""
    fwhm = psf_fwhm_mas(psf_fine, fine_scale_mas)
    ee1, ee3x3, _ = encircled_energy_per_spaxel(psf_fine, fine_scale_mas, spaxel_scale_mas)

    # Flux-weighted centroid offset from the grid centre.
    n = psf_fine.shape[0]
    half = (n - 1) / 2.0
    axis = (np.arange(n) - half) * fine_scale_mas
    xx, yy = np.meshgrid(axis, axis)
    tot = float(np.sum(psf_fine))
    cx = float(np.sum(xx * psf_fine) / tot) if tot > 0 else 0.0
    cy = float(np.sum(yy * psf_fine) / tot) if tot > 0 else 0.0
    centroid_shift = float(np.hypot(cx, cy))

    # Nyquist: need at least two spaxels across the FWHM.
    undersampled = bool(np.isfinite(fwhm) and (fwhm < 2.0 * spaxel_scale_mas))
    return PSFMetrics(
        fwhm_mas=fwhm,
        centroid_shift_mas=centroid_shift,
        ee1_spaxel=ee1,
        ee3x3_spaxel=ee3x3,
        undersampled=undersampled,
    )
