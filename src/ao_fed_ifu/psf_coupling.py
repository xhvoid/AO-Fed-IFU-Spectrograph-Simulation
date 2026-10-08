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

All PSFs are returned as 2-D arrays giving the energy integrated in square
pixels whose scale is given in milli-arcseconds (mas), normalised to unit total
flux **over the whole plane**: the Gaussian components use exact error-function
pixel integrals, while the Airy proxy uses fine-grid quadrature and its analytic
whole-plane integral.  A finite grid therefore sums to the fraction of the PSF
energy it actually encloses, never to 1 by fiat.  This
matters when a PSF is used as a convolution kernel: renormalising a truncated
kernel silently folds the seeing halo back into the core and inflates every
downstream throughput and S/N (a 9-spaxel kernel at 10 mas holds only ~51 % of
a 600 mas-seeing, Strehl 0.5 PSF).  Use :func:`halo_containing_n_pix` for a
grid that captures (nearly) all of the energy when a unit sum is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf, j1  # erf integrates Gaussian pixels; J1 makes Airy

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
        Encircled energy in the single central spaxel, as a fraction of the
        PSF's infinite-plane flux (0..1).
    ee3x3_spaxel
        Encircled energy in the central 3x3 block of spaxels, with the same
        infinite-plane normalisation (0..1).
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


def _integrated_gaussian_psf(
    sigma_mas: float,
    pixel_scale_mas: float,
    n_pix: int,
) -> np.ndarray:
    """Exact Gaussian probability in every square image pixel.

    A PSF array is used as a redistribution kernel, so each entry must be the
    integral over the corresponding pixel, not the surface brightness sampled
    at its centre.  A circular Gaussian is separable: the 1-D probability in a
    pixel is the difference of two error functions and the 2-D probability is
    their outer product.  Infinite-plane normalisation is therefore analytic;
    a finite returned grid sums to exactly the Gaussian energy inside its
    square footprint.  This remains correct when the Gaussian is narrower than
    a pixel, where centre sampling gives especially biased coupling fractions.
    """
    if not np.isfinite(sigma_mas) or sigma_mas <= 0.0:
        raise ValueError("sigma_mas must be finite and positive")
    if not np.isfinite(pixel_scale_mas) or pixel_scale_mas <= 0.0:
        raise ValueError("pixel_scale_mas must be finite and positive")

    xx, _, _ = _coordinate_grid(pixel_scale_mas, n_pix)
    centres = xx[0]
    half_pixel = pixel_scale_mas / 2.0
    scale = np.sqrt(2.0) * sigma_mas
    one_d = 0.5 * (
        erf((centres + half_pixel) / scale)
        - erf((centres - half_pixel) / scale)
    )
    return np.outer(one_d, one_d)


def halo_containing_n_pix(
    seeing_fwhm_mas: float,
    fine_scale_mas: float,
    extent_fwhms: float = 1.9,
    min_n_pix: int = 161,
) -> int:
    """Odd grid size whose extent contains (almost all of) the seeing halo.

    Gaussian builders retain their infinite-plane normalisation even on a
    finite grid.  A grid extent of ``extent_fwhms`` seeing-FWHMs (default 1.9,
    i.e. half-extent 0.95 FWHM ~ 2.24 sigma) captures ~95% of a Gaussian halo.
    This is useful when the full PSF image or its centroid is inspected; EE in
    a smaller requested aperture is still an absolute fraction and is never
    renormalised by the captured grid flux.
    """
    n = int(np.ceil(extent_fwhms * float(seeing_fwhm_mas) / float(fine_scale_mas)))
    n = max(n, int(min_n_pix))
    return n + 1 if n % 2 == 0 else n


# Sub-pixel sampling of the Airy pattern per lambda/D when integrating pixels.
_AIRY_SUBSAMPLES_PER_LAMBDA_OVER_D = 8.0


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
    first-order Bessel function.  The pattern is normalised by its analytic
    integral ``4 lambda^2 / (pi D^2)`` (in steradians, converted to the pixel
    scale), i.e. to unit flux over the whole plane; the returned grid sums to
    the enclosed fraction (~1 - 2/(pi x_max) for a grid of half-extent
    ``x_max = pi D theta_max / lambda``).  Each value is the pixel-AREA integral
    (midpoint rule on sub-pixels no larger than ``lambda/D / 8``), not a centre
    sample: centre-sampling a core narrower than a pixel would overstate the
    pixel's energy (a 10 mas grid at 1 um on 39 m summed to ~3).

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
    xx, yy, _ = _coordinate_grid(grid_mas, n_pix)
    lam_over_d_mas = (lambda_um * 1e-6) / telescope_diameter_m * _RAD_TO_MAS
    sub = max(1, int(np.ceil(_AIRY_SUBSAMPLES_PER_LAMBDA_OVER_D * grid_mas / lam_over_d_mas)))
    sub += 1 - sub % 2                      # odd: one sub-sample on each pixel centre
    offsets = (np.arange(sub) - (sub - 1) / 2.0) * (grid_mas / sub)
    psf = np.zeros_like(xx)
    for dy in offsets:
        for dx in offsets:
            theta_rad = np.hypot(xx + dx, yy + dy) / _RAD_TO_MAS
            x = np.pi * telescope_diameter_m * theta_rad / (lambda_um * 1e-6)
            # Handle the removable singularity at x = 0 where 2 J1(x)/x -> 1.
            with np.errstate(divide="ignore", invalid="ignore"):
                amp = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
            psf += amp**2
    psf /= sub * sub                        # pixel-mean intensity
    # int [2 J1(x)/x]^2 dOmega over the plane = 4 lambda^2 / (pi D^2), in sr;
    # one pixel is (grid_mas / rad_to_mas)^2 sr.
    pixel_sr = (grid_mas / _RAD_TO_MAS) ** 2
    total = 4.0 * (lambda_um * 1e-6) ** 2 / (np.pi * telescope_diameter_m**2) / pixel_sr
    return psf / total


def seeing_gaussian_psf(
    fwhm_mas: float,
    grid_mas: float,
    n_pix: int = 129,
) -> np.ndarray:
    """Seeing-limited Gaussian PSF with the given FWHM.

    Atmospheric seeing (without AO) blurs point sources into an approximately
    Gaussian core.  ``sigma = FWHM / (2 sqrt(2 ln 2))``.  Every returned value
    is the exact integral over that image pixel, normalised to unit flux over
    the whole plane, so a grid that truncates the profile sums to its enclosed
    fraction.
    """
    sigma = fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return _integrated_gaussian_psf(sigma, grid_mas, n_pix)


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

        PSF = S * core + (1 - S) * halo

    with each component integrated exactly over every square pixel and
    normalised to unit flux over the whole plane, so the Strehl really is the
    core's energy fraction and a finite grid sums to the fraction of the total
    PSF energy it encloses --
    it is **not** renormalised to 1 on the requested grid.

    The Strehl ratio ``S`` in [0, 1] is the only AO "quality" knob: S -> 1 gives
    a near-diffraction-limited PSF, S -> 0 gives the seeing PSF.

    This is a ``synthetic_proxy`` -- it captures the energy-concentration trend
    of AO correction without modelling residual wavefront error in detail.
    """
    strehl = float(np.clip(strehl, 0.0, 1.0))
    core_fwhm = diffraction_fwhm_mas(lambda_um, telescope_diameter_m)

    sigma_core = core_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_halo = seeing_fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    core = _integrated_gaussian_psf(sigma_core, grid_mas, n_pix)
    halo = _integrated_gaussian_psf(sigma_halo, grid_mas, n_pix)

    return strehl * core + (1.0 - strehl) * halo


def convolve_cube_with_psf(
    cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    psf_builder,
    spaxel_scale_mas: float,
    kernel_n_pix: int | None = None,
    conserve_flux: bool = False,
) -> np.ndarray:
    """Convolve every wavelength plane of a cube with a (possibly wavelength-
    dependent) PSF sampled at the spaxel scale.

    Kernel size and honesty
    -----------------------
    The kernel is the PSF builder's output on the spaxel grid.  Builders in
    this module normalise to unit flux over the whole plane, so a kernel sums
    to the energy it encloses and light scattered beyond it is *lost* -- which
    is only physical if it would have left the field anyway.  The default
    (``kernel_n_pix=None``) therefore sizes the kernel to reach every field
    pixel from every other, ``2 max(ny, nx) - 1``; with a zero boundary that is
    exact, whatever the halo size.  A smaller explicit kernel discards in-field
    light and is only appropriate for a PSF known to be compact on that scale.

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
        ``None`` (default) uses the field-covering size ``2 max(ny, nx) - 1``.
    conserve_flux
        If True, edge-normalise so the field flux is conserved (see above).

    Returns
    -------
    convolved_cube
        Same shape as ``cube``; each plane convolved with its PSF.
    """
    from scipy.signal import fftconvolve

    ny, nx, nlam = cube.shape
    if kernel_n_pix is None:
        kernel_n_pix = 2 * max(ny, nx) - 1
    kernel_n_pix = int(kernel_n_pix)
    if kernel_n_pix < 1:
        raise ValueError(f"kernel_n_pix must be positive; got {kernel_n_pix}")
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
    A separable area-overlap operator apportions any fine pixel cut by a spaxel
    boundary, rather than assigning its entire flux according to its centre.
    This is exactly conservative for the discrete pixel-integrated image and
    converges to the continuous-area integral as the fine scale is reduced.

    Returns
    -------
    spaxel_psf
        Array of shape ``(n_spaxel, n_spaxel)`` (n_spaxel forced odd) giving the
        fraction of total PSF flux in each spaxel, centred on the PSF peak.
    """
    if n_spaxel % 2 == 0:
        n_spaxel += 1
    image = np.asarray(psf_fine, dtype=float)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("psf_fine must be a square two-dimensional array")
    if not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("psf_fine must be finite and nonnegative")
    if not np.isfinite(fine_scale_mas) or fine_scale_mas <= 0.0:
        raise ValueError("fine_scale_mas must be finite and positive")
    if not np.isfinite(spaxel_scale_mas) or spaxel_scale_mas <= 0.0:
        raise ValueError("spaxel_scale_mas must be finite and positive")

    n_fine = image.shape[0]
    half = (n_fine - 1) / 2.0
    axis = (np.arange(n_fine) - half) * fine_scale_mas
    half_extent = n_spaxel * spaxel_scale_mas / 2.0
    edges = np.linspace(-half_extent, half_extent, n_spaxel + 1)
    fine_low = axis - fine_scale_mas / 2.0
    fine_high = axis + fine_scale_mas / 2.0
    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:, None], fine_high[None, :])
        - np.maximum(edges[:-1, None], fine_low[None, :]),
    )
    weights = overlap / fine_scale_mas
    return weights @ image @ weights.T


def encircled_energy_per_spaxel(
    psf_fine: np.ndarray,
    fine_scale_mas: float,
    spaxel_scale_mas: float,
) -> tuple[float, float, np.ndarray]:
    """Return ``(EE1, EE3x3, spaxel_psf)`` encircled-energy metrics.

    * ``EE1``   -- fraction of the **total** PSF flux in the single central spaxel,
    * ``EE3x3`` -- fraction of the total in the central 3x3 block of spaxels.

    Input PSF pixels follow this module's convention: each already contains a
    fraction of the unit-normalised, infinite-plane PSF.  EE values therefore
    sum the relevant pixels directly; they are *not* divided by the finite
    fine-grid sum or by the 7x7-spaxel sum.  A truncated input cannot silently
    turn field loss into coupling efficiency.  The fine grid only needs to
    cover the requested 7x7 spaxel footprint, although a halo-containing grid
    is useful for other diagnostics.

    These are the headline numbers for AO-fed IFU coupling: a good AO correction
    drives EE1 and EE3x3 up by concentrating light into few spaxels.
    """
    spaxel_psf = spaxelize_psf(psf_fine, fine_scale_mas, spaxel_scale_mas, n_spaxel=7)
    if float(np.sum(psf_fine)) <= 0:
        return 0.0, 0.0, spaxel_psf
    c = spaxel_psf.shape[0] // 2  # central index
    ee1 = float(spaxel_psf[c, c])
    ee3x3 = float(np.sum(spaxel_psf[c - 1 : c + 2, c - 1 : c + 2]))
    return ee1, ee3x3, spaxel_psf


def _radial_profile(psf_fine: np.ndarray, fine_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged profile: ``(radius_mas, mean energy per mas^2)``."""
    n = psf_fine.shape[0]
    half = (n - 1) / 2.0
    axis = (np.arange(n) - half) * fine_scale_mas
    xx, yy = np.meshgrid(axis, axis)
    rr = np.hypot(xx, yy).ravel()
    vals = np.asarray(psf_fine, dtype=float).ravel() / fine_scale_mas**2
    nbin = max(n // 2, 8)
    r_bins = np.linspace(0, rr.max(), nbin)
    index = np.digitize(rr, r_bins) - 1
    keep = (index >= 0) & (index < nbin - 1)
    counts = np.bincount(index[keep], minlength=nbin - 1)
    sums = np.bincount(index[keep], weights=vals[keep], minlength=nbin - 1)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    good = counts > 0
    return r_centres[good], sums[good] / counts[good]


def _half_max_crossing(r_centres: np.ndarray, profile: np.ndarray, half_level: float,
                       r_min: float = 0.0) -> float:
    """First radius >= ``r_min`` where ``profile`` falls to ``half_level`` (NaN if none)."""
    below = np.where((profile <= half_level) & (r_centres >= r_min))[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(r_centres[0])
    r0, r1 = r_centres[i - 1], r_centres[i]
    p0, p1 = profile[i - 1], profile[i]
    return float(r0 + (half_level - p0) * (r1 - r0) / (p1 - p0))


def psf_fwhm_two_grid_mas(
    psf_core: np.ndarray, core_scale_mas: float, psf_wide: np.ndarray, wide_scale_mas: float,
) -> float:
    """FWHM with the peak from a finely sampled core grid and the half-maximum
    crossing from whichever grid contains it.

    For a halo-dominated PSF the half-maximum lies beyond a small core grid,
    while a coarse wide grid underestimates a sharp peak; combining them (in
    energy per mas^2) keeps the reported width independent of the wide grid's
    sampling.
    """
    r_core, p_core = _radial_profile(psf_core, core_scale_mas)
    if p_core.size < 2:
        return float("nan")
    half_level = p_core[0] / 2.0
    core_reach = 0.75 * (psf_core.shape[0] - 1) / 2.0 * core_scale_mas
    r_half = _half_max_crossing(r_core, p_core, half_level)
    if np.isfinite(r_half) and r_half < core_reach:
        return 2.0 * r_half
    r_wide, p_wide = _radial_profile(psf_wide, wide_scale_mas)
    return 2.0 * _half_max_crossing(r_wide, p_wide, half_level, r_min=core_reach)


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
