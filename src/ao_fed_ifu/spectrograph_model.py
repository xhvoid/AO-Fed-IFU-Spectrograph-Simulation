"""A simple single-order spectrograph: wavelength solution, LSF and sampling.

A spectrograph disperses light so that wavelength maps to position along the
detector.  This module models that map in the simplest useful way:

    * a **linear wavelength solution** lambda(x) along the dispersion axis,
    * a Gaussian **line-spread function** (LSF) that blurs the spectrum to the
      instrument's spectral resolution,
    * the **resolving power** R = lambda / delta_lambda and the associated
      sampling (pixels per resolution element).

Key physical quantities
    R                         resolving power (dimensionless); larger R resolves
                              finer spectral detail.
    delta_lambda = lambda / R width of one resolution element (a spectral "FWHM"
                              the instrument can deliver).
    pixels_per_resel          how many detector pixels sample one resolution
                              element; >= 2 is required to avoid undersampling.

Everything here is a ``synthetic_proxy`` / ``literature_derived`` construction:
mode presets resemble published R and wavelength ranges, but no real grating or
optical design is reproduced.

References (anchors only -- see docs/references.md):
    * HARMONI overview, arXiv:2103.11215; MUSE, arXiv:astro-ph/0606329 --
      R and wavelength-range presets.
    * BlueMUSE pipeline design, arXiv:2209.06022 -- LSF / wavelength-calibration
      importance.
    * PypeIt, arXiv:2005.06505 and PyWiFeS, arXiv:1311.2666 -- trace / wavelength /
      extraction vocabulary; ESO ETC 2.0, arXiv:2407.02972 -- ETC thinking.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def wavelength_bin_edges(wavelength_centres_um: np.ndarray) -> np.ndarray:
    """Return bin edges whose midpoints are ``wavelength_centres_um``.

    Interior edges lie halfway between adjacent centres; the two outer bins use
    the adjacent half-spacing.  This convention matches the detector grid used
    by :class:`SpectrographConfig` while also supporting non-uniform reference
    grids in the public-data loaders.
    """
    centres = np.asarray(wavelength_centres_um, dtype=float)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("wavelength centres must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(centres)) or np.any(np.diff(centres) <= 0):
        raise ValueError("wavelength centres must be finite and strictly increasing")
    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def integrate_piecewise_linear_bins(
    wavelength_um: np.ndarray,
    spectral_density: np.ndarray,
    bin_edges_um: np.ndarray,
) -> np.ndarray:
    """Integrate a sampled spectral density exactly under linear interpolation.

    The tabulated density is treated as zero outside its native wavelength
    support.  In particular, a detector bin extending half a pixel beyond a
    cache boundary loses that unsupported half-bin instead of inheriting a
    silently repeated endpoint value from ``numpy.interp``.
    """
    wavelength = np.asarray(wavelength_um, dtype=float)
    density = np.asarray(spectral_density, dtype=float)
    edges = np.asarray(bin_edges_um, dtype=float)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("native wavelength grid must be one-dimensional with length >= 2")
    if density.shape != wavelength.shape:
        raise ValueError("spectral density must match the native wavelength grid")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin edges must be a one-dimensional array of length >= 2")
    if (
        not np.all(np.isfinite(wavelength))
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(wavelength) <= 0)
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("wavelengths and bin edges must be finite and strictly increasing")

    segment_width = np.diff(wavelength)
    segment_slope = np.diff(density) / segment_width
    cumulative = np.concatenate((
        np.array([0.0]),
        np.cumsum(0.5 * (density[:-1] + density[1:]) * segment_width),
    ))

    # Analytic antiderivative of the piecewise-linear interpolant.  Values
    # outside the native interval are exactly zero-density, with no artificial
    # ramp between a requested edge and the first/last native sample.
    clipped = np.clip(edges, wavelength[0], wavelength[-1])
    index = np.searchsorted(wavelength, clipped, side="right") - 1
    index = np.clip(index, 0, wavelength.size - 2)
    offset = clipped - wavelength[index]
    antiderivative = (
        cumulative[index]
        + density[index] * offset
        + 0.5 * segment_slope[index] * offset**2
    )
    antiderivative = np.where(edges <= wavelength[0], 0.0, antiderivative)
    antiderivative = np.where(edges >= wavelength[-1], cumulative[-1], antiderivative)
    return np.diff(antiderivative)


def _project_onto_uniform_hats(
    wavelength_um: np.ndarray, spectral_density: np.ndarray, nodes_um: np.ndarray
) -> np.ndarray:
    """Project a piecewise-linear density onto hat functions on uniform ``nodes_um``.

    Every native piece between consecutive breakpoints (native knots and nodes)
    deposits ``integral f(y) hat_k(y) dy`` onto its two bracketing nodes, and
    the deposits are divided by the node spacing.  The piecewise-linear
    interpolant of the result has exactly the native area and first moment
    (the native support must lie strictly inside the node range); for a density
    that is linear across a node's support it equals point sampling.
    """
    step = float(nodes_um[1] - nodes_um[0])
    inside = (nodes_um > wavelength_um[0]) & (nodes_um < wavelength_um[-1])
    knots = np.union1d(wavelength_um, nodes_um[inside])
    values = np.interp(knots, wavelength_um, spectral_density)
    x0, x1 = knots[:-1], knots[1:]
    f0, f1 = values[:-1], values[1:]
    j = np.clip(np.searchsorted(nodes_um, x0, side="right") - 1, 0, nodes_um.size - 2)
    a, b = x0 - nodes_um[j], x1 - nodes_um[j]          # local coordinates in [0, step]
    mass = 0.5 * (f0 + f1) * (b - a)
    moment = (b - a) * (f0 * (2.0 * a + b) + f1 * (a + 2.0 * b)) / 6.0   # integral (y - n_j) f
    upper = moment / step
    deposit = np.zeros(nodes_um.size)
    np.add.at(deposit, j, mass - upper)
    np.add.at(deposit, j + 1, upper)
    return deposit / step


def bin_average_spectral_density(
    wavelength_um: np.ndarray,
    spectral_density: np.ndarray,
    wavelength_centres_um: np.ndarray,
) -> np.ndarray:
    """Flux-conserving mean density in bins centred on the requested grid."""
    edges = wavelength_bin_edges(wavelength_centres_um)
    return integrate_piecewise_linear_bins(wavelength_um, spectral_density, edges) / np.diff(edges)


@dataclass
class SpectrographConfig:
    """Configuration of the single-order spectrograph.

    Parameters
    ----------
    lambda_min_um, lambda_max_um
        Wavelength range covered along the dispersion axis, in micrometres.
    resolving_power_R
        Target resolving power R = lambda / delta_lambda (mode preset).
    pixels_per_resolution_element
        Detector pixels per resolution element (the sampling).  Must be >= 2 for
        a Nyquist-sampled spectrum; presets typically use ~2.0-2.5.
    dispersion_axis
        Which detector axis carries wavelength; only ``"x"`` is implemented in
        this first milestone (dispersion along detector columns).
    lsf_fwhm_pix
        FWHM of the Gaussian LSF in pixels.  Defaults to
        ``pixels_per_resolution_element`` (the resolution element *is* the LSF
        FWHM by definition).
    detector_pixels
        Fixed number of detector columns.  ``None`` (default) derives it from R
        and the sampling; a perturbed wavelength solution sets it so the
        detector geometry stays that of the nominal instrument.
    """

    lambda_min_um: float = 1.95
    lambda_max_um: float = 2.45
    resolving_power_R: float = 3000.0
    pixels_per_resolution_element: float = 2.5
    dispersion_axis: str = "x"
    lsf_fwhm_pix: float | None = None
    detector_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.dispersion_axis != "x":
            raise NotImplementedError(
                "Only dispersion_axis='x' is implemented in this milestone."
            )
        if self.lsf_fwhm_pix is None:
            self.lsf_fwhm_pix = float(self.pixels_per_resolution_element)
        if self.detector_pixels is not None and (
            isinstance(self.detector_pixels, bool)
            or not isinstance(self.detector_pixels, (int, np.integer))
            or self.detector_pixels < 1
        ):
            raise ValueError("detector_pixels must be a positive integer or None")

    # -- Wavelength sampling -------------------------------------------------

    @property
    def lambda_mid_um(self) -> float:
        """Band-centre wavelength in micrometres."""
        return 0.5 * (self.lambda_min_um + self.lambda_max_um)

    def delta_lambda_resel_um(self, lambda_um: float | None = None) -> float:
        """Width of one resolution element, ``delta_lambda = lambda / R`` (um).

        Evaluated at ``lambda_um`` (defaults to the band centre).
        """
        lam = self.lambda_mid_um if lambda_um is None else lambda_um
        return lam / self.resolving_power_R

    @property
    def n_detector_pixels(self) -> int:
        """Number of detector pixels along the dispersion axis.

        Chosen so the band is sampled at ``pixels_per_resolution_element`` pixels
        per resolution element:

            n_resel = (lambda_max - lambda_min) / delta_lambda(band centre)
            n_pix   = ceil(n_resel * pixels_per_resolution_element)

        Tying the detector width to R and the sampling keeps the model
        self-consistent: the recovered spectrum is neither over- nor
        under-resolved relative to the stated R.  An explicit
        ``detector_pixels`` overrides this (fixed physical detector).
        """
        if self.detector_pixels is not None:
            return int(self.detector_pixels)
        n_resel = (self.lambda_max_um - self.lambda_min_um) / self.delta_lambda_resel_um()
        return int(np.ceil(n_resel * self.pixels_per_resolution_element))

    @property
    def dispersion_um_per_pix(self) -> float:
        """Linear dispersion: micrometres of wavelength per detector pixel."""
        return (self.lambda_max_um - self.lambda_min_um) / max(self.n_detector_pixels - 1, 1)

    def actual_pixels_per_resolution_element(self, lambda_um: float | None = None) -> float:
        """Realised sampling = delta_lambda_resel / dispersion-per-pixel."""
        return self.delta_lambda_resel_um(lambda_um) / self.dispersion_um_per_pix

    def is_well_sampled(self) -> bool:
        """True if the spectrum is sampled at >= 2 pixels per resolution element."""
        return self.actual_pixels_per_resolution_element() >= 2.0

    # -- Wavelength solution (linear, optionally perturbed) ------------------

    def wavelength_to_detector_x(self, lambda_um) -> np.ndarray:
        """Map wavelength(s) to fractional detector column index.

        Linear solution: ``x = (lambda - lambda_min) / dispersion``.
        Returns floats (callers round/interpolate as needed).
        """
        lambda_um = np.asarray(lambda_um, dtype=float)
        return (lambda_um - self.lambda_min_um) / self.dispersion_um_per_pix

    def detector_x_to_wavelength(self, x) -> np.ndarray:
        """Map detector column index(es) to wavelength in micrometres.

        Inverse of :meth:`wavelength_to_detector_x`:
        ``lambda = lambda_min + x * dispersion``.
        """
        x = np.asarray(x, dtype=float)
        return self.lambda_min_um + x * self.dispersion_um_per_pix

    def wavelength_grid_um(self) -> np.ndarray:
        """Wavelength at every detector column, shape ``(n_detector_pixels,)``."""
        return self.detector_x_to_wavelength(np.arange(self.n_detector_pixels))

    def with_wavelength_offset(self, delta_lambda_um: float) -> "SpectrographConfig":
        """Return a copy whose wavelength solution is shifted by a constant.

        This models a **wavelength-calibration error**: if reconstruction uses a
        solution offset by ``delta_lambda_um`` from the truth, every line appears
        Doppler-shifted by ``delta_v = c * delta_lambda / lambda``.  The
        validation suite uses exactly this to check the predicted velocity bias.

        The shift is applied by moving ``lambda_min`` (and ``lambda_max``) with
        the detector column count held fixed, so the dispersion is unchanged --
        a pure zero-point offset (re-deriving the column count from the shifted
        band centre would also change the detector geometry).
        """
        shifted = copy.deepcopy(self)
        shifted.detector_pixels = self.n_detector_pixels
        shifted.lambda_min_um = self.lambda_min_um + delta_lambda_um
        shifted.lambda_max_um = self.lambda_max_um + delta_lambda_um
        return shifted

    def with_dispersion_stretch(self, fractional_stretch: float) -> "SpectrographConfig":
        """Return a copy whose dispersion is stretched by ``(1 + fractional_stretch)``.

        Models a first-order (linear) wavelength-scale error, the next term
        beyond a pure offset.  Useful for calibration-sensitivity tests where the
        velocity bias varies across the band.
        """
        stretched = copy.deepcopy(self)
        stretched.detector_pixels = self.n_detector_pixels   # same detector, longer span
        span = (self.lambda_max_um - self.lambda_min_um) * (1.0 + fractional_stretch)
        stretched.lambda_max_um = self.lambda_min_um + span
        return stretched

    # -- Line-spread function ------------------------------------------------

    @property
    def lsf_sigma_pix(self) -> float:
        """Gaussian LSF sigma in pixels (from the configured FWHM)."""
        return float(self.lsf_fwhm_pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def apply_lsf(self, array: np.ndarray, axis: int = -1) -> np.ndarray:
        """Convolve ``array`` with the Gaussian LSF along ``axis``.

        Used to blur a noiseless spectrum / detector image to the instrument's
        spectral resolution before detection.  Interior flux is conserved
        exactly; light the LSF spreads past the first/last detector column is
        LOST (``mode="constant"``, zero padding) -- a detector edge has no pixel
        beyond it -- so an edge-channel impulse sums to the fraction of its
        LSF that lands on the array (< 1), never more.  (``mode="nearest"``
        used to replicate the edge column outward and blur those copies back
        in, so a unit impulse in the first channel summed to ~1.08: phantom
        flux for lines near the band edges.)
        """
        if self.lsf_sigma_pix <= 0:
            return np.asarray(array, dtype=float)
        return gaussian_filter1d(
            np.asarray(array, dtype=float), sigma=self.lsf_sigma_pix, axis=axis,
            mode="constant", cval=0.0,
        )

    def integrate_spectral_density(
        self,
        wavelength_um: np.ndarray,
        spectral_density: np.ndarray,
        *,
        apply_lsf: bool = True,
        minimum_samples_per_detector_pixel: int = 8,
    ) -> np.ndarray:
        """Apply the instrumental LSF on a fine grid, then integrate detector bins.

        Parameters
        ----------
        wavelength_um, spectral_density
            Native samples of a spectral density (for example SkyCalc radiance
            after multiplication by a native-grid filter curve).  Samples are
            linearly interpolated, and the density is zero outside the supplied
            wavelength support.
        apply_lsf
            If true, convolve exactly once before detector-bin integration.  A
            caller using this for sky must therefore bypass the later coarse-grid
            LSF when projecting that already-convolved sky onto the detector.
        minimum_samples_per_detector_pixel
            Minimum internal sampling used to represent the piecewise-linear
            input and Gaussian.  Fine native data retain their finer spacing.

        Returns
        -------
        numpy.ndarray
            Integrated quantity in each detector wavelength bin.  The units are
            the input density units times micrometres; do not multiply by another
            ``delta_lambda`` downstream.

        Notes
        -----
        The physical wavelength sigma is
        ``lsf_sigma_pix * dispersion_um_per_pix``.  This deliberately matches
        :meth:`apply_lsf`, i.e. the current constant-FWHM-in-detector-pixels
        spectrograph proxy rather than introducing a different variable-R model.
        Four-sigma zero padding captures the same truncated Gaussian kernel as
        SciPy's default used by :meth:`apply_lsf`.
        """
        wavelength = np.asarray(wavelength_um, dtype=float)
        density = np.asarray(spectral_density, dtype=float)
        if wavelength.ndim != 1 or wavelength.size < 2:
            raise ValueError("native wavelength grid must be one-dimensional with length >= 2")
        if density.shape != wavelength.shape:
            raise ValueError("spectral density must match the native wavelength grid")
        if (
            not np.all(np.isfinite(wavelength))
            or not np.all(np.isfinite(density))
            or np.any(np.diff(wavelength) <= 0)
        ):
            raise ValueError("native wavelength and density must be finite with increasing wavelength")
        if (
            isinstance(minimum_samples_per_detector_pixel, bool)
            or not isinstance(minimum_samples_per_detector_pixel, (int, np.integer))
            or minimum_samples_per_detector_pixel < 2
        ):
            raise ValueError("minimum_samples_per_detector_pixel must be an integer >= 2")

        detector_edges = wavelength_bin_edges(self.wavelength_grid_um())
        sigma_um = self.lsf_sigma_pix * self.dispersion_um_per_pix
        if not apply_lsf or sigma_um <= 0:
            return integrate_piecewise_linear_bins(
                wavelength, density, detector_edges
            )

        native_diffs = np.diff(wavelength)
        native_step = float(np.median(native_diffs))
        target_step = self.dispersion_um_per_pix / minimum_samples_per_detector_pixel
        uniform_native = np.allclose(
            native_diffs,
            native_step,
            rtol=1e-8,
            atol=32 * np.finfo(float).eps * max(abs(float(wavelength[-1])), 1.0),
        )
        if uniform_native:
            # Use an integer subdivision so every native knot is retained.  A
            # merely smaller, unrelated step shifts narrow lines relative to
            # the interpolation grid and can change their integrated flux.
            subdivisions = max(1, int(np.ceil(native_step / target_step)))
            fine_step = native_step / subdivisions
        else:
            # There is no single step that can retain every irregular knot, so
            # the native interpolant is PROJECTED onto the fine grid's hat
            # functions (below): a feature narrower than a fine cell keeps its
            # exact area and centroid instead of vanishing between point samples
            # or snapping to a node.  Direct no-LSF binning remains exact via the
            # helper above.
            fine_step = min(native_step, target_step) / 2.0
        padding_um = 4.0 * sigma_um

        # Anchor the internal grid on the first native sample.  When the input is
        # already uniform (as SkyCalc is), its samples therefore land on the fine
        # grid rather than acquiring a phase shift during resampling.
        lower_needed = min(float(detector_edges[0]), float(wavelength[0])) - padding_um
        upper_needed = max(float(detector_edges[-1]), float(wavelength[-1])) + padding_um
        n_left = int(np.ceil((wavelength[0] - lower_needed) / fine_step))
        fine_start = float(wavelength[0]) - n_left * fine_step
        n_interval = int(np.ceil((upper_needed - fine_start) / fine_step))
        fine_wavelength = fine_start + np.arange(n_interval + 1, dtype=float) * fine_step
        if uniform_native:
            # Every native knot is a fine node, so point sampling is exact inside
            # the support.  A nonzero density at a support ENDPOINT is a step to
            # zero, which point sampling would ramp over one extra fine cell
            # outside the support (creating flux); that endpoint part alone is
            # projected onto the hat functions, conserving its area and centroid.
            interior = density.copy()
            interior[[0, -1]] = 0.0
            fine_density = np.interp(
                fine_wavelength, wavelength, interior, left=0.0, right=0.0
            )
            endpoints = density - interior
            if np.any(endpoints != 0.0):
                fine_density = fine_density + _project_onto_uniform_hats(
                    wavelength, endpoints, fine_wavelength)
        else:
            fine_density = _project_onto_uniform_hats(wavelength, density, fine_wavelength)

        fine_density = gaussian_filter1d(
            fine_density,
            sigma=sigma_um / fine_step,
            mode="constant",
            cval=0.0,
        )
        return integrate_piecewise_linear_bins(
            fine_wavelength, fine_density, detector_edges
        )


# Instrument-inspired presets (literature_derived: plausible mode parameters,
# NOT reproductions of any real instrument's as-built configuration).
SPECTROGRAPH_PRESETS: dict[str, SpectrographConfig] = {
    # Near-IR, AO-fed, moderate resolution -- HARMONI-*inspired* K-band-ish mode.
    "nir_R3000": SpectrographConfig(
        lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=3000.0,
        pixels_per_resolution_element=2.5,
    ),
    # Visible, MUSE-*inspired* wide-field mode (single synthetic order here).
    "vis_R3000": SpectrographConfig(
        lambda_min_um=0.48, lambda_max_um=0.93, resolving_power_R=3000.0,
        pixels_per_resolution_element=2.5,
    ),
    # Higher-resolution NIR mode for line-kinematics demonstrations.
    "nir_R7000": SpectrographConfig(
        lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=7000.0,
        pixels_per_resolution_element=2.5,
    ),
}
