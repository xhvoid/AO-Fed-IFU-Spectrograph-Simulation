"""Detector noise model: turn a noiseless electron frame into a noisy readout.

A real detector adds several independent noise terms on top of the astronomical
signal.  We model the dominant ones with scalar, **engineering-estimate**
parameters chosen to resemble published HAWAII-2RG / 4RG infrared arrays -- not
the cosmetics of any specific device:

    photon (shot) noise   Poisson statistics of arriving photo-electrons,
                          variance = signal (the fundamental floor).
    dark current          thermally-generated electrons, ~ rate x exposure,
                          themselves Poisson.
    read noise            Gaussian noise added each time the array is read out.
    flat-field response   multiplicative pixel-to-pixel sensitivity variations;
                          retained for ideal-known calibration or explicit
                          residual-response experiments.
    bad pixels            dead/hot pixels that carry no usable signal.
    full-well saturation  pixels clip once they hold too many electrons.

The model returns the **raw noisy electron frame**, its **variance frame**, and
the realized response/masks. :func:`calibrate_readout` removes the known detector
offsets and optionally corrects the response before science extraction. The
dark-current shot noise remains after its expectation is subtracted.

References (anchors only -- see docs/references.md):
    * MOSFIRE H2RG performance, arXiv:1208.0314 -- read-noise / dark-current scale.
    * JWST/NIRSpec detector noise, arXiv:2208.12686 -- dark, read/total noise.
    * Rauscher H1RG/H2RG/H4RG noise generator, arXiv:1509.06264 -- noise-component
      vocabulary (we model only the scalar white-read/dark/Poisson terms).
    * Horne 1986, 1986PASP...98..609H -- optimal (variance-weighted) extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Read-noise margin used to infer saturation for legacy readouts that carry no
# saturation mask (see :func:`calibrate_readout`).
LEGACY_SATURATION_MARGIN_SIGMA = 3.0


@dataclass
class DetectorConfig:
    """Scalar detector parameters (H2RG/H4RG-*like* engineering proxies).

    Parameters
    ----------
    read_noise_e
        RMS read noise per pixel per readout, in electrons.
    dark_current_e_s_pix
        Dark current in electrons per second per pixel.
    bias_e
        Constant electronic offset added to every pixel, in electrons.
    full_well_e
        Charge capacity of a pixel in electrons: collected photo/dark charge
        clips here, before read noise and the electronic bias are added (so a
        large bias never saturates a pixel). ADC clipping is not modelled.
    gain_e_per_adu
        Conversion gain (electrons per ADU) for the optional ADU output.
    bad_pixel_fraction
        Fraction of pixels that are dead/hot and carry no usable signal.
    flat_field_sigma
        RMS of the multiplicative pixel-response non-uniformity (e.g. 0.01 = 1%).
    """

    read_noise_e: float = 10.0
    dark_current_e_s_pix: float = 0.01
    bias_e: float = 0.0
    full_well_e: float = 1.0e5
    gain_e_per_adu: float = 2.0
    bad_pixel_fraction: float = 0.01
    flat_field_sigma: float = 0.01
    # Non-destructive reads for the up-the-ramp / Fowler sampling proxy. n_reads=1
    # (default) leaves the read noise == read_noise_e (backward compatible); more
    # reads reduce the effective read noise (engineering_estimate scaling).
    n_reads: int = 1
    read_mode: str = "fowler"  # "fowler" | "up_the_ramp"


@dataclass
class DetectorReadout:
    """Result of exposing a detector frame.

    Attributes
    ----------
    image_e
        Noisy detector frame in electrons (charge clipped at the full well,
        then read noise and bias added; bad pixels zeroed).
    variance_e2
        Per-pixel variance estimate in electrons^2 (shot + dark + read noise).
        This is the *propagated* variance an extraction would use, not the
        realised noise. Bad, saturated, and zero-response pixels have infinite
        variance, because they do not provide an unbiased source estimate.
    bad_pixel_mask
        Boolean array, True where a pixel is bad (and should be excluded).
    flat_response
        Realized multiplicative response map. ``None`` is allowed for legacy
        manually constructed readouts; it does not imply a known unit response.
    saturation_mask
        Boolean array, True where the collected charge reached ``full_well_e``
        (before read noise and bias are added). Saturated pixels cannot be
        recovered by linear calibration.
    """

    image_e: np.ndarray
    variance_e2: np.ndarray
    bad_pixel_mask: np.ndarray
    flat_response: np.ndarray | None = None
    saturation_mask: np.ndarray | None = None

    def to_adu(self, gain_e_per_adu: float) -> np.ndarray:
        """Convert the electron frame to ADU using the supplied gain."""
        return self.image_e / gain_e_per_adu


@dataclass
class CalibratedReadout:
    """Detector-calibrated data, with the raw readout preserved separately.

    ``image_e`` is bias/dark-expectation subtracted, and response corrected when
    requested. ``variance_e2`` retains signal/dark shot noise and read noise,
    transformed by the same response correction. Unusable pixels have zero
    placeholder data, infinite variance, and ``invalid_pixel_mask=True``.
    Calibration parameters are assumed exactly known: no uncertainty from a
    measured bias, dark, or flat calibration frame is added.
    """

    image_e: np.ndarray
    variance_e2: np.ndarray
    invalid_pixel_mask: np.ndarray


def calibrate_readout(
    readout: DetectorReadout,
    cfg: DetectorConfig,
    exposure_s: float,
    *,
    correct_flat: bool = True,
) -> CalibratedReadout:
    """Remove known detector offsets and propagate the calibration variance.

    The current detector proxy applies response ``F`` to ``signal + dark``
    before realizing Poisson noise, then adds read noise and bias. Its inverse
    is therefore ``(raw - bias) / F - dark`` and ``variance_raw / F**2``.
    This preserves the current forward-model ordering; it does not claim dark
    charge is physically generated by the optical sensitivity map.

    ``correct_flat=False`` is an explicit residual-response experiment: remove
    ``bias + F * dark`` but retain the response-modulated astronomical signal
    and raw variance. This mode is not a fully calibrated science frame.

    Bad, saturated, non-finite, and non-positive-response pixels are excluded
    in both modes. The input readout is never modified. A legacy readout with
    no response map is supported only when ``cfg.flat_field_sigma == 0``;
    otherwise its unknown response realization makes calibration undefined.
    """
    raw = np.asarray(readout.image_e, dtype=float)
    variance = np.asarray(readout.variance_e2, dtype=float)
    bad = np.asarray(readout.bad_pixel_mask, dtype=bool)
    if variance.shape != raw.shape or bad.shape != raw.shape:
        raise ValueError("Readout image, variance, and bad-pixel mask must have matching shapes.")
    if not np.isfinite(exposure_s) or exposure_s < 0:
        raise ValueError("exposure_s must be finite and non-negative.")
    if not np.isfinite(cfg.dark_current_e_s_pix) or cfg.dark_current_e_s_pix < 0:
        raise ValueError("dark_current_e_s_pix must be finite and non-negative.")
    if not np.isfinite(cfg.bias_e):
        raise ValueError("bias_e must be finite.")
    if readout.flat_response is None:
        if cfg.flat_field_sigma != 0:
            raise ValueError("Calibration requires the realized flat_response when flat_field_sigma is nonzero.")
        response = np.ones_like(raw)
    else:
        response = np.asarray(readout.flat_response, dtype=float)
        if response.shape != raw.shape:
            raise ValueError("flat_response must match the readout image shape.")
    if readout.saturation_mask is None:
        # Legacy readouts do not retain the clipping mask. Infer it with the
        # charge-well semantics of ``expose``: the bias-subtracted value of a
        # clipped pixel is full_well + read noise, so anything within
        # LEGACY_SATURATION_MARGIN_SIGMA read-noise sigma of the well is treated
        # as unusable (a bias, however large, never saturates a pixel).
        from .detector_systematics import effective_read_noise

        rn_eff = effective_read_noise(cfg.read_noise_e, cfg.n_reads, cfg.read_mode)
        # A relative floor absorbs the rounding of (full_well + bias) - bias
        # when there is no read noise to provide a margin.
        margin = max(LEGACY_SATURATION_MARGIN_SIGMA * rn_eff, 1e-9 * cfg.full_well_e)
        saturated = (raw - cfg.bias_e) >= cfg.full_well_e - margin
    else:
        saturated = np.asarray(readout.saturation_mask, dtype=bool)
        if saturated.shape != raw.shape:
            raise ValueError("saturation_mask must match the readout image shape.")
    invalid = (
        bad | saturated | ~np.isfinite(raw) | ~np.isfinite(variance)
        | (variance < 0) | ~np.isfinite(response) | (response <= 0)
    )
    valid = ~invalid
    image = np.zeros_like(raw)
    calibrated_variance = np.full_like(raw, np.inf)
    dark_e = cfg.dark_current_e_s_pix * exposure_s
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if correct_flat:
            image[valid] = (raw[valid] - cfg.bias_e) / response[valid] - dark_e
            calibrated_variance[valid] = variance[valid] / response[valid]**2
        else:
            image[valid] = raw[valid] - cfg.bias_e - response[valid] * dark_e
            calibrated_variance[valid] = variance[valid]
    # A finite but vanishing response can overflow the inverse even when the
    # input arrays are finite; do not advertise such a pixel as calibrated.
    invalid |= ~np.isfinite(image) | ~np.isfinite(calibrated_variance)
    image[invalid] = 0.0
    calibrated_variance[invalid] = np.inf
    return CalibratedReadout(image, calibrated_variance, invalid)


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce ``rng`` (Generator, int seed, or None) into a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def apply_dark_current(frame_e: np.ndarray, cfg: DetectorConfig, exposure_s: float) -> np.ndarray:
    """Add the *mean* dark-current electrons (rate x exposure) to every pixel.

    The shot noise of these dark electrons is realised later in the Poisson step,
    so here we only add the deterministic expectation value.
    """
    return frame_e + cfg.dark_current_e_s_pix * exposure_s


def apply_flat_field_error(
    frame_e: np.ndarray, cfg: DetectorConfig, rng: np.random.Generator | int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply by a per-pixel response map ``~ 1 + N(0, flat_field_sigma)``.

    Returns ``(modified_frame, flat_map)``.  The flat map is returned so callers
    can, if they wish, model imperfect flat-field *correction* separately.
    """
    rng = _as_rng(rng)
    if cfg.flat_field_sigma <= 0:
        return frame_e, np.ones_like(frame_e)
    flat_map = 1.0 + rng.normal(0.0, cfg.flat_field_sigma, size=frame_e.shape)
    flat_map = np.clip(flat_map, 0.0, None)  # response cannot be negative
    return frame_e * flat_map, flat_map


def apply_poisson_noise(
    mean_e: np.ndarray, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Draw Poisson realisations of the per-pixel mean electron counts.

    Poisson shot noise is the fundamental limit: variance equals the mean.
    Negative means (numerical only) are clipped to zero before drawing.
    """
    rng = _as_rng(rng)
    return rng.poisson(np.clip(mean_e, 0.0, None)).astype(float)


def apply_read_noise(
    frame_e: np.ndarray, cfg: DetectorConfig, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Add zero-mean Gaussian read noise with RMS ``read_noise_e``."""
    rng = _as_rng(rng)
    if cfg.read_noise_e <= 0:
        return frame_e
    return frame_e + rng.normal(0.0, cfg.read_noise_e, size=frame_e.shape)


def apply_bad_pixels(
    frame_e: np.ndarray, cfg: DetectorConfig, rng: np.random.Generator | int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out a random fraction of pixels and return ``(frame, mask)``.

    Bad (dead/hot) pixels carry no usable signal; the boolean mask lets
    reconstruction and flux-conservation accounting exclude them.
    """
    rng = _as_rng(rng)
    mask = np.zeros(frame_e.shape, dtype=bool)
    if cfg.bad_pixel_fraction > 0:
        mask = rng.random(frame_e.shape) < cfg.bad_pixel_fraction
        frame_e = frame_e.copy()
        frame_e[mask] = 0.0
    return frame_e, mask


def expose(
    signal_e: np.ndarray,
    cfg: DetectorConfig,
    exposure_s: float,
    rng: np.random.Generator | int | None = None,
) -> DetectorReadout:
    """Run the full detection chain on a noiseless signal frame.

    Order of operations (each models a distinct physical effect):

        1. add mean dark current        (rate x exposure)
        2. apply flat-field response    (multiplicative pixel non-uniformity)
        3. Poisson-realise signal+dark  (shot noise: the collected charge)
        4. clip the charge at the full well (saturation is a well limit)
        5. add Gaussian read noise
        6. add bias offset              (electronic; cannot saturate the well)
        7. zero out bad pixels

    Conditional on the realized response ``F``, the variance frame is
    ``F * (signal + dark) + read_noise^2`` -- the expected Poisson counts plus
    read noise, independent of the particular noise realization. Unusable bad,
    saturated, or zero-response pixels have infinite variance. Response
    calibration uncertainty is not included; :func:`calibrate_readout` treats
    the retained response as exactly known and transforms variance consistently.

    Parameters
    ----------
    signal_e
        Noiseless source electrons per pixel (from the IFU forward projection).
    cfg
        Detector parameters.
    exposure_s
        Exposure time in seconds (sets the dark-current contribution).
    rng
        Seed or Generator for reproducible noise.

    Returns
    -------
    DetectorReadout
        Raw noisy frame, variance frame, response, and bad/saturation masks.
    """
    rng = _as_rng(rng)

    # Effective read noise after the up-the-ramp / Fowler sampling proxy.
    # n_reads=1 (default) -> rn_eff == cfg.read_noise_e (backward compatible).
    from .detector_systematics import effective_read_noise

    rn_eff = effective_read_noise(cfg.read_noise_e, cfg.n_reads, cfg.read_mode)

    # 1. mean counts = signal + dark.
    mean_e = apply_dark_current(signal_e.astype(float), cfg, exposure_s)

    # 2. flat-field response.
    mean_e, flat_response = apply_flat_field_error(mean_e, cfg, rng)

    # Conditional Poisson variance uses the SAME response-scaled expectation
    # as the draw, including its non-negative clipping convention.
    variance_e2 = np.clip(mean_e, 0.0, None) + rn_eff**2

    # 3. Poisson shot noise on the (flat-fielded) mean: the collected charge.
    noisy = apply_poisson_noise(mean_e, rng)

    # 4. full-well saturation of the collected charge (not of bias/read noise).
    saturation_mask = noisy >= cfg.full_well_e
    noisy = np.clip(noisy, None, cfg.full_well_e)

    # 5. read noise (effective, after the sampling proxy). Inlined so it can use
    #    rn_eff; identical to apply_read_noise(cfg) when n_reads == 1.
    if rn_eff > 0:
        noisy = noisy + rng.normal(0.0, rn_eff, size=noisy.shape)

    # 6. bias offset.
    noisy = noisy + cfg.bias_e

    # 7. bad pixels.
    noisy, bad_mask = apply_bad_pixels(noisy, cfg, rng)
    variance_e2 = variance_e2.copy()
    # A clipped/zero-response pixel is not an unbiased linear source estimate.
    # Retain separate masks, but ensure raw-data weighting excludes it as well.
    variance_e2[bad_mask | saturation_mask | (flat_response <= 0)] = np.inf

    return DetectorReadout(
        image_e=noisy, variance_e2=variance_e2, bad_pixel_mask=bad_mask,
        flat_response=flat_response, saturation_mask=saturation_mask,
    )
