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
    flat-field error      pixel-to-pixel sensitivity variations we cannot
                          perfectly calibrate (multiplicative).
    bad pixels            dead/hot pixels that carry no usable signal.
    full-well saturation  pixels clip once they hold too many electrons.

The model returns both the **noisy electron frame** and a **variance frame** --
the latter is what an optimal (inverse-variance-weighted) extraction needs
downstream, following the logic of Horne (1986) optimal extraction.

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
        Saturation level in electrons; pixels clip above this.
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


@dataclass
class DetectorReadout:
    """Result of exposing a detector frame.

    Attributes
    ----------
    image_e
        Noisy detector frame in electrons (bias included, saturation applied,
        bad pixels zeroed).
    variance_e2
        Per-pixel variance estimate in electrons^2 (shot + dark + read noise).
        This is the *propagated* variance an extraction would use, not the
        realised noise.
    bad_pixel_mask
        Boolean array, True where a pixel is bad (and should be excluded).
    """

    image_e: np.ndarray
    variance_e2: np.ndarray
    bad_pixel_mask: np.ndarray

    def to_adu(self, gain_e_per_adu: float) -> np.ndarray:
        """Convert the electron frame to ADU using the supplied gain."""
        return self.image_e / gain_e_per_adu


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
        3. Poisson-realise signal+dark  (shot noise)
        4. add Gaussian read noise
        5. add bias offset
        6. clip at the full well        (saturation)
        7. zero out bad pixels

    The variance frame is computed from the *expected* counts
    (``signal + dark + read_noise^2``) -- the quantity an inverse-variance
    weighting scheme needs, independent of the particular noise realisation.
    It deliberately captures only Poisson + dark + read noise: the flat-field
    response and full-well saturation are applied to the data but treated as
    calibratable systematics and are **not** folded into the variance.

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
        Noisy frame, variance frame, and bad-pixel mask.
    """
    rng = _as_rng(rng)

    # 1. mean counts = signal + dark.
    mean_e = apply_dark_current(signal_e.astype(float), cfg, exposure_s)

    # Variance estimate from the expected counts (before realisation):
    #   shot noise of (signal+dark)  +  read noise^2.
    variance_e2 = mean_e + cfg.read_noise_e**2

    # 2. flat-field response.
    mean_e, _flat = apply_flat_field_error(mean_e, cfg, rng)

    # 3. Poisson shot noise on the (flat-fielded) mean.
    noisy = apply_poisson_noise(mean_e, rng)

    # 4. read noise.
    noisy = apply_read_noise(noisy, cfg, rng)

    # 5. bias offset.
    noisy = noisy + cfg.bias_e

    # 6. full-well saturation.
    noisy = np.clip(noisy, None, cfg.full_well_e)

    # 7. bad pixels.
    noisy, bad_mask = apply_bad_pixels(noisy, cfg, rng)
    variance_e2 = variance_e2.copy()
    # Give bad pixels an effectively infinite variance so any weighting ignores them.
    variance_e2[bad_mask] = np.inf

    return DetectorReadout(image_e=noisy, variance_e2=variance_e2, bad_pixel_mask=bad_mask)
