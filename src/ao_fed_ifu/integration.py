"""End-to-end pipeline orchestration and the three run modes.

This module wires the per-stage modules into one runnable pipeline:

    source_cube -> psf_coupling -> ifu_mapping -> spectrograph (LSF)
    -> detector_model -> cube_reconstruction -> (sky subtraction) -> diagnostics

When a sky background is injected (public-conditioned runs) and
``subtract_sky`` is enabled, the science diagnostics run on a sky-subtracted
cube; the sky-added products and a noiseless "observed truth" reconstruction
stay available on the result for comparison.

and exposes three presets that share the same code path but differ in their
parameters and data provenance:

    ``toy``                tiny, fast, fully synthetic -- for CI and demos.
    ``instrument_inspired`` HARMONI/MUSE-*inspired* parameter presets, clearly
                           labelled ``literature_derived`` (not a reproduction).
    ``public_conditioned`` additionally conditions the cube on optional cached
                           public data (SVO filters, SkyCalc sky/telluric,
                           CALSPEC/PHOENIX source); falls back to synthetic
                           proxies and records which was used.

A run returns a :class:`PipelineResult` holding every intermediate product, a
metrics dict, and a populated provenance log.  :func:`write_artifacts` then
saves figures, a CSV metrics table and a JSON run summary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import SPEED_OF_LIGHT_KM_S
from . import cube_reconstruction as recon
from . import diagnostics as diag
from . import psf_coupling as psf
from .data_sources import (
    ProvenanceLog,
    load_filter_curve,
    load_sky_spectrum,
    load_source_spectrum,
)
from .detector_model import DetectorConfig, calibrate_readout, expose
from .ifu_mapping import IFUConfig, SlicerMapping, build_slicer_mapping, forward_project_cube
from .source_cube import (
    SourceCubeConfig,
    create_emission_line_cube,
    create_extended_galaxy_cube,
    create_point_source_cube,
    photon_budget_from_abmag,
    scale_cube_to_total_electrons,
)
from .spectrograph_model import SpectrographConfig

# Nominal near-IR sky-continuum radiance used to scale the *synthetic* sky proxy
# (whose raw units are arbitrary) onto a SkyCalc-like level, in
# photons / s / m^2 / um / arcsec^2.  An engineering_estimate, not a measurement;
# the cached ESO SkyCalc radiance is used directly when available.
SYNTHETIC_SKY_RADIANCE_PH = 1500.0

# Cache coverage only, NOT a guarantee of useful transmission or a detection.
# Actual filter x atmosphere attenuation is evaluated after configuration overrides.
SUPPORTED_HIGHZ_OBS_UM = (2.00, 2.40)

# Cube estimators that can back the science products (PipelineConfig.science_extraction).
SCIENCE_EXTRACTIONS = ("nearest", "deblend")


@dataclass
class PipelineConfig:
    """All knobs for one pipeline run (built by the mode presets below).

    Grouped by stage; see each stage's module for the physics behind a knob.
    """

    mode: str = "toy"

    # -- source field & photometry --
    nx: int = 12
    ny: int = 12
    spaxel_scale_mas: float = 10.0
    lambda_min_um: float = 2.05
    lambda_max_um: float = 2.35
    exposure_s: float = 600.0
    telescope_diameter_m: float = 8.2
    throughput_scalar: float = 0.25
    ab_magnitude: float = 19.0
    source_kind: str = "emission_line"  # 'emission_line' | 'point' | 'galaxy'

    # -- emission line --
    line_center_um: float | None = None  # None -> band centre
    line_flux_rel: float = 16.0
    # Nearly-unresolved intrinsic line: narrow enough that the recovered line
    # width is dominated by the instrument LSF, so recovered R tracks configured R.
    line_sigma_um: float = 0.00025
    velocity_gradient_kms_per_spaxel: float = 15.0

    # -- photon budget semantics --
    # "total_abmag" (default): the WHOLE relative cube (continuum + line) is
    #   rescaled to the AB-magnitude total, so line_flux_rel only redistributes
    #   a fixed budget towards the line (shape-only, backward compatible).
    # "continuum_line": the continuum is scaled to the AB-magnitude (or surface-
    #   brightness) budget and the line carries its OWN physically-derived
    #   electrons from line_flux_cgs or line_equivalent_width_um; line_flux_rel
    #   then only shapes the line's relative spatial/velocity structure.
    photon_budget_mode: str = "total_abmag"
    # Integrated observed-frame line flux in erg/s/cm^2 (continuum_line mode).
    line_flux_cgs: float | None = None
    # Alternative line strength: equivalent width in um over the continuum.
    line_equivalent_width_um: float | None = None
    # Optional continuum normalisation for extended sources: AB mag / arcsec^2
    # integrated over the field of view (continuum_line mode); when None the
    # continuum uses ab_magnitude as a total magnitude.
    surface_brightness_ab_arcsec2: float | None = None

    # -- spectrograph --
    resolving_power_R: float = 3000.0
    pixels_per_resolution_element: float = 2.5

    # -- IFU image slicer --
    slice_width_spaxels: int = 1
    cross_talk_sigma_pix: float = 0.6
    slice_gap_pix: int = 2

    # -- AO PSF coupling --
    apply_psf: bool = True
    strehl: float = 0.3
    seeing_fwhm_mas: float = 600.0

    # -- detector --
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    # Ideal known bias/dark calibration is always applied before reconstruction.
    # False deliberately leaves response errors for calibration-sensitivity tests.
    correct_flat_field: bool = True

    # -- cube reconstruction --
    # Estimator behind the science products (maps, SNR, aperture spectrum and
    # observed truth); both are always computed and kept on the result.
    # "nearest" reads each voxel's own detector pixel: independent noise and
    # flux-conserving aperture sums, with the cross-talk left as a small extra
    # spatial blur -- what slicer pipelines do (they resample, not deconvolve).
    # "deblend" jointly inverts the cross-talk. Each spaxel owns ONE detector
    # row, so that inversion has no redundant data: it costs ~2-3x per-spaxel
    # variance, and one bad pixel can leave a slice segment unidentifiable.
    science_extraction: str = "nearest"

    # Reporting thresholds, not physical boundaries or calibrated detection limits.
    line_transmission_warning_threshold: float = 0.05
    moment_min_snr: float = 3.0

    # -- run control --
    seed: int = 0
    use_public_cache: bool = False
    # Where the public-data loaders look for cached files. None -> the package
    # default (data/public_cache). Point it at an empty directory to force the
    # synthetic-proxy fallback (used by the public-vs-synthetic comparison).
    public_cache_dir: str | None = None
    # Engineering scaling applied to the sky-background electron level. The
    # SkyCalc-radiance -> electrons conversion is physically motivated but its
    # absolute normalisation is NOT a calibrated ETC; this knob makes that
    # explicit and keeps the demo in a sensible regime.
    sky_scaling: float = 1.0
    # Sky subtraction: when True (the public-conditioned / high-z presets), the
    # science diagnostics are computed on a sky-SUBTRACTED cube. The subtracted
    # model is the injected sky pushed through the SAME forward path (LSF +
    # cross-talk) and the SAME reconstruction estimator, i.e. a perfect-
    # calibration assumption (engineering_estimate, not a real reduction).
    # The raw sky-added products remain available on the result.
    subtract_sky: bool = False
    # Sky-model mismatch experiment: the subtracted model is scaled by
    # (1 + sky_subtraction_scale_error), so a non-zero value leaves a deliberate
    # over-/under-subtraction residual in the reduced products.
    sky_subtraction_scale_error: float = 0.0
    # Optional richer photon budget: when True, the flat instrument throughput
    # scalar is replaced by the band-averaged INSTRUMENT throughput from a
    # decomposed ThroughputBudget (telescope/AO/IFU/spectrograph/QE). Atmosphere
    # and filter stay applied spectrally by the public conditioning, so they are
    # excluded here. Default False -> behaviour and metrics are unchanged.
    use_throughput_budget: bool = False
    # Integer spaxel dither (dy, dx) applied to the source before coupling, so the
    # source lands on different spaxels/detector pixels per exposure. Used by the
    # observing-strategy layer; default (0, 0) -> no shift (backward compatible).
    source_dither_spaxels: tuple[int, int] = (0, 0)


def make_toy_config(**overrides) -> PipelineConfig:
    """Tiny, fast, fully-synthetic configuration for CI and quick demos."""
    cfg = PipelineConfig(
        mode="toy",
        nx=12, ny=12,
        lambda_min_um=2.10, lambda_max_um=2.30,
        resolving_power_R=2000.0,
        telescope_diameter_m=8.0,
        ab_magnitude=17.5,
        cross_talk_sigma_pix=0.6,
        strehl=0.3,
        detector=DetectorConfig(read_noise_e=10.0, dark_current_e_s_pix=0.01),
    )
    return _apply_overrides(cfg, overrides)


def make_instrument_inspired_config(**overrides) -> PipelineConfig:
    """HARMONI/MUSE-*inspired* preset (literature_derived parameter ranges).

    Uses an ELT-scale aperture and a fine spaxel scale to resemble published
    AO-fed IFU modes.  These are plausible mode parameters, **not** a reproduction
    of any instrument's as-built configuration.
    """
    cfg = PipelineConfig(
        mode="instrument_inspired",
        nx=20, ny=20,
        spaxel_scale_mas=10.0,            # HARMONI-inspired fine scale
        lambda_min_um=1.95, lambda_max_um=2.45,  # K-band-ish
        resolving_power_R=3000.0,
        telescope_diameter_m=39.0,        # ELT-scale aperture
        throughput_scalar=0.3,
        ab_magnitude=20.0,
        strehl=0.5,                       # decent AO correction in K
        seeing_fwhm_mas=600.0,
        cross_talk_sigma_pix=0.6,
        detector=DetectorConfig(read_noise_e=8.0, dark_current_e_s_pix=0.005, full_well_e=8.0e4),
    )
    return _apply_overrides(cfg, overrides)


def make_public_conditioned_config(**overrides) -> PipelineConfig:
    """Like instrument_inspired but also conditions on optional cached public data.

    Sky subtraction is ON by default: the science diagnostics of a
    public-conditioned run are computed on the sky-subtracted (reduced) cube,
    with the raw sky-added products kept alongside.  Override
    ``subtract_sky=False`` to diagnose the sky-added cube instead.
    """
    cfg = make_instrument_inspired_config(
        mode="public_conditioned", use_public_cache=True, subtract_sky=True
    )
    return _apply_overrides(cfg, overrides)


def make_highz_galaxy_config(line: str = "Halpha", redshift: float = 2.30, **overrides) -> PipelineConfig:
    """High-z emission-line galaxy preset (public-conditioned, extended source).

    Places a **redshifted** nebular line (default H-alpha) into the simulated
    near-IR band, with an extended-galaxy spatial profile, a rotation field and an
    intrinsic line-dispersion proxy.  Public data (SVO filter, SkyCalc
    telluric/sky, CALSPEC continuum) conditions the cube when the cache is present.

    This is a **synthetic science-recovery scenario**, not a real galaxy
    observation and not a validated redshift/flux measurement.
    """
    from .source_cube import REST_FRAME_LINES_UM, observed_wavelength_um

    if line not in REST_FRAME_LINES_UM:
        raise ValueError(f"Unknown line {line!r}; choose from {list(REST_FRAME_LINES_UM)}")
    obs = observed_wavelength_um(REST_FRAME_LINES_UM[line], redshift)
    # The observed line must land inside the band supported by the cached public
    # data (SVO Ks ~1.95-2.40, SkyCalc 1.90-2.50, CALSPEC 0.30-2.60 um). Reject
    # redshift/line combinations that fall outside this overlap rather than
    # silently clamping to a degenerate band.
    if not (SUPPORTED_HIGHZ_OBS_UM[0] <= obs <= SUPPORTED_HIGHZ_OBS_UM[1]):
        raise ValueError(
            f"observed {line} (rest {REST_FRAME_LINES_UM[line]:.4f} um) at z={redshift} "
            f"is {obs:.4f} um, outside the supported public-cache coverage "
            f"[{SUPPORTED_HIGHZ_OBS_UM[0]}-{SUPPORTED_HIGHZ_OBS_UM[1]}] um. "
            f"Choose a redshift that places the line in that range "
            f"(e.g. z ~ {SUPPORTED_HIGHZ_OBS_UM[0] / REST_FRAME_LINES_UM[line] - 1:.2f}"
            f"-{SUPPORTED_HIGHZ_OBS_UM[1] / REST_FRAME_LINES_UM[line] - 1:.2f})."
        )
    # Band around the observed line, clamped to the cached public-data overlap.
    lam_min = max(1.95, obs - 0.18)
    lam_max = min(2.45, obs + 0.18)
    cfg = make_public_conditioned_config(
        mode="highz_galaxy",
        nx=20, ny=20,
        lambda_min_um=round(lam_min, 4), lambda_max_um=round(lam_max, 4),
        resolving_power_R=4000.0,
        source_kind="galaxy",
        line_center_um=obs,
        line_sigma_um=0.0008,        # intrinsic dispersion proxy (~100 km/s at 2 um)
        line_flux_rel=14.0,
        velocity_gradient_kms_per_spaxel=12.0,
        ab_magnitude=21.0,           # a faint high-z source
        exposure_s=900.0,            # a deeper integration so recovery is demonstrable
    )
    return _apply_overrides(cfg, overrides)


def _apply_overrides(cfg: PipelineConfig, overrides: dict) -> PipelineConfig:
    """Return ``cfg`` with any provided attributes overridden (validated)."""
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"PipelineConfig has no field {key!r}")
        setattr(cfg, key, value)
    return cfg


@dataclass
class PipelineResult:
    """Container for every product of a pipeline run."""

    config: PipelineConfig
    source_config: SourceCubeConfig
    spectrograph: SpectrographConfig
    ifu_config: IFUConfig
    wavelength_grid_um: np.ndarray
    line_center_um: float

    # Truth surfaces at increasing realism (all in electrons):
    #   intrinsic_source_cube  synthetic source after the photon budget, BEFORE
    #                          any public conditioning (continuum shape /
    #                          telluric / filter) -- the intrinsic truth;
    #   source_cube            after public conditioning (== intrinsic when the
    #                          run is not public-conditioned) -- pre-PSF truth;
    #   coupled_cube           after AO-PSF coupling -- the observed scene;
    #   observed_truth_cube    (below) the noiseless, sky-free source run through
    #                          the SAME reduction operator (LSF + cross-talk +
    #                          the science extraction) -- a REDUCTION-LEVEL
    #                          observed truth, not an intrinsic truth cube.
    source_cube: np.ndarray
    coupled_cube: np.ndarray           # after PSF coupling
    detector_clean: np.ndarray         # noiseless detector frame
    detector_noisy: np.ndarray         # RAW noisy detector frame (bias included)
    variance_frame: np.ndarray         # CALIBRATED variance used in reconstruction
    bad_pixel_mask: np.ndarray

    rebuilt_nearest: np.ndarray
    rebuilt_weighted: np.ndarray       # joint least-squares deblend
    # The variance of the science estimator (``config.science_extraction``).
    # The per-estimator variances are exposed explicitly below; pair flux and
    # variance from the SAME estimator when building SNRs.
    variance_cube: np.ndarray
    variance_cube_nearest: np.ndarray
    variance_cube_weighted: np.ndarray

    line_flux_map: np.ndarray
    velocity_map: np.ndarray
    snr_map: np.ndarray
    extracted_spectrum: np.ndarray
    aperture_mask: np.ndarray

    psf_metrics: psf.PSFMetrics | None
    metrics: dict
    provenance: ProvenanceLog

    # Public-conditioning products (zeros / empty when not public-conditioned).
    # Sky electrons are integrated detector bins AFTER the spectral LSF, before
    # spatial trace cross-talk. Never apply the LSF to this spectrum again.
    sky_background_spectrum: np.ndarray | None = None
    conditioning_info: dict | None = None

    # Intrinsic truth: the photon-budget-scaled synthetic cube BEFORE public
    # conditioning (== source_cube for non-public-conditioned runs).
    intrinsic_source_cube: np.ndarray | None = None

    # --- Reduction-level products (None when not applicable). ---
    # REDUCTION-MATCHED observed truth: the noiseless, sky-free source frame run
    # through the SAME extraction as the science cube.  It is the surface the
    # recovered products should be compared against (post-PSF/LSF/cross-talk,
    # NOT deconvolved), so it is *reduction-specific*, not an intrinsic truth.
    # The nearest read-out makes it deterministic; the deblend shares the
    # science variance weights, whose seed-drawn bad-pixel mask makes it weakly
    # seed-dependent.  Use source_cube / intrinsic_source_cube for a
    # pre-reduction truth.
    observed_truth_cube: np.ndarray | None = None
    # Sky-subtracted science cubes (only when a sky background was injected and
    # cfg.subtract_sky is True); the subtracted model is the forward-modelled
    # sky reconstructed with the same estimator, scaled by
    # (1 + cfg.sky_subtraction_scale_error).
    rebuilt_weighted_skysub: np.ndarray | None = None
    rebuilt_nearest_skysub: np.ndarray | None = None
    # The (noiseless, forward-modelled) sky cube that was subtracted, at scale 1.
    sky_cube_estimate_weighted: np.ndarray | None = None
    # Banded normal matrices of the joint weighted extraction (per wavelength
    # bin): lets diagnostics propagate the exact spatial covariance of the
    # reconstruction (aperture sums are NOT sums of independent spaxels).
    extraction_model: object | None = None
    detector_calibrated: np.ndarray | None = None
    raw_variance_frame: np.ndarray | None = None
    detector_invalid_mask: np.ndarray | None = None
    line_flux_partial_map: np.ndarray | None = None
    snr_partial_map: np.ndarray | None = None
    line_window_coverage_map: np.ndarray | None = None
    # The cube the science diagnostics ran on: the science estimator's
    # reconstruction, sky-subtracted when sky subtraction was applied.
    science_cube: np.ndarray | None = None

    @property
    def science_extraction(self) -> str:
        """Estimator behind the science products: ``"nearest"`` or ``"deblend"``."""
        return self.config.science_extraction

    @property
    def science_cube_sky_added(self) -> np.ndarray:
        """The science estimator's reconstruction BEFORE any sky subtraction."""
        return self.rebuilt_weighted if self.science_extraction == "deblend" else self.rebuilt_nearest

    @property
    def science_extraction_model(self):
        """Covariance model behind the science cube, or None for nearest.

        Nearest voxels read distinct detector pixels, so their noise is
        independent and aperture variances simply add; the deblend couples
        neighbouring spaxels, so aperture sums need its exact covariance.
        """
        return self.extraction_model if self.science_extraction == "deblend" else None


def _build_source_cube(
    cfg: PipelineConfig,
    src_cfg: SourceCubeConfig,
    line_center_um: float,
    line_flux_rel: float | None = None,
) -> np.ndarray:
    """Dispatch to the requested source-cube builder.

    ``line_flux_rel`` overrides the configured relative line amplitude; the
    continuum/line photon-budget mode uses it to build a continuum-only cube
    (``line_flux_rel=0``) alongside the full one.
    """
    lfr = cfg.line_flux_rel if line_flux_rel is None else line_flux_rel
    # With public conditioning the (photon-shape) source spectrum supplies the
    # continuum, so the synthetic cube starts FLAT; otherwise the synthetic
    # blackbody continuum would be multiplied underneath it (double shaping).
    temperature = {"point": 6000.0, "galaxy": 5000.0}.get(cfg.source_kind)
    if cfg.use_public_cache:
        temperature = None
    if cfg.source_kind == "emission_line":
        return create_emission_line_cube(
            src_cfg, line_center_um=line_center_um, line_flux_rel=lfr,
            line_sigma_um=cfg.line_sigma_um,
            velocity_gradient_kms_per_spaxel=cfg.velocity_gradient_kms_per_spaxel,
        )
    if cfg.source_kind == "point":
        return create_point_source_cube(
            src_cfg, source_temperature_k=temperature,
            line_center_um=line_center_um, line_flux_rel=lfr,
            line_sigma_um=cfg.line_sigma_um,
        )
    if cfg.source_kind == "galaxy":
        return create_extended_galaxy_cube(
            src_cfg, source_temperature_k=temperature,
            line_center_um=line_center_um, line_flux_rel=lfr,
            line_sigma_um=cfg.line_sigma_um,
            velocity_gradient_kms_per_spaxel=cfg.velocity_gradient_kms_per_spaxel,
        )
    raise ValueError(f"Unknown source_kind {cfg.source_kind!r}")


def _apply_public_conditioning(
    cube: np.ndarray,
    wavelength_grid_um: np.ndarray,
    src_cfg: SourceCubeConfig,
    cfg: PipelineConfig,
    prov: ProvenanceLog,
    *,
    continuum_cube: np.ndarray,
    spectrograph: SpectrographConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Push public (or proxy) data through the *physical* chain, not just the log.

    Conditions the source cube on three public products and builds a sky
    background, returning ``(conditioned_cube, sky_electrons_spectrum, info)``:

    1. **Source continuum shape** (CALSPEC ``source.dat``): the cube's continuum
       is multiplied by the standard-star spectral shape (mean-normalised so it
       reshapes the spectrum without changing the photon-budget normalisation).
    2. **Telluric absorption** (ESO SkyCalc ``transmission``): the *source* light
       is attenuated by the atmosphere, ``cube *= transmission(lambda)``.
    3. **Filter bandpass** (SVO 2MASS Ks ``transmission``): a wavelength-dependent
       throughput multiplies the source, imposing the K-band bandpass shape.
    4. **Sky background** (ESO SkyCalc ``radiance``): converted to electrons per
       spaxel per wavelength bin via
       ``integral(LSF(radiance * filter)) * area * exposure * throughput * Omega_spaxel``
       where ``Omega_spaxel`` is the spaxel solid angle.  This is a physically
       *motivated* conversion, but its absolute normalisation is an
       **engineering_estimate**, not a calibrated ETC (see ``sky_scaling``).

    Provenance is honest: the loaders return ``direct_public_data`` only when a
    real cache file is found (under ``cfg.public_cache_dir`` or the default),
    otherwise ``synthetic_proxy`` -- so pointing ``public_cache_dir`` at an empty
    directory cleanly flips the whole chain to the synthetic fallback.
    """
    cache_dir = cfg.public_cache_dir
    sky, sky_prov = load_sky_spectrum(wavelength_grid_um, cache_dir=cache_dir)
    prov.records.append(sky_prov)
    source_spec, src_prov = load_source_spectrum(wavelength_grid_um, cache_dir=cache_dir)
    prov.records.append(src_prov)
    filt, filt_prov = load_filter_curve("K", cache_dir=cache_dir)
    prov.records.append(filt_prov)

    # Filter transmission and telluric transmission on the cube wavelength grid.
    filt_t = np.clip(
        np.interp(wavelength_grid_um, filt["wavelength_um"], filt["transmission"], left=0.0, right=0.0),
        0.0, 1.0,
    )
    telluric = np.clip(sky["transmission"], 0.0, 1.0)

    # Source continuum PHOTON shape (mean-normalised: reshape, don't rescale the
    # budget).  The base cube was built with a flat continuum for conditioned
    # runs, so this is the only continuum shape applied.
    shape = source_spec["flux"]
    shape = shape / np.mean(shape) if np.mean(shape) > 0 else np.ones_like(shape)

    # The source spectrum describes the CONTINUUM, not an additional throughput.
    # Keep emission-line photons (including an explicit line_flux_cgs budget)
    # independent of that shape; atmosphere and filter attenuate both parts.
    line_cube = cube - continuum_cube
    shaped_source = continuum_cube * shape[None, None, :] + line_cube
    conditioned = shaped_source * telluric[None, None, :] * filt_t[None, None, :]
    line_center = cfg.line_center_um if cfg.line_center_um is not None else src_cfg_mid(src_cfg)
    if not np.isfinite(line_center) or not cfg.lambda_min_um <= line_center <= cfg.lambda_max_um:
        raise ValueError("line_center_um must be finite and inside the configured wavelength band")
    if not np.isfinite(cfg.moment_min_snr) or cfg.moment_min_snr <= 0:
        raise ValueError("moment_min_snr must be finite and positive")
    transmission = telluric * filt_t
    transmission_at_line = float(np.interp(
        line_center, wavelength_grid_um, transmission, left=0.0, right=0.0
    ))
    line_weights = np.clip(line_cube, 0.0, None).sum(axis=(0, 1))
    line_weighted_transmission = (
        float(np.dot(line_weights, transmission) / line_weights.sum())
        if line_weights.sum() > 0 else float("nan")
    )
    threshold = cfg.line_transmission_warning_threshold
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("line_transmission_warning_threshold must be in [0, 1]")
    if not line_weights.sum() > 0:
        line_status = "no_intrinsic_line"
    elif line_weighted_transmission == 0:
        line_status = "blocked"
    elif min(transmission_at_line, line_weighted_transmission) < threshold:
        line_status = "low_transmission"
    else:
        line_status = "transmitted"
    if line_status in ("blocked", "low_transmission"):
        import warnings
        warnings.warn(
            f"Line at {line_center:.6f} um has {line_status}: filter x telluric "
            f"transmission at centre={transmission_at_line:.4g}, "
            f"line-photon-weighted={line_weighted_transmission:.4g}. "
            "Cache coverage is not a detectability guarantee; inspect detection flags.",
            RuntimeWarning, stacklevel=2,
        )

    # --- Sky background electrons per spaxel per wavelength bin. ---
    # Work on native samples until AFTER filtering and instrumental broadening.
    # Point-interpolating a narrow sky line onto detector centres first can
    # discard almost all of its photons (or amplify it depending on bin phase).
    native_grid = sky["native_wavelength_um"]
    radiance = np.clip(np.asarray(sky["native_radiance"], dtype=float), 0.0, None)
    if sky_prov.source_class != "direct_public_data":
        # The synthetic proxy radiance has arbitrary units; rescale its mean to a
        # SkyCalc-like physical level so the demo still shows a sky contribution.
        mean_r = float(np.mean(sky["radiance"]))
        radiance = radiance / mean_r * SYNTHETIC_SKY_RADIANCE_PH if mean_r > 0 else np.full_like(radiance, SYNTHETIC_SKY_RADIANCE_PH)
    native_filter = np.clip(np.interp(
        native_grid, filt["wavelength_um"], filt["transmission"], left=0.0, right=0.0
    ), 0.0, 1.0)
    sky_bin_integrals = spectrograph.integrate_spectral_density(
        native_grid, radiance * native_filter
    )
    spaxel_solid_angle_arcsec2 = (cfg.spaxel_scale_mas / 1000.0) ** 2
    sky_e = (
        sky_bin_integrals
        * src_cfg.collecting_area_m2
        * cfg.exposure_s
        # Resolved instrument throughput (== cfg.throughput_scalar unless a
        # throughput budget is enabled, in which case src_cfg already carries it).
        * src_cfg.throughput_scalar
        * spaxel_solid_angle_arcsec2
        * cfg.sky_scaling
    )

    # Effective in-band throughput = source-weighted mean filter transmission.
    src_weight = np.clip(shaped_source.sum(axis=(0, 1)), 0.0, None)
    if np.sum(src_weight) > 0:
        filter_weighted_throughput = float(np.sum(filt_t * src_weight) / np.sum(src_weight))
    else:
        filter_weighted_throughput = float(np.mean(filt_t))

    components = {
        "sky_spectrum": sky_prov.source_class,
        "source_spectrum": src_prov.source_class,
        "filter_K": filt_prov.source_class,
    }
    n_public = sum(1 for v in components.values() if v == "direct_public_data")

    prov.add(
        "filter_throughput", filt_prov.source_class,
        "SVO 2MASS Ks transmission as a wavelength-dependent bandpass",
        notes="Filter curve multiplies the source spectrum (bandpass weighting), not a flat scalar.",
    )
    prov.add(
        "sky_background", "engineering_estimate",
        "native sky * filter -> spectral LSF -> detector-bin integration -> electrons",
        notes=(
            "Physically-motivated unit conversion; ABSOLUTE level is an engineering estimate "
            f"(sky_scaling={cfg.sky_scaling}), NOT a calibrated ETC. "
            "LSF applied once before flux-conserving bin integration; source/telluric "
            "conditioning still uses the science grid."
        ),
    )

    info = {
        "transmission_at_line": transmission_at_line,
        "line_weighted_transmission": line_weighted_transmission,
        "line_transmission_status": line_status,
        "line_transmission_warning_threshold": float(threshold),
        "filter_weighted_throughput": filter_weighted_throughput,
        "components": components,
        "public_data_components_used": n_public,
        "sky_lsf_applied": True,
        "sky_resampling": "native-filter-lsf-bin-integral",
        # Sky-cache sampling relative to the science grid: a cache coarser than
        # the grid cannot support line-scale sky claims (OH mask fractions).
        "sky_cache_step_nm": float(sky.get("cache_step_um", float("nan"))) * 1e3,
        "sky_grid_step_nm": float(sky.get("grid_step_um", float("nan"))) * 1e3,
        "sky_cache_undersampled": bool(sky.get("cache_undersampled", False)),
    }
    return conditioned, sky_e, info


def _apply_photon_budget(
    cube: np.ndarray,
    cfg: PipelineConfig,
    src_cfg: SourceCubeConfig,
    line_center: float,
    nlambda: int,
    prov: ProvenanceLog,
) -> dict:
    """Scale the relative cube onto an absolute electron scale.

    Two modes (both engineering estimates, never a calibrated ETC):

    ``total_abmag`` (default, backward compatible)
        The whole cube (continuum + line) is rescaled so its total equals the
        AB-magnitude budget.  ``line_flux_rel`` is then a *shape* parameter:
        it redistributes a fixed total towards the line without adding photons.

    ``continuum_line``
        The continuum-only cube is scaled to the AB-magnitude (or optional
        surface-brightness) budget, and the line component carries its OWN
        electrons derived from a physical ``line_flux_cgs`` (erg/s/cm^2) or a
        ``line_equivalent_width_um``.  Adding line flux now adds photons on
        top of an unchanged continuum.

    Returns ``{"cube", "continuum_cube", "mode", "continuum_total_electrons",
    "line_total_electrons", "line_in_band_electrons"}`` (the totals are NaN in
    ``total_abmag`` mode, where the split is not defined).  The physical line
    budget is the line's TOTAL flux: a profile cut by the band edge keeps only
    its captured fraction (``line_in_band_electrons``); the missing wing is
    lost, never redistributed into the surviving channels.
    """
    from .source_cube import (
        line_electrons_from_equivalent_width,
        photon_budget_from_surface_brightness,
        photon_budget_line_electrons,
    )

    mode = cfg.photon_budget_mode
    if mode not in ("total_abmag", "continuum_line"):
        raise ValueError(
            f"Unknown photon_budget_mode {mode!r}; expected 'total_abmag' or 'continuum_line'"
        )

    if mode == "total_abmag":
        for name, value in [("line_flux_cgs", cfg.line_flux_cgs),
                            ("line_equivalent_width_um", cfg.line_equivalent_width_um),
                            ("surface_brightness_ab_arcsec2", cfg.surface_brightness_ab_arcsec2)]:
            if value is not None:
                raise ValueError(
                    f"{name} requires photon_budget_mode='continuum_line' "
                    "(the default 'total_abmag' mode rescales the whole cube to the "
                    "AB-magnitude total and cannot honour a physical line flux)."
                )
        electrons_per_bin = photon_budget_from_abmag(src_cfg, cfg.ab_magnitude)
        total_electrons = max(electrons_per_bin * nlambda, 1.0)
        continuum_rel = _build_source_cube(cfg, src_cfg, line_center, line_flux_rel=0.0)
        # Apply the SAME normalization to both components, so their sum retains
        # the shape-only total budget and later continuum shaping leaves the
        # additive line untouched.
        normalization = total_electrons / float(np.sum(cube))
        return {
            "cube": scale_cube_to_total_electrons(cube, total_electrons),
            "continuum_cube": continuum_rel * normalization,
            "mode": mode,
            "continuum_total_electrons": float("nan"),
            "line_total_electrons": float("nan"),
            "line_in_band_electrons": float("nan"),
        }

    # --- continuum_line mode. ---
    if (cfg.line_flux_cgs is None) == (cfg.line_equivalent_width_um is None):
        raise ValueError(
            "photon_budget_mode='continuum_line' needs exactly one of "
            "line_flux_cgs or line_equivalent_width_um."
        )

    # Continuum budget: total AB magnitude, or surface brightness for extended
    # sources (AB mag / arcsec^2 integrated over the field of view).
    if cfg.surface_brightness_ab_arcsec2 is not None:
        electrons_per_bin = photon_budget_from_surface_brightness(
            src_cfg, cfg.surface_brightness_ab_arcsec2
        )
    else:
        electrons_per_bin = photon_budget_from_abmag(src_cfg, cfg.ab_magnitude)
    continuum_total = max(electrons_per_bin * nlambda, 1.0)

    # Split the relative cube into continuum-only and line-only components; the
    # line term is additive in the builders, so the difference is exact.
    continuum_rel = _build_source_cube(cfg, src_cfg, line_center, line_flux_rel=0.0)
    line_rel = np.clip(cube - continuum_rel, 0.0, None)
    if float(np.sum(line_rel)) <= 0:
        raise ValueError(
            "continuum_line budget requested but the source cube carries no line "
            "component (line_flux_rel is 0 or no line is configured)."
        )

    if cfg.line_flux_cgs is not None:
        line_total = photon_budget_line_electrons(src_cfg, cfg.line_flux_cgs, line_center)
    else:
        line_total = line_electrons_from_equivalent_width(
            electrons_per_bin, src_cfg.delta_lambda_um, cfg.line_equivalent_width_um
        )

    continuum_scaled = scale_cube_to_total_electrons(continuum_rel, continuum_total)
    scaled = continuum_scaled
    # Normalise the line by its UNTRUNCATED profile (same builder, same channel
    # spacing, on a grid extended past every Doppler-shifted wing), so the band
    # keeps only the captured fraction of the total line electrons.
    captured = float(np.sum(line_rel)) / _untruncated_line_sum(cfg, src_cfg, line_center)
    line_in_band = line_total * captured
    if line_total > 0:
        scaled = scaled + scale_cube_to_total_electrons(line_rel, line_in_band)
    prov.add(
        "photon_budget", "engineering_estimate",
        "separate continuum (AB mag / surface brightness) and line (flux or EW) budgets",
        notes=(
            f"continuum {continuum_total:.3e} e- + line {line_total:.3e} e-; the line adds "
            "photons on top of the continuum (physical-line mode), not a calibrated ETC."
        ),
    )
    return {
        "cube": scaled,
        "continuum_cube": continuum_scaled,
        "mode": mode,
        "continuum_total_electrons": float(continuum_total),
        "line_total_electrons": float(line_total),
        "line_in_band_electrons": float(line_in_band),
    }


def _untruncated_line_sum(cfg: PipelineConfig, src_cfg: SourceCubeConfig, line_center: float) -> float:
    """Sum of the relative line component on a wavelength grid wide enough to
    hold the whole (Doppler-shifted) line profile.

    Uses the same builder and channel spacing as the in-band cube, extended on
    both sides, so ``in-band sum / untruncated sum`` is the captured fraction in
    the model's own discretisation (1 for a line wholly inside the band).
    """
    step = src_cfg.delta_lambda_um
    v_max = abs(cfg.velocity_gradient_kms_per_spaxel) * float(max(cfg.nx, cfg.ny))
    reach = 10.0 * cfg.line_sigma_um + abs(line_center) * v_max / SPEED_OF_LIGHT_KM_S
    pad = int(np.ceil(reach / step)) + 1
    extended = replace(src_cfg, nlambda=src_cfg.nlambda + 2 * pad,
                       lambda_min_um=src_cfg.lambda_min_um - pad * step,
                       lambda_max_um=src_cfg.lambda_max_um + pad * step)
    full = _build_source_cube(cfg, extended, line_center)
    continuum = _build_source_cube(cfg, extended, line_center, line_flux_rel=0.0)
    return float(np.sum(np.clip(full - continuum, 0.0, None)))


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """Run the full source -> detector -> reconstruction -> diagnostics pipeline.

    Each numbered stage is a small helper below; this function only wires their
    products together.  The configuration is snapshotted (deep-copied) when the
    run starts, and the result keeps that snapshot: reusing or mutating the
    caller's object later (e.g. in a parameter sweep) cannot change what a
    saved result reports.
    """
    cfg = copy.deepcopy(cfg)
    if cfg.science_extraction not in SCIENCE_EXTRACTIONS:
        raise ValueError(
            f"science_extraction must be one of {SCIENCE_EXTRACTIONS}, got {cfg.science_extraction!r}"
        )
    rng = np.random.default_rng(cfg.seed)
    prov = ProvenanceLog()

    spectro, src_cfg, effective_throughput, line_center = _set_up_instrument(cfg, prov)
    wavelength_grid = src_cfg.wavelength_grid_um
    source = _build_source(cfg, src_cfg, spectro, line_center, prov)                       # 1
    coupled, psf_metrics = _couple_psf(cfg, source.cube, src_cfg, wavelength_grid, prov)   # 2
    fwd = _forward_model(cfg, coupled, source.sky_e_spectrum, spectro, wavelength_grid, prov)  # 3-4
    readout, calibrated = _read_out_detector(cfg, fwd.detector_clean, rng, prov)          # 5
    red = _reconstruct(cfg, calibrated, fwd, prov)                                         # 6
    sci = _science_products(cfg, red, spectro, wavelength_grid, line_center, source)      # 7
    metrics = _collect_metrics(
        cfg, spectro, source, coupled, psf_metrics, fwd, calibrated, red, sci, effective_throughput
    )

    return PipelineResult(
        config=cfg, source_config=src_cfg, spectrograph=spectro, ifu_config=fwd.ifu_cfg,
        wavelength_grid_um=wavelength_grid, line_center_um=line_center,
        source_cube=source.cube, coupled_cube=coupled,
        detector_clean=fwd.detector_clean, detector_noisy=readout.image_e,
        variance_frame=calibrated.variance_e2, bad_pixel_mask=readout.bad_pixel_mask,
        detector_calibrated=calibrated.image_e, raw_variance_frame=readout.variance_e2,
        detector_invalid_mask=calibrated.invalid_pixel_mask,
        rebuilt_nearest=red.rebuilt_nearest, rebuilt_weighted=red.rebuilt_weighted,
        variance_cube=red.science_variance,
        variance_cube_nearest=red.variance_nearest, variance_cube_weighted=red.variance_weighted,
        line_flux_map=sci.line_maps["line_flux"], velocity_map=sci.velocity_map,
        snr_map=sci.line_maps["snr"],
        line_flux_partial_map=sci.line_maps["partial_line_flux"],
        snr_partial_map=sci.line_maps["partial_snr"],
        line_window_coverage_map=sci.line_maps["measured_channel_fraction"],
        extracted_spectrum=sci.spectrum, aperture_mask=sci.aperture,
        psf_metrics=psf_metrics, metrics=metrics, provenance=prov,
        sky_background_spectrum=source.sky_e_spectrum, conditioning_info=source.cond_info,
        intrinsic_source_cube=source.intrinsic_cube,
        observed_truth_cube=red.observed_truth_cube,
        rebuilt_weighted_skysub=red.rebuilt_weighted_skysub,
        rebuilt_nearest_skysub=red.rebuilt_nearest_skysub,
        sky_cube_estimate_weighted=red.sky_cube_weighted,
        extraction_model=red.extraction_model,
        science_cube=red.science_cube,
    )


@dataclass
class _SourceStage:
    """Stage 1 output: the source cube entering PSF coupling, plus its sky."""

    cube: np.ndarray            # after public conditioning (== intrinsic otherwise)
    intrinsic_cube: np.ndarray  # photon-budget-scaled, before any conditioning
    budget_info: dict
    sky_e_spectrum: np.ndarray  # sky electrons per spaxel per bin (zeros = no sky)
    cond_info: dict


@dataclass
class _ForwardStage:
    """Stages 3-4 output: the slicer map and the noiseless detector frames."""

    ifu_cfg: IFUConfig
    mapping: SlicerMapping
    detector_source_clean: np.ndarray  # source only, for flux/throughput accounting
    detector_clean: np.ndarray         # source + sky
    has_sky: bool


@dataclass
class _Reduction:
    """Stage 6 output: both reconstructions and the selected science cube."""

    rebuilt_nearest: np.ndarray
    variance_nearest: np.ndarray
    rebuilt_weighted: np.ndarray
    variance_weighted: np.ndarray
    extraction_model: recon.TraceExtractionModel
    observed_truth_cube: np.ndarray
    # Per-estimator cubes the diagnostics run on (sky-subtracted when applied).
    science_nearest: np.ndarray
    science_weighted: np.ndarray
    rebuilt_nearest_skysub: np.ndarray | None
    rebuilt_weighted_skysub: np.ndarray | None
    sky_cube_weighted: np.ndarray | None
    sky_subtraction_applied: bool
    # The cfg.science_extraction pair.
    science_cube: np.ndarray
    science_variance: np.ndarray


@dataclass
class _ScienceStage:
    """Stage 7 output: maps, aperture spectrum and summary diagnostics."""

    window: float
    line_maps: dict
    velocity_map: np.ndarray
    moment_detected: np.ndarray
    measured_snr: np.ndarray
    snr_by_method: dict
    aperture: np.ndarray
    spectrum: np.ndarray
    recovered_R: float
    velocity_span: float
    sky_line_masked_frac: float
    sky_line_masked_in_window_frac: float


def _set_up_instrument(
    cfg: PipelineConfig, prov: ProvenanceLog
) -> tuple[SpectrographConfig, SourceCubeConfig, float, float]:
    """Spectrograph, source grid, effective throughput and line centre.

    The spectrograph comes first: it fixes the number of detector columns, and
    the cube's wavelength sampling is matched to it for clean invertibility.
    """
    spectro = SpectrographConfig(
        lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um,
        resolving_power_R=cfg.resolving_power_R,
        pixels_per_resolution_element=cfg.pixels_per_resolution_element,
    )
    prov.add(
        "wavelength_range_R", "literature_derived",
        "HARMONI overview arXiv:2103.11215; MUSE arXiv:astro-ph/0606329",
        notes=f"{cfg.lambda_min_um}-{cfg.lambda_max_um} um, R={cfg.resolving_power_R}; mode presets only.",
    )

    # Optional richer photon budget: replace the flat instrument throughput scalar
    # by the band-averaged INSTRUMENT throughput from a decomposed budget. Default
    # off -> effective_throughput == cfg.throughput_scalar and nothing changes.
    effective_throughput = cfg.throughput_scalar
    if cfg.use_throughput_budget:
        from .throughput_budget import INSTRUMENT_STAGES, build_default_budget

        tb_grid = np.linspace(cfg.lambda_min_um, cfg.lambda_max_um, 200)
        tb = build_default_budget(tb_grid, cache_dir=cfg.public_cache_dir, include=INSTRUMENT_STAGES)
        effective_throughput = tb.effective_throughput(tb_grid)
        prov.add(
            "throughput_budget", "engineering_estimate",
            "decomposed instrument throughput (telescope/AO relay/IFU/spectrograph/QE)",
            notes=(
                f"effective instrument throughput {effective_throughput:.3f} replaces the flat "
                f"scalar {cfg.throughput_scalar}; engineering estimate, not a calibrated budget."
            ),
        )

    src_cfg = SourceCubeConfig(
        nx=cfg.nx, ny=cfg.ny, nlambda=spectro.n_detector_pixels,
        spaxel_scale_mas=cfg.spaxel_scale_mas,
        lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um,
        exposure_s=cfg.exposure_s, telescope_diameter_m=cfg.telescope_diameter_m,
        throughput_scalar=effective_throughput,
    )
    line_center = cfg.line_center_um if cfg.line_center_um is not None else src_cfg_mid(src_cfg)
    if not np.isfinite(line_center) or not cfg.lambda_min_um <= line_center <= cfg.lambda_max_um:
        raise ValueError("line_center_um must be finite and inside the configured wavelength band")
    if not np.isfinite(cfg.moment_min_snr) or cfg.moment_min_snr <= 0:
        raise ValueError("moment_min_snr must be finite and positive")

    prov.add(
        "telescope_diameter", "literature_derived",
        "VLT/ELT-like presets (HARMONI overview arXiv:2103.11215)",
        notes=f"{cfg.telescope_diameter_m} m aperture; not a facility prediction.",
    )
    prov.add(
        "ifu_spaxel_scale", "literature_derived",
        "HARMONI image slicer arXiv:1807.06313; MUSE arXiv:astro-ph/0606329",
        notes=f"{cfg.spaxel_scale_mas} mas spaxel; inspired values, not a reproduction.",
    )
    return spectro, src_cfg, effective_throughput, line_center


def _build_source(
    cfg: PipelineConfig,
    src_cfg: SourceCubeConfig,
    spectro: SpectrographConfig,
    line_center: float,
    prov: ProvenanceLog,
) -> _SourceStage:
    """Stage 1: source datacube, photon budget, dither and public conditioning."""
    nlambda = src_cfg.nlambda
    cube = _build_source_cube(cfg, src_cfg, line_center)
    budget_info = _apply_photon_budget(cube, cfg, src_cfg, line_center, nlambda, prov)
    cube = budget_info["cube"]
    continuum_cube = budget_info["continuum_cube"]
    # Optional integer dither: shift the source on the spaxel grid (the telescope
    # points slightly differently each exposure). np.roll conserves flux exactly.
    dy, dx = cfg.source_dither_spaxels
    if dy or dx:
        cube = np.roll(cube, shift=(int(dy), int(dx)), axis=(0, 1))
        continuum_cube = np.roll(continuum_cube, shift=(int(dy), int(dx)), axis=(0, 1))
    # Intrinsic truth: photon-budget-scaled (and dithered) synthetic cube BEFORE
    # any public conditioning reshapes it. Kept for truth-level comparisons.
    intrinsic_cube = cube
    # The base cube's continuum + line is always synthetic; absolute scaling comes
    # from the AB-magnitude photon budget (an engineering estimate).  When public
    # conditioning is requested, _apply_public_conditioning adds the *actual*
    # provenance of the sky/source/filter data it used (public if cached, else proxy).
    # Default to "no sky, no public data" for the toy / instrument_inspired modes.
    sky_e_spectrum = np.zeros(nlambda, dtype=float)
    cond_info = {
        "transmission_at_line": 1.0, "line_weighted_transmission": 1.0,
        "line_transmission_status": "not_conditioned",
        "line_transmission_warning_threshold": float(cfg.line_transmission_warning_threshold),
        "filter_weighted_throughput": 1.0, "components": {}, "public_data_components_used": 0,
        "sky_cache_step_nm": float("nan"), "sky_grid_step_nm": float("nan"),
        "sky_cache_undersampled": False,
    }
    if not cfg.use_public_cache:
        prov.add(
            "source_spectra", "synthetic_proxy",
            "synthetic continuum (blackbody/flat) + Gaussian line",
            notes="Absolute scale from AB-mag photon budget (engineering_estimate).",
        )
    else:
        cube, sky_e_spectrum, cond_info = _apply_public_conditioning(
            cube, src_cfg.wavelength_grid_um, src_cfg, cfg, prov, continuum_cube=continuum_cube,
            spectrograph=spectro,
        )
    return _SourceStage(cube=cube, intrinsic_cube=intrinsic_cube, budget_info=budget_info,
                        sky_e_spectrum=sky_e_spectrum, cond_info=cond_info)


def _couple_psf(
    cfg: PipelineConfig,
    cube: np.ndarray,
    src_cfg: SourceCubeConfig,
    wavelength_grid: np.ndarray,
    prov: ProvenanceLog,
) -> tuple[np.ndarray, psf.PSFMetrics | None]:
    """Stage 2: AO PSF coupling, plus encircled-energy / FWHM diagnostics."""
    psf_metrics = None
    coupled = cube
    if cfg.apply_psf:
        psf_builder = partial(
            psf.ao_corrected_psf_proxy,
            strehl=cfg.strehl,
            telescope_diameter_m=cfg.telescope_diameter_m,
            seeing_fwhm_mas=cfg.seeing_fwhm_mas,
        )
        # The uniform emission-line field represents a sub-field of a larger
        # extended scene, so edge-normalise to conserve its flux; isolated point /
        # galaxy sources fit inside the field and use the literal convolution
        # (any flux reaching the FOV edge genuinely leaves). Either way the
        # realised field throughput is reported in the metrics.
        conserve = cfg.source_kind == "emission_line"
        # Field-covering kernel: the PSF builders normalise over the whole plane,
        # so the seeing halo that lands outside the field is genuinely lost and
        # shows up in psf_field_throughput.
        coupled = psf.convolve_cube_with_psf(
            cube, wavelength_grid, psf_builder, cfg.spaxel_scale_mas,
            kernel_n_pix=None, conserve_flux=conserve,
        )
        # Independent fine PSF (at the band centre) for the encircled-energy
        # metrics, on a grid large enough to contain the seeing halo so the EE
        # fractions are relative to (nearly) the total PSF flux.  An odd
        # subdivision aligns centred fine/coarse pixel edges, so the diagnostic
        # and the actual spaxel kernel measure the same area, and it must
        # resolve the narrowest PSF component (the diffraction core when
        # Strehl > 0).
        narrowest_fwhm = (min(psf.diffraction_fwhm_mas(src_cfg_mid(src_cfg), cfg.telescope_diameter_m),
                              cfg.seeing_fwhm_mas)
                          if cfg.strehl > 0 else cfg.seeing_fwhm_mas)
        subdivision = max(5, int(np.ceil(
            PSF_DIAGNOSTIC_SAMPLES_PER_FWHM * cfg.spaxel_scale_mas / narrowest_fwhm)))
        subdivision += 1 - subdivision % 2
        fine_scale = cfg.spaxel_scale_mas / subdivision
        psf_fine = psf.ao_corrected_psf_proxy(
            cfg.strehl, src_cfg_mid(src_cfg), cfg.telescope_diameter_m,
            cfg.seeing_fwhm_mas, fine_scale,
            n_pix=psf.halo_containing_n_pix(cfg.seeing_fwhm_mas, fine_scale),
        )
        psf_metrics = psf.psf_metrics(psf_fine, fine_scale, cfg.spaxel_scale_mas)
        # The FWHM needs much finer sampling than the encircled energies (the
        # radial-profile estimator is biased by ~+7% at 5 samples per FWHM), so
        # take the peak from a small dedicated core grid, and the half-maximum
        # crossing from the halo grid only when it lies outside the core grid
        # (halo-dominated PSF).
        core_scale = narrowest_fwhm / PSF_FWHM_SAMPLES_PER_FWHM
        core_half_width = PSF_FWHM_GRID_HALF_WIDTH_FWHMS * narrowest_fwhm
        psf_core = psf.ao_corrected_psf_proxy(
            cfg.strehl, src_cfg_mid(src_cfg), cfg.telescope_diameter_m, cfg.seeing_fwhm_mas,
            core_scale, n_pix=2 * int(np.ceil(core_half_width / core_scale)) + 1,
        )
        fwhm_core = psf.psf_fwhm_two_grid_mas(psf_core, core_scale, psf_fine, fine_scale)
        if np.isfinite(fwhm_core):
            psf_metrics = replace(
                psf_metrics, fwhm_mas=fwhm_core,
                undersampled=bool(fwhm_core < 2.0 * cfg.spaxel_scale_mas))
    prov.add(
        "ao_psf", "synthetic_proxy",
        "seeing/AO/diffraction proxies",
        notes=f"Strehl-weighted core+halo, Strehl={cfg.strehl}; no real AO residual model.",
    )
    return coupled, psf_metrics


def _forward_model(
    cfg: PipelineConfig,
    coupled: np.ndarray,
    sky_e_spectrum: np.ndarray,
    spectro: SpectrographConfig,
    wavelength_grid: np.ndarray,
    prov: ProvenanceLog,
) -> _ForwardStage:
    """Stages 3-4: image-slicer remapping and spectral dispersion (LSF)."""
    ifu_cfg = IFUConfig(
        nx_spaxel=cfg.nx, ny_spaxel=cfg.ny, spaxel_scale_mas=cfg.spaxel_scale_mas,
        slice_width_spaxels=cfg.slice_width_spaxels,
        cross_talk_sigma_pix=cfg.cross_talk_sigma_pix, slice_gap_pix=cfg.slice_gap_pix,
    )
    mapping = build_slicer_mapping(ifu_cfg, spectro, wavelength_grid)
    # The source detector frame (no sky) -- used for the source-only flux/throughput
    # accounting so the PSF/LSF bookkeeping is not polluted by the sky background.
    detector_source_clean = forward_project_cube(coupled, mapping, apply_lsf=True, apply_cross_talk=True)
    # Sky background is spatially uniform (PSF-invariant); add it to every spaxel
    # AFTER PSF coupling. Its spectral LSF was already applied on the native sky
    # grid, so only the spatial remap + cross-talk remain.
    has_sky = bool(np.any(sky_e_spectrum > 0))
    if has_sky:
        sky_cube = np.broadcast_to(sky_e_spectrum, coupled.shape)
        detector_sky = forward_project_cube(
            sky_cube, mapping, apply_lsf=False, apply_cross_talk=True
        )
        detector_clean = detector_source_clean + detector_sky
    else:
        detector_clean = detector_source_clean
    prov.add(
        "ifu_remapping", "synthetic_proxy",
        "image-slicer proxy (concept: HARMONI arXiv:1807.06313, NIRSpec arXiv:physics/0512002)",
        notes="Rectangular slicer with deterministic trace table; not real slicer geometry.",
    )
    return _ForwardStage(ifu_cfg=ifu_cfg, mapping=mapping, detector_source_clean=detector_source_clean,
                         detector_clean=detector_clean, has_sky=has_sky)


def _read_out_detector(cfg: PipelineConfig, detector_clean: np.ndarray, rng, prov: ProvenanceLog):
    """Stage 5: detector noise, then the known bias/dark/flat calibration."""
    readout = expose(detector_clean, cfg.detector, cfg.exposure_s, rng)
    calibrated = calibrate_readout(
        readout, cfg.detector, cfg.exposure_s, correct_flat=cfg.correct_flat_field
    )
    prov.add(
        "detector_calibration", "engineering_estimate", "known bias/dark and pixel-response calibration",
        notes=("Raw readout retained separately. Bias/dark expectation removed; shot noise remains. "
               + ("Known response divided out with matching variance propagation."
                  if cfg.correct_flat_field else "Response deliberately left uncorrected: residual-flat experiment.")
               + " Bad, saturated and zero-response detector samples remain masked."),
    )
    prov.add(
        "detector_noise", "engineering_estimate",
        "H2RG/H4RG-like: MOSFIRE arXiv:1208.0314, NIRSpec arXiv:2208.12686, Rauscher arXiv:1509.06264",
        notes=(
            f"read_noise={cfg.detector.read_noise_e} e-, dark={cfg.detector.dark_current_e_s_pix} e-/s; "
            "engineering proxies, not real detector cosmetics."
        ),
    )
    return readout, calibrated


def _reconstruct(cfg: PipelineConfig, calibrated, fwd: _ForwardStage, prov: ProvenanceLog) -> _Reduction:
    """Stage 6: both cube reconstructions, observed truth and sky subtraction.

    Each reconstructor is paired with its OWN propagated variance so SNRs are
    internally consistent: nearest flux <-> nearest-look-up variance, deblended
    (joint least-squares) flux <-> diag((A^T W A)^-1).  The banded normal
    matrices are kept so deblended aperture sums can use the exact spatial
    covariance.  ``cfg.science_extraction`` picks the pair the science
    products are built from.
    """
    mapping = fwd.mapping
    detector_variance = calibrated.variance_e2
    deblend = cfg.science_extraction == "deblend"
    rebuilt_nearest = recon.reconstruct_cube_nearest(calibrated.image_e, mapping)
    variance_nearest = recon.estimate_variance_cube(detector_variance, mapping)
    rebuilt_weighted, variance_weighted, extraction_model = recon.reconstruct_cube_joint_with_model(
        calibrated.image_e, detector_variance, mapping
    )
    prov.add(
        "cube_reconstruction", "synthetic_proxy",
        "controlled inverse mapping (concept: MUSE pipeline arXiv:2006.08638, Horne 1986)",
        notes=(
            f"Science products use the {cfg.science_extraction} estimator. Nearest read-out and "
            "joint inverse-variance least-squares deblend of all traces per column (covariance "
            "(A^T W A)^-1 kept in banded form) are both computed, each with its own propagated "
            "variance; not a real pipeline."
        ),
    )

    # Reduction-matched observed truth: the noiseless, sky-free source frame
    # pushed through the SAME science estimator (with the SAME variance weights
    # for the deblend), so the recovery-vs-truth bias isolates noise rather than
    # an operator change.  It is post-PSF/LSF/cross-talk, NOT a deconvolved
    # intrinsic map (see the PipelineResult field docstring).
    if deblend:
        observed_truth_cube = recon.reconstruct_cube_weighted(
            fwd.detector_source_clean, detector_variance, mapping
        )
    else:
        observed_truth_cube = recon.reconstruct_cube_nearest(fwd.detector_source_clean, mapping)

    sky_subtraction_applied = False
    rebuilt_weighted_skysub: np.ndarray | None = None
    rebuilt_nearest_skysub: np.ndarray | None = None
    sky_cube_weighted: np.ndarray | None = None
    science_weighted = rebuilt_weighted
    science_nearest = rebuilt_nearest
    if fwd.has_sky and cfg.subtract_sky:
        # The forward map is linear, so the noiseless sky-only frame is exactly
        # the difference of the two clean frames (same LSF + cross-talk path).
        # Reconstructing it with the same estimators (same variance weights)
        # makes the subtraction remove the sky's expected contribution exactly;
        # the sky's Poisson noise remains in the data and in the variance -- a
        # perfect-calibration assumption, not a real sky reduction.
        detector_sky_clean = fwd.detector_clean - fwd.detector_source_clean
        sky_cube_weighted = recon.reconstruct_cube_weighted(
            detector_sky_clean, detector_variance, mapping
        )
        sky_cube_nearest = recon.reconstruct_cube_nearest(detector_sky_clean, mapping)
        scale = 1.0 + cfg.sky_subtraction_scale_error
        rebuilt_weighted_skysub = rebuilt_weighted - scale * sky_cube_weighted
        rebuilt_nearest_skysub = rebuilt_nearest - scale * sky_cube_nearest
        science_weighted = rebuilt_weighted_skysub
        science_nearest = rebuilt_nearest_skysub
        sky_subtraction_applied = True
        prov.add(
            "sky_subtraction", "engineering_estimate",
            "forward-modelled sky (same LSF/cross-talk/reconstruction path) subtracted",
            notes=(
                ("Perfect known detector calibration; sky scaled by " if cfg.correct_flat_field
                 else "Deliberately uncorrected response leaves sky residuals; optical sky scaled by ")
                + f"(1 + {cfg.sky_subtraction_scale_error}); the sky's photon noise stays in "
                "the data and the variance. NOT a real sky reduction or calibrated ETC."
            ),
        )

    return _Reduction(
        rebuilt_nearest=rebuilt_nearest, variance_nearest=variance_nearest,
        rebuilt_weighted=rebuilt_weighted, variance_weighted=variance_weighted,
        extraction_model=extraction_model, observed_truth_cube=observed_truth_cube,
        science_nearest=science_nearest, science_weighted=science_weighted,
        rebuilt_nearest_skysub=rebuilt_nearest_skysub, rebuilt_weighted_skysub=rebuilt_weighted_skysub,
        sky_cube_weighted=sky_cube_weighted, sky_subtraction_applied=sky_subtraction_applied,
        science_cube=science_weighted if deblend else science_nearest,
        science_variance=variance_weighted if deblend else variance_nearest,
    )


def _science_products(
    cfg: PipelineConfig,
    red: _Reduction,
    spectro: SpectrographConfig,
    wavelength_grid: np.ndarray,
    line_center: float,
    source: _SourceStage,
) -> _ScienceStage:
    """Stage 7: line maps, velocity, SNR, aperture spectrum and recovered R."""
    # Window the line on its core (a few resolution elements): wide enough to
    # capture the line, narrow enough to keep noisy off-line channels out of the
    # flux integral and the velocity first moment.
    window = 3.0 * spectro.delta_lambda_resel_um(line_center)
    science, variance = red.science_cube, red.science_variance
    line_maps = diag.line_window_maps(science, variance, wavelength_grid, line_center, window)
    partial_lf = line_maps["partial_line_flux"]
    raw_vel_map = diag.velocity_centroid_map(science, wavelength_grid, line_center, window)
    # Subtracting the deterministic sky model does not reduce the noise, so the
    # variance cubes apply unchanged to the sky-subtracted cubes.  Both
    # estimators' SNR maps are kept so either can be quoted as the baseline.
    snr = line_maps["snr"]
    other = "deblend" if cfg.science_extraction == "nearest" else "nearest"
    other_cube, other_var = ((red.science_weighted, red.variance_weighted) if other == "deblend"
                             else (red.science_nearest, red.variance_nearest))
    snr_by_method = {
        cfg.science_extraction: snr,
        other: diag.snr_map(other_cube, other_var, wavelength_grid, line_center, window),
    }
    line_measured = line_maps["complete_window"]
    measured_snr = snr[line_measured & np.isfinite(snr)]
    line_available = source.cond_info["line_transmission_status"] not in ("blocked", "no_intrinsic_line")
    moment_detected = line_measured & np.isfinite(snr) & (snr >= cfg.moment_min_snr) & line_available
    vel_map = np.where(moment_detected, raw_vel_map, np.nan)

    # Sky-line (OH) mask REPORTING: which wavelengths a real reduction would
    # typically mask, and how much of the science line window they cover.  The
    # mask is reported, not applied -- masking channels inside the line window
    # would bias the flux integral.
    sky_line_masked_frac = 0.0
    sky_line_masked_in_window_frac = 0.0
    if np.any(source.sky_e_spectrum > 0):
        from .sky_telluric import masked_fraction, science_window_masked_fraction, sky_line_mask

        oh_mask = sky_line_mask(source.sky_e_spectrum, threshold_factor=3.0)
        sky_line_masked_frac = masked_fraction(oh_mask)
        sky_line_masked_in_window_frac = science_window_masked_fraction(
            wavelength_grid, oh_mask, line_center, window
        )

    # Aperture for the 1-D spectrum: spaxels brighter than 20% of the peak line
    # flux, falling back to the central 3x3 if the line is faint/uniform.
    # Selection uses explicitly partial flux, not invented missing-channel data.
    # The resulting aperture is data-selected; errors are conditional on it.
    finite_lf = partial_lf[np.isfinite(partial_lf)]
    peak_lf = float(np.max(finite_lf)) if finite_lf.size else 0.0
    aperture = np.isfinite(partial_lf) & (partial_lf > 0.2 * peak_lf)
    if aperture.sum() < 4:
        aperture = np.zeros_like(partial_lf, dtype=bool)
        cy, cx = cfg.ny // 2, cfg.nx // 2
        aperture[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
    spectrum = recon.extract_1d_spectrum(science, aperture)

    # A numerical moment width is not evidence of a line. Publish it only for
    # a complete, positive-SNR central window; otherwise leave it unavailable.
    cy0, cx0 = cfg.ny // 2, cfg.nx // 2
    recovered_R = (diag.resolving_power_proxy(science[cy0, cx0, :], wavelength_grid, line_center, window)
                   if moment_detected[cy0, cx0] else float("nan"))

    # Robust velocity span over line-bright spaxels (5th-95th percentile), which
    # avoids the wild centroids of near-empty, noise-dominated spaxels.
    vel_bright = vel_map[aperture]
    vel_bright = vel_bright[np.isfinite(vel_bright)]
    if vel_bright.size >= 2:
        velocity_span = float(np.nanpercentile(vel_bright, 95) - np.nanpercentile(vel_bright, 5))
    else:
        velocity_span = float("nan")

    return _ScienceStage(
        window=window, line_maps=line_maps, velocity_map=vel_map, moment_detected=moment_detected,
        measured_snr=measured_snr, snr_by_method=snr_by_method, aperture=aperture, spectrum=spectrum,
        recovered_R=recovered_R, velocity_span=velocity_span,
        sky_line_masked_frac=sky_line_masked_frac,
        sky_line_masked_in_window_frac=sky_line_masked_in_window_frac,
    )


def _collect_metrics(
    cfg: PipelineConfig,
    spectro: SpectrographConfig,
    source: _SourceStage,
    coupled: np.ndarray,
    psf_metrics: psf.PSFMetrics | None,
    fwd: _ForwardStage,
    calibrated,
    red: _Reduction,
    sci: _ScienceStage,
    effective_throughput: float,
) -> dict:
    """Flatten a run into the metrics dict written to the CSV/JSON summaries."""
    cube, intrinsic_cube, cond_info = source.cube, source.intrinsic_cube, source.cond_info
    # Two complementary "how much flux survives" numbers:
    #  * conservation ratio -- a *pure-rearrangement* round trip (no LSF, no
    #    cross-talk) that should return 1.0, proving the map is invertible;
    #  * detected fraction -- the realistic path WITH LSF + cross-talk, where some
    #    light legitimately spills into inter-slice gap rows and band edges.
    rearranged = forward_project_cube(coupled, fwd.mapping, apply_lsf=False, apply_cross_talk=False)
    fc = recon.flux_conservation(coupled, recon.reconstruct_cube_nearest(rearranged, fwd.mapping))
    # Detected fraction and electron budgets are *source-only* (sky excluded);
    # the sky is uniform per spaxel over the field.
    source_electrons = float(np.sum(coupled))
    detected_fraction = (
        float(np.sum(fwd.detector_source_clean) / source_electrons) if source_electrons > 0 else float("nan")
    )
    sky_background_electrons = float(np.sum(source.sky_e_spectrum) * cfg.nx * cfg.ny)
    sky_to_source_ratio = (
        sky_background_electrons / source_electrons if source_electrons > 0 else float("nan")
    )
    line_maps, measured_snr, moment_detected = sci.line_maps, sci.measured_snr, sci.moment_detected
    cy0, cx0 = cfg.ny // 2, cfg.nx // 2

    metrics = {
        "mode": cfg.mode,
        "n_spaxels": int(cfg.nx * cfg.ny),
        "n_wavelength": int(spectro.n_detector_pixels),
        "detector_shape": list(fwd.mapping.det_shape),
        # Electrons entering PSF coupling (AFTER public conditioning) and the
        # fraction that survives PSF coupling within the field, reported
        # explicitly so the PSF field-edge throughput is never hidden; the
        # pre-conditioning budget and the conditioning loss follow.
        "input_source_electrons": float(np.sum(cube)),
        # The AB-magnitude photon budget BEFORE public conditioning (source
        # spectral shape x telluric x filter), and the fraction that survives
        # it; input_source_electrons above is the conditioned cube entering
        # PSF coupling.  Without conditioning the two totals coincide.
        "photon_budget_electrons": float(np.sum(intrinsic_cube)),
        "conditioning_throughput": (
            float(np.sum(cube) / np.sum(intrinsic_cube)) if np.sum(intrinsic_cube) > 0 else float("nan")
        ),
        # Which photon-budget semantics produced those electrons: "total_abmag"
        # rescales the whole cube (line_flux_rel is shape-only), "continuum_line"
        # adds a physically-derived line on top of the continuum budget.
        "photon_budget_mode": source.budget_info["mode"],
        "coupled_electrons": source_electrons,
        "psf_field_throughput": float(np.sum(coupled) / np.sum(cube)) if np.sum(cube) > 0 else float("nan"),
        "flux_conservation_ratio": float(fc["ratio"]),
        "detected_fraction_with_lsf_crosstalk": detected_fraction,
        # Public-conditioning budget (0 / 1.0 defaults when not public-conditioned).
        "filter_weighted_throughput": float(cond_info["filter_weighted_throughput"]),
        "transmission_at_line": float(cond_info["transmission_at_line"]),
        "line_weighted_transmission": float(cond_info["line_weighted_transmission"]),
        "line_transmission_status": cond_info["line_transmission_status"],
        "line_transmission_warning_threshold": float(cond_info["line_transmission_warning_threshold"]),
        "detector_calibration": "known_bias_dark_flat" if cfg.correct_flat_field else "bias_dark_only_residual_flat",
        "detector_invalid_fraction": float(np.mean(calibrated.invalid_pixel_mask)),
        "source_electrons": source_electrons,
        "sky_background_electrons": sky_background_electrons,
        "sky_to_source_ratio": sky_to_source_ratio,
        # Reduction state: whether the diagnostics above were computed on the
        # sky-subtracted cube, with what deliberate model-scale error, and how
        # much of the band / line window a typical OH-line mask would cover.
        "sky_subtraction_applied": bool(red.sky_subtraction_applied),
        "sky_subtraction_scale_error": float(cfg.sky_subtraction_scale_error),
        "sky_line_masked_fraction": float(sci.sky_line_masked_frac),
        "sky_line_masked_in_window_fraction": float(sci.sky_line_masked_in_window_frac),
        # Whether those mask fractions can be trusted at this resolution: the
        # sky cache must sample at least as finely as the science grid.
        "sky_cache_step_nm": float(cond_info["sky_cache_step_nm"]),
        "sky_grid_step_nm": float(cond_info["sky_grid_step_nm"]),
        "sky_cache_undersampled": bool(cond_info["sky_cache_undersampled"]),
        "public_data_components_used": int(cond_info["public_data_components_used"]),
        # Headline SNR from the science estimator's flux + matching variance;
        # both estimators' SNRs are reported alongside as baselines.
        # Full-window summaries exclude unavailable values, never encode them
        # as physical zero SNR. Partial-window measurements have separate names.
        "science_extraction": cfg.science_extraction,
        "median_snr": float(np.median(measured_snr)) if measured_snr.size else float("nan"),
        "median_snr_measured_spaxels": float(np.median(measured_snr)) if measured_snr.size else float("nan"),
        "median_snr_partial_window": _finite_stat(line_maps["partial_snr"], np.median),
        "mean_line_window_channel_coverage": float(np.mean(line_maps["measured_channel_fraction"])),
        "line_window_measured_spaxel_fraction": float(np.mean(line_maps["complete_window"])),
        "unmeasured_voxel_fraction": float(np.mean(~np.isfinite(red.science_variance))),
        "max_snr": _finite_stat(line_maps["snr"], np.max),
        "median_snr_nearest": _finite_stat(sci.snr_by_method["nearest"], np.median),
        "max_snr_nearest": _finite_stat(sci.snr_by_method["nearest"], np.max),
        "median_snr_deblend": _finite_stat(sci.snr_by_method["deblend"], np.median),
        "max_snr_deblend": _finite_stat(sci.snr_by_method["deblend"], np.max),
        "central_line_moment_detected": bool(moment_detected[cy0, cx0]),
        "line_moment_detected_spaxel_fraction": float(np.mean(moment_detected)),
        "moment_min_snr": float(cfg.moment_min_snr),
        "recovered_R_bright_spaxel": float(sci.recovered_R),
        "configured_R": float(cfg.resolving_power_R),
        "velocity_span_p5_p95_kms": sci.velocity_span,
        "well_sampled_spectrum": bool(spectro.is_well_sampled()),
        "pixels_per_resel": float(spectro.actual_pixels_per_resolution_element()),
    }
    if psf_metrics is not None:
        metrics.update(
            psf_fwhm_mas=float(psf_metrics.fwhm_mas),
            psf_ee1_spaxel=float(psf_metrics.ee1_spaxel),
            psf_ee3x3_spaxel=float(psf_metrics.ee3x3_spaxel),
            psf_undersampled=bool(psf_metrics.undersampled),
        )
    # Report both the configured scalar and the budget-resolved throughput, but
    # only when the budget path is enabled (keeps default runs/metrics unchanged).
    if cfg.use_throughput_budget:
        metrics["configured_throughput_scalar"] = float(cfg.throughput_scalar)
        metrics["effective_throughput_used"] = float(effective_throughput)
    # The continuum/line electron split only exists in continuum_line mode.
    if source.budget_info["mode"] == "continuum_line":
        for key in ("continuum_total_electrons", "line_total_electrons", "line_in_band_electrons"):
            metrics[key] = source.budget_info[key]
    return metrics


def _finite_stat(values, statistic) -> float:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(statistic(finite)) if finite.size else float("nan")


# Minimum fine-grid samples across the narrowest PSF FWHM for the encircled-
# energy diagnostics, and the (finer) sampling and half-width of the dedicated
# grid the reported FWHM is measured on (converged to ~0.2% at 40 samples).
PSF_DIAGNOSTIC_SAMPLES_PER_FWHM = 5.0
PSF_FWHM_SAMPLES_PER_FWHM = 40.0
PSF_FWHM_GRID_HALF_WIDTH_FWHMS = 4.0


def src_cfg_mid(src_cfg: SourceCubeConfig) -> float:
    """Band-centre wavelength of a source config (micrometres)."""
    return 0.5 * (src_cfg.lambda_min_um + src_cfg.lambda_max_um)


# Provenance class for each pipeline metric, so the output metrics table can
# carry a per-row label (the project requires every output table to do so).
# Self-consistency numbers are ``internal_validation``; numbers that come out of
# the synthetic forward/inverse model are ``synthetic_proxy``; configuration
# anchors are ``literature_derived``; the photon-budget total is an
# ``engineering_estimate``.
_METRIC_PROVENANCE: dict[str, str] = {
    "mode": "internal_validation",
    "n_spaxels": "literature_derived",
    "n_wavelength": "literature_derived",
    "detector_shape": "synthetic_proxy",
    "input_source_electrons": "engineering_estimate",
    "photon_budget_mode": "internal_validation",
    "continuum_total_electrons": "engineering_estimate",
    "line_total_electrons": "engineering_estimate",
    "line_in_band_electrons": "engineering_estimate",
    "coupled_electrons": "synthetic_proxy",
    "psf_field_throughput": "synthetic_proxy",
    "flux_conservation_ratio": "internal_validation",
    "detected_fraction_with_lsf_crosstalk": "internal_validation",
    # Public-conditioning budget metrics (the underlying sky/source/filter data
    # provenance is in the provenance log; these derived budget numbers are
    # engineering estimates, and the components count is a self-describing meta).
    "filter_weighted_throughput": "engineering_estimate",
    "source_electrons": "engineering_estimate",
    "sky_background_electrons": "engineering_estimate",
    "sky_to_source_ratio": "engineering_estimate",
    "sky_subtraction_applied": "internal_validation",
    "sky_subtraction_scale_error": "engineering_estimate",
    "sky_line_masked_fraction": "engineering_estimate",
    "sky_line_masked_in_window_fraction": "engineering_estimate",
    "public_data_components_used": "internal_validation",
    "configured_throughput_scalar": "engineering_estimate",
    "effective_throughput_used": "engineering_estimate",
    "median_snr": "synthetic_proxy",
    "median_snr_measured_spaxels": "synthetic_proxy",
    "median_snr_partial_window": "synthetic_proxy",
    "mean_line_window_channel_coverage": "internal_validation",
    "transmission_at_line": "engineering_estimate",
    "line_weighted_transmission": "engineering_estimate",
    "line_transmission_status": "internal_validation",
    "line_transmission_warning_threshold": "engineering_estimate",
    "central_line_moment_detected": "internal_validation",
    "line_moment_detected_spaxel_fraction": "internal_validation",
    "moment_min_snr": "engineering_estimate",
    "detector_calibration": "internal_validation",
    "detector_invalid_fraction": "internal_validation",
    "line_window_measured_spaxel_fraction": "internal_validation",
    "unmeasured_voxel_fraction": "internal_validation",
    "max_snr": "synthetic_proxy",
    "science_extraction": "internal_validation",
    "median_snr_nearest": "synthetic_proxy",
    "max_snr_nearest": "synthetic_proxy",
    "median_snr_deblend": "synthetic_proxy",
    "max_snr_deblend": "synthetic_proxy",
    "recovered_R_bright_spaxel": "synthetic_proxy",
    "configured_R": "literature_derived",
    "velocity_span_p5_p95_kms": "synthetic_proxy",
    "well_sampled_spectrum": "internal_validation",
    "pixels_per_resel": "internal_validation",
    "psf_fwhm_mas": "synthetic_proxy",
    "psf_ee1_spaxel": "synthetic_proxy",
    "psf_ee3x3_spaxel": "synthetic_proxy",
    "psf_undersampled": "internal_validation",
}


def metric_provenance(metric_name: str) -> str:
    """Provenance class for a metric name (default ``synthetic_proxy``)."""
    return _METRIC_PROVENANCE.get(metric_name, "synthetic_proxy")


# ----------------------------------------------------------------------------
# Artifact writing (figures, CSV metrics, JSON summary).
# ----------------------------------------------------------------------------

def write_artifacts(
    result: PipelineResult,
    outdir: str | Path,
    prefix: str = "run",
) -> dict[str, Path]:
    """Write figures, a CSV metrics table and a JSON run summary.

    Returns a dict mapping artifact name -> path.  Uses the Agg backend so it
    runs headless (CI, scripts) without a display.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: render to files, never to a screen
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- Figure 1: the detector frame (the raw thing a real array would read). ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = axes[0].imshow(result.detector_clean, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Detector frame (noiseless)")
    axes[0].set_xlabel("dispersion / wavelength  ->  det_x")
    axes[0].set_ylabel("cross-dispersion (slices)  ->  det_y")
    fig.colorbar(im0, ax=axes[0], label="electrons")
    im1 = axes[1].imshow(result.detector_noisy, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Detector frame (with noise)")
    axes[1].set_xlabel("dispersion / wavelength  ->  det_x")
    axes[1].set_ylabel("cross-dispersion (slices)  ->  det_y")
    fig.colorbar(im1, ax=axes[1], label="electrons")
    fig.tight_layout()
    p = outdir / f"{prefix}_detector_frame.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths["detector_frame"] = p

    # --- Figure 2: reconstructed-cube science diagnostics. ---
    # Blank spaxels have a bad pixel inside the line window: they are reported
    # as unmeasured rather than interpolated (coverage is in the suptitle).
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    m00 = axes[0, 0].imshow(result.line_flux_map, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Line flux [e-]")
    fig.colorbar(m00, ax=axes[0, 0])
    vmax = _finite_stat(np.abs(result.velocity_map), np.max)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    m01 = axes[0, 1].imshow(
        result.velocity_map, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    axes[0, 1].set_title(f"Line-of-sight velocity [km/s] (SNR >= {result.config.moment_min_snr:g})")
    fig.colorbar(m01, ax=axes[0, 1])
    m10 = axes[1, 0].imshow(result.snr_map, origin="lower", cmap="cividis")
    axes[1, 0].set_title("Line SNR")
    fig.colorbar(m10, ax=axes[1, 0])
    axes[1, 1].plot(result.wavelength_grid_um, result.extracted_spectrum, lw=1.2)
    axes[1, 1].axvline(result.line_center_um, color="r", ls="--", lw=0.8, label="line centre")
    axes[1, 1].set_title("Aperture spectrum")
    axes[1, 1].set_xlabel("wavelength [um]")
    axes[1, 1].set_ylabel("flux [e-]")
    axes[1, 1].legend(fontsize=8)
    coverage = result.metrics["line_window_measured_spaxel_fraction"]
    fig.suptitle(
        f"Recovered cube ({result.config.mode}, {result.science_extraction} extraction): "
        f"{coverage:.0%} of spaxels measured; blank = bad pixel in the line window"
    )
    fig.tight_layout()
    p = outdir / f"{prefix}_reconstructed_diagnostics.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths["reconstructed_diagnostics"] = p

    # --- CSV: flat metrics table (one metric per row), each with a provenance
    #     label so the output table is self-describing (a project requirement). ---
    metrics_rows = [
        {"metric": k, "value": v, "source_class": metric_provenance(k)}
        for k, v in result.metrics.items()
    ]
    csv_path = outdir / f"{prefix}_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
    paths["metrics_csv"] = csv_path

    # --- JSON: metrics + provenance + a short config summary. ---
    # Use the shared helper so the summary carries the same author / version /
    # non-overclaiming note as every other artifact in the project.
    from . import reporting as rep

    # The complete nested configuration and the input-cache hashes make a saved
    # custom observation reproducible (seed, exposure, kinematics, detector...).
    json_path = rep.write_summary_json(outdir / f"{prefix}_summary.json", {
        "mode": result.config.mode,
        "configuration": result.config,
        "input_caches": rep.input_cache_identity(result.provenance.to_records()),
        "metrics": result.metrics,
        "provenance": result.provenance.to_records(),
        "provenance_counts": result.provenance.summary_counts(),
    })
    paths["summary_json"] = json_path

    return paths
