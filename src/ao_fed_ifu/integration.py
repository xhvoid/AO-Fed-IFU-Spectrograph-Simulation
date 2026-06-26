"""End-to-end pipeline orchestration and the three run modes.

This module wires the per-stage modules into one runnable pipeline:

    source_cube -> psf_coupling -> ifu_mapping -> spectrograph (LSF)
    -> detector_model -> cube_reconstruction -> diagnostics

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

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import cube_reconstruction as recon
from . import diagnostics as diag
from . import psf_coupling as psf
from .data_sources import (
    ProvenanceLog,
    load_filter_curve,
    load_sky_spectrum,
    load_source_spectrum,
)
from .detector_model import DetectorConfig, expose
from .ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
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

# Observed-wavelength range (um) in which a high-z line can be placed: the overlap
# of the cached public data (SVO Ks, ESO SkyCalc, CALSPEC), with a small margin so
# a window fits around the line. Used by make_highz_galaxy_config to validate.
SUPPORTED_HIGHZ_OBS_UM = (2.00, 2.40)


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
    """Like instrument_inspired but also conditions on optional cached public data."""
    cfg = make_instrument_inspired_config(mode="public_conditioned", use_public_cache=True)
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
            f"is {obs:.4f} um, outside the supported public-conditioned band "
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

    source_cube: np.ndarray            # truth cube (electrons)
    coupled_cube: np.ndarray           # after PSF coupling
    detector_clean: np.ndarray         # noiseless detector frame
    detector_noisy: np.ndarray         # noisy detector frame
    variance_frame: np.ndarray         # detector variance frame
    bad_pixel_mask: np.ndarray

    rebuilt_nearest: np.ndarray
    rebuilt_weighted: np.ndarray
    variance_cube: np.ndarray

    line_flux_map: np.ndarray
    velocity_map: np.ndarray
    snr_map: np.ndarray
    extracted_spectrum: np.ndarray
    aperture_mask: np.ndarray

    psf_metrics: psf.PSFMetrics | None
    metrics: dict
    provenance: ProvenanceLog

    # Public-conditioning products (zeros / empty when not public-conditioned).
    sky_background_spectrum: np.ndarray | None = None
    conditioning_info: dict | None = None


def _build_source_cube(cfg: PipelineConfig, src_cfg: SourceCubeConfig, line_center_um: float) -> np.ndarray:
    """Dispatch to the requested source-cube builder."""
    if cfg.source_kind == "emission_line":
        return create_emission_line_cube(
            src_cfg, line_center_um=line_center_um, line_flux_rel=cfg.line_flux_rel,
            line_sigma_um=cfg.line_sigma_um,
            velocity_gradient_kms_per_spaxel=cfg.velocity_gradient_kms_per_spaxel,
        )
    if cfg.source_kind == "point":
        return create_point_source_cube(
            src_cfg, line_center_um=line_center_um, line_flux_rel=cfg.line_flux_rel,
            line_sigma_um=cfg.line_sigma_um,
        )
    if cfg.source_kind == "galaxy":
        return create_extended_galaxy_cube(
            src_cfg, line_center_um=line_center_um, line_flux_rel=cfg.line_flux_rel,
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
       ``radiance * area * exposure * throughput * filter * dlambda * Omega_spaxel``
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

    # Source continuum shape (mean-normalised: reshape, don't rescale the budget).
    shape = source_spec["flux"]
    shape = shape / np.mean(shape) if np.mean(shape) > 0 else np.ones_like(shape)

    conditioned = cube * shape[None, None, :] * telluric[None, None, :] * filt_t[None, None, :]

    # --- Sky background electrons per spaxel per wavelength bin. ---
    radiance = np.clip(np.asarray(sky["radiance"], dtype=float), 0.0, None)
    if sky_prov.source_class != "direct_public_data":
        # The synthetic proxy radiance has arbitrary units; rescale its mean to a
        # SkyCalc-like physical level so the demo still shows a sky contribution.
        mean_r = float(np.mean(radiance))
        radiance = radiance / mean_r * SYNTHETIC_SKY_RADIANCE_PH if mean_r > 0 else np.full_like(radiance, SYNTHETIC_SKY_RADIANCE_PH)
    spaxel_solid_angle_arcsec2 = (cfg.spaxel_scale_mas / 1000.0) ** 2
    sky_e = (
        radiance
        * src_cfg.collecting_area_m2
        * cfg.exposure_s
        * cfg.throughput_scalar
        * filt_t
        * src_cfg.delta_lambda_um
        * spaxel_solid_angle_arcsec2
        * cfg.sky_scaling
    )

    # Effective in-band throughput = source-weighted mean filter transmission.
    src_weight = np.clip(cube.sum(axis=(0, 1)), 0.0, None)
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
        "ESO SkyCalc radiance -> electrons (area*exposure*throughput*filter*dlambda*spaxel_solid_angle)",
        notes=(
            "Physically-motivated unit conversion; ABSOLUTE level is an engineering estimate "
            f"(sky_scaling={cfg.sky_scaling}), NOT a calibrated ETC."
        ),
    )

    info = {
        "filter_weighted_throughput": filter_weighted_throughput,
        "components": components,
        "public_data_components_used": n_public,
    }
    return conditioned, sky_e, info


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """Run the full source -> detector -> reconstruction -> diagnostics pipeline."""
    rng = np.random.default_rng(cfg.seed)
    prov = ProvenanceLog()

    # --- Spectrograph first: it fixes the number of detector columns, and we
    #     match the cube's wavelength sampling to it for clean invertibility. ---
    spectro = SpectrographConfig(
        lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um,
        resolving_power_R=cfg.resolving_power_R,
        pixels_per_resolution_element=cfg.pixels_per_resolution_element,
    )
    nlambda = spectro.n_detector_pixels
    prov.add(
        "wavelength_range_R", "literature_derived",
        "HARMONI overview arXiv:2103.11215; MUSE arXiv:astro-ph/0606329",
        notes=f"{cfg.lambda_min_um}-{cfg.lambda_max_um} um, R={cfg.resolving_power_R}; mode presets only.",
    )

    src_cfg = SourceCubeConfig(
        nx=cfg.nx, ny=cfg.ny, nlambda=nlambda,
        spaxel_scale_mas=cfg.spaxel_scale_mas,
        lambda_min_um=cfg.lambda_min_um, lambda_max_um=cfg.lambda_max_um,
        exposure_s=cfg.exposure_s, telescope_diameter_m=cfg.telescope_diameter_m,
        throughput_scalar=cfg.throughput_scalar,
    )
    wavelength_grid = src_cfg.wavelength_grid_um
    line_center = cfg.line_center_um if cfg.line_center_um is not None else src_cfg_mid(src_cfg)

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

    # --- 1. Source datacube and photon budget. ---
    cube = _build_source_cube(cfg, src_cfg, line_center)
    electrons_per_bin = photon_budget_from_abmag(src_cfg, cfg.ab_magnitude)
    total_electrons = max(electrons_per_bin * nlambda, 1.0)
    cube = scale_cube_to_total_electrons(cube, total_electrons)
    # The base cube's continuum + line is always synthetic; absolute scaling comes
    # from the AB-magnitude photon budget (an engineering estimate).  When public
    # conditioning is requested, _apply_public_conditioning adds the *actual*
    # provenance of the sky/source/filter data it used (public if cached, else proxy).
    # Sky background (per spaxel per wavelength bin) and public-conditioning info;
    # default to "no sky, no public data" for the toy / instrument_inspired modes.
    sky_e_spectrum = np.zeros(nlambda, dtype=float)
    cond_info = {"filter_weighted_throughput": 1.0, "components": {}, "public_data_components_used": 0}
    if not cfg.use_public_cache:
        prov.add(
            "source_spectra", "synthetic_proxy",
            "synthetic continuum (blackbody/flat) + Gaussian line",
            notes="Absolute scale from AB-mag photon budget (engineering_estimate).",
        )
    else:
        cube, sky_e_spectrum, cond_info = _apply_public_conditioning(
            cube, wavelength_grid, src_cfg, cfg, prov
        )

    # --- 2. AO PSF coupling. ---
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
        # realised field throughput is reported in the metrics below.
        conserve = cfg.source_kind == "emission_line"
        coupled = psf.convolve_cube_with_psf(
            cube, wavelength_grid, psf_builder, cfg.spaxel_scale_mas,
            kernel_n_pix=9, conserve_flux=conserve,
        )
        # Independent fine PSF (at the band centre) for the encircled-energy metrics.
        fine_scale = cfg.spaxel_scale_mas / 4.0
        psf_fine = psf.ao_corrected_psf_proxy(
            cfg.strehl, src_cfg_mid(src_cfg), cfg.telescope_diameter_m,
            cfg.seeing_fwhm_mas, fine_scale, n_pix=161,
        )
        psf_metrics = psf.psf_metrics(psf_fine, fine_scale, cfg.spaxel_scale_mas)
    prov.add(
        "ao_psf", "synthetic_proxy",
        "seeing/AO/diffraction proxies",
        notes=f"Strehl-weighted core+halo, Strehl={cfg.strehl}; no real AO residual model.",
    )

    # --- 3. IFU image-slicer remapping + 4. spectral dispersion (LSF). ---
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
    # AFTER PSF coupling, then disperse so it shares the source's noise path.
    has_sky = bool(np.any(sky_e_spectrum > 0))
    if has_sky:
        coupled_plus_sky = coupled + sky_e_spectrum[None, None, :]
        detector_clean = forward_project_cube(
            coupled_plus_sky, mapping, apply_lsf=True, apply_cross_talk=True
        )
    else:
        detector_clean = detector_source_clean
    prov.add(
        "ifu_remapping", "synthetic_proxy",
        "image-slicer proxy (concept: HARMONI arXiv:1807.06313, NIRSpec arXiv:physics/0512002)",
        notes="Rectangular slicer with deterministic trace table; not real slicer geometry.",
    )

    # --- 5. Detector noise. ---
    readout = expose(detector_clean, cfg.detector, cfg.exposure_s, rng)
    prov.add(
        "detector_noise", "engineering_estimate",
        "H2RG/H4RG-like: MOSFIRE arXiv:1208.0314, NIRSpec arXiv:2208.12686, Rauscher arXiv:1509.06264",
        notes=(
            f"read_noise={cfg.detector.read_noise_e} e-, dark={cfg.detector.dark_current_e_s_pix} e-/s; "
            "engineering proxies, not real detector cosmetics."
        ),
    )

    # --- 6. Cube reconstruction. ---
    rebuilt_nearest = recon.reconstruct_cube_nearest(readout.image_e, mapping)
    rebuilt_weighted = recon.reconstruct_cube_weighted(
        readout.image_e, readout.variance_e2, mapping
    )
    variance_cube = recon.estimate_variance_cube(readout.variance_e2, mapping)
    prov.add(
        "cube_reconstruction", "synthetic_proxy",
        "controlled inverse mapping (concept: MUSE pipeline arXiv:2006.08638, Horne 1986)",
        notes="Nearest + inverse-variance (Horne-like) extraction; not a real pipeline.",
    )

    # Two complementary "how much flux survives" numbers:
    #  * conservation ratio -- a *pure-rearrangement* round trip (no LSF, no
    #    cross-talk) that should return 1.0, proving the map is invertible;
    #  * detected fraction -- the realistic path WITH LSF + cross-talk, where some
    #    light legitimately spills into inter-slice gap rows and band edges.
    rearranged = forward_project_cube(coupled, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt_rearranged = recon.reconstruct_cube_nearest(rearranged, mapping)
    fc = recon.flux_conservation(coupled, rebuilt_rearranged)
    # Detected fraction is a *source-only* quantity (exclude the sky background).
    detected_fraction = (
        float(np.sum(detector_source_clean) / np.sum(coupled)) if np.sum(coupled) > 0 else float("nan")
    )

    # Source vs sky electron budget (sky is uniform per spaxel over the field).
    source_electrons = float(np.sum(coupled))
    sky_background_electrons = float(np.sum(sky_e_spectrum) * cfg.nx * cfg.ny)
    sky_to_source_ratio = (
        sky_background_electrons / source_electrons if source_electrons > 0 else float("nan")
    )

    # --- 7. Diagnostics. ---
    # Window the line on its core (a few resolution elements): wide enough to
    # capture the line, narrow enough to keep noisy off-line channels out of the
    # flux integral and the velocity first moment.
    window = 3.0 * spectro.delta_lambda_resel_um(line_center)
    lf_map = diag.line_flux_map(rebuilt_weighted, wavelength_grid, line_center, window)
    vel_map = diag.velocity_centroid_map(rebuilt_weighted, wavelength_grid, line_center, window)
    snr = diag.snr_map(rebuilt_weighted, variance_cube, wavelength_grid, line_center, window)

    # Aperture for the 1-D spectrum: spaxels brighter than 20% of the peak line
    # flux, falling back to the central 3x3 if the line is faint/uniform.
    aperture = lf_map > 0.2 * np.nanmax(lf_map)
    if aperture.sum() < 4:
        aperture = np.zeros_like(lf_map, dtype=bool)
        cy, cx = cfg.ny // 2, cfg.nx // 2
        aperture[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
    spectrum = recon.extract_1d_spectrum(rebuilt_weighted, aperture)

    # Recovered resolving power is measured on the central spaxel: it carries a
    # real, high-SNR line at (near) zero velocity, so the width reflects the
    # instrument LSF (plus intrinsic line width) and is neither broadened by the
    # velocity gradient across an aperture nor biased by picking a noise spike.
    cy0, cx0 = cfg.ny // 2, cfg.nx // 2
    central_spectrum = rebuilt_weighted[cy0, cx0, :]
    r_window = 3.0 * spectro.delta_lambda_resel_um(line_center)
    recovered_R = diag.resolving_power_proxy(central_spectrum, wavelength_grid, line_center, r_window)

    # Robust velocity span over line-bright spaxels (5th-95th percentile), which
    # avoids the wild centroids of near-empty, noise-dominated spaxels.
    vel_bright = vel_map[aperture]
    vel_bright = vel_bright[np.isfinite(vel_bright)]
    if vel_bright.size >= 2:
        velocity_span = float(np.nanpercentile(vel_bright, 95) - np.nanpercentile(vel_bright, 5))
    else:
        velocity_span = float("nan")

    metrics = {
        "mode": cfg.mode,
        "n_spaxels": int(cfg.nx * cfg.ny),
        "n_wavelength": int(nlambda),
        "detector_shape": list(mapping.det_shape),
        # Electrons *before* PSF coupling (the AB-mag photon budget total) and
        # the fraction that survives PSF coupling within the field, reported
        # explicitly so the PSF field-edge throughput is never hidden.
        "input_source_electrons": float(np.sum(cube)),
        "coupled_electrons": float(np.sum(coupled)),
        "psf_field_throughput": float(np.sum(coupled) / np.sum(cube)) if np.sum(cube) > 0 else float("nan"),
        "flux_conservation_ratio": float(fc["ratio"]),
        "detected_fraction_with_lsf_crosstalk": detected_fraction,
        # Public-conditioning budget (0 / 1.0 defaults when not public-conditioned).
        "filter_weighted_throughput": float(cond_info["filter_weighted_throughput"]),
        "source_electrons": source_electrons,
        "sky_background_electrons": sky_background_electrons,
        "sky_to_source_ratio": sky_to_source_ratio,
        "public_data_components_used": int(cond_info["public_data_components_used"]),
        "median_snr": float(np.nanmedian(snr)),
        "max_snr": float(np.nanmax(snr)),
        "recovered_R_bright_spaxel": float(recovered_R),
        "configured_R": float(cfg.resolving_power_R),
        "velocity_span_p5_p95_kms": velocity_span,
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

    return PipelineResult(
        config=cfg, source_config=src_cfg, spectrograph=spectro, ifu_config=ifu_cfg,
        wavelength_grid_um=wavelength_grid, line_center_um=line_center,
        source_cube=cube, coupled_cube=coupled,
        detector_clean=detector_clean, detector_noisy=readout.image_e,
        variance_frame=readout.variance_e2, bad_pixel_mask=readout.bad_pixel_mask,
        rebuilt_nearest=rebuilt_nearest, rebuilt_weighted=rebuilt_weighted,
        variance_cube=variance_cube,
        line_flux_map=lf_map, velocity_map=vel_map, snr_map=snr,
        extracted_spectrum=spectrum, aperture_mask=aperture,
        psf_metrics=psf_metrics, metrics=metrics, provenance=prov,
        sky_background_spectrum=sky_e_spectrum, conditioning_info=cond_info,
    )


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
    "public_data_components_used": "internal_validation",
    "median_snr": "synthetic_proxy",
    "max_snr": "synthetic_proxy",
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    m00 = axes[0, 0].imshow(result.line_flux_map, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Line flux map [e-]")
    fig.colorbar(m00, ax=axes[0, 0])
    vmax = np.nanmax(np.abs(result.velocity_map))
    m01 = axes[0, 1].imshow(
        result.velocity_map, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    axes[0, 1].set_title("Velocity centroid map [km/s]")
    fig.colorbar(m01, ax=axes[0, 1])
    m10 = axes[1, 0].imshow(result.snr_map, origin="lower", cmap="cividis")
    axes[1, 0].set_title("SNR map")
    fig.colorbar(m10, ax=axes[1, 0])
    axes[1, 1].plot(result.wavelength_grid_um, result.extracted_spectrum, lw=1.2)
    axes[1, 1].axvline(result.line_center_um, color="r", ls="--", lw=0.8, label="line centre")
    axes[1, 1].set_title("Extracted 1-D spectrum (aperture)")
    axes[1, 1].set_xlabel("wavelength [um]")
    axes[1, 1].set_ylabel("flux [e-]")
    axes[1, 1].legend(fontsize=8)
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

    json_path = rep.write_summary_json(outdir / f"{prefix}_summary.json", {
        "mode": result.config.mode,
        "metrics": result.metrics,
        "provenance": result.provenance.to_records(),
        "provenance_counts": result.provenance.summary_counts(),
    })
    paths["summary_json"] = json_path

    return paths
