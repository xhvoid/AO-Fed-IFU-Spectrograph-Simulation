# Validation

> **Scope of validation.** This project is validated **only by internal sanity
> checks** against its own physics. It is **not** validated against real
> instrument data, real IFU calibration files, real detector cosmetics, or real
> observatory throughput. Every check below is labelled `internal_validation`.

The checks confirm that the model behaves the way first-principles physics
requires — that quantities move in the right direction, with the right scaling,
when a single knob is changed. They do **not** establish that any number matches
a real instrument.

Run them with:

```bash
python examples/run_validation_checks.py
# writes data/reference_metrics/validation_results.csv
```

or from Python:

```python
from ao_fed_ifu.validation import run_all_validations
for r in run_all_validations():
    print(r.passed, r.name, r.detail)
```

## The seven checks

| # | Check | What it asserts | Why it must hold |
|---|-------|-----------------|------------------|
| 1 | `flux_conservation` | A noiseless forward-projection + nearest reconstruction returns `reconstructed/input = 1`. | The slicer map is a pure rearrangement of electrons; nothing should be created or lost. |
| 2 | `snr_sqrt_exposure` | In the photon-limited regime, `d log(SNR)/d log(t) ≈ 0.5`. | Photon (shot) noise scales as √signal, so SNR ∝ √t when signal ∝ t. |
| 3 | `resolving_power_scaling` | A larger configured `R` yields a narrower recovered line (higher recovered `R`). | `delta_lambda = lambda/R`: higher R ⇒ smaller resolution element ⇒ narrower LSF. |
| 4 | `read_noise_reduces_snr` | More read noise lowers SNR for a faint source. | Read noise adds in quadrature to the variance; for a faint source it dominates. |
| 5 | `crosstalk_increases_leakage` | More cross-talk `sigma` ⇒ more flux leaks out of an isolated spaxel. | A wider cross-dispersion blur spreads more signal into neighbouring traces. |
| 6 | `wavelength_offset_velocity_bias` | A wavelength offset `d_lambda` biases velocity by `delta_v = c · d_lambda / lambda`. | A zero-point calibration error maps directly to a Doppler velocity error. |
| 7 | `ao_psf_improves_ee` | Higher Strehl ⇒ higher central / 3×3-spaxel encircled energy. | AO correction concentrates light from the seeing halo into the diffraction core. |

## Physics / engineering upgrades — labelled estimates, not calibrations

* **Throughput budget** (`throughput_budget.py`) decomposes the throughput into
  atmosphere / telescope / AO / IFU / spectrograph / filter / QE stages. The
  atmosphere and filter are cached public data; the instrument stages are
  **engineering-estimate / synthetic-proxy levels**, not measured curves. The
  combined throughput is a decomposed engineering estimate, **not** a calibrated
  throughput or an ETC. The optional pipeline path (`use_throughput_budget=True`)
  only swaps the flat scalar for the budget-resolved instrument throughput.
* **Wavelength-dependent AO PSF** (`ao_psf_physics.py`) grows the diffraction core
  as `lambda/D` and raises the Strehl toward the red via the Maréchal law for a
  fixed residual wavefront error. This is a wavelength-dependent AO **proxy**
  (`synthetic_proxy` / `engineering_estimate`), **not** AO telemetry, a measured
  Strehl budget, or a phase-screen simulation. The encircled-energy and
  undersampling trends (EE improves with Strehl; core FWHM scales with `lambda/D`)
  are the meaningful outputs, not absolute coupling predictions.
* **Observing strategy** (`observing_strategy.py`) demonstrates dither + sky
  subtraction + cosmic-ray rejection + inverse-variance coadd. The verified trends
  (SNR ~ √N, coadd variance ~ 1/N, cosmic rays rejected at ≈ their injected rate,
  dither-align recovers the source centroid) are internal trend checks; the
  cosmic-ray injection/rejection is a `synthetic_proxy` and the sky model is the
  pipeline proxy. This is **not** a real scheduler or reduction pipeline, and only
  integer dithering is modelled.
* **Detector systematics** (`detector_systematics.py`) add up-the-ramp/Fowler
  sampling, nonlinearity, correlated read noise (+ reference-pixel removal) and
  IPC. The sampling and nonlinearity formulas are `engineering_estimate`; the
  correlated-noise and IPC kernels are `synthetic_proxy`. **None are
  data-calibrated** — there is no measured device model, persistence calibration,
  or real array cosmetics. The trends (more reads → lower read noise; high-count
  pixels compress more; reference-pixel removal lowers the row common-mode; IPC
  conserves flux) are the meaningful outputs.
* **Spectrograph / IFU geometry** (`geometry_systematics.py`) adds parabolic trace
  curvature, a polynomial wavelength-solution residual, a slice-dependent LSF and a
  flexure offset. These are **illustrative** `synthetic_proxy` geometry, **not** an
  as-built slicer or optical model. Zero-amplitude defaults reproduce the nominal
  invertible map exactly (verified); the demonstrated trends (curvature stays in
  bounds and round-trips when corrected but biases flux when uncorrected; the
  wavelength residual maps to a velocity residual; a wider slice LSF lowers the
  recovered R; flexure biases line flux/velocity) are internal trend checks.
* **Sky / telluric variability** (`sky_telluric.py`) adds continuum/OH-line
  scaling, a telluric airmass/water-vapour mismatch, and sky-line masking. The
  SkyCalc radiance/transmission are public data; the variability and airmass
  scalings are `engineering_estimate`. This is a **controlled residual
  experiment, not real telluric reduction**. The trends (OH scaling raises the
  residual near sky lines; mismatch residual is worst in low-transmission troughs;
  a lower mask threshold masks more; a perfect match leaves ~0 residual) are
  internal trend checks. Sky variability context: Jones et al. (arXiv:1310.7030).
* **High-z galaxy source** (`galaxy_kinematics.py`) is a **synthetic source-recovery
  experiment, NOT measured galaxy physics**. Rest wavelengths are
  `literature_derived` atomic constants; line ratios and rotation/dispersion
  amplitudes are `engineering_estimate`; the Sersic morphology, inclination and
  dust law are `synthetic_proxy`. The demonstrated behaviours (face-on velocity
  span < edge-on; larger Sersic radius broadens the light; lines land at
  `(1+z)·rest`; the dust law attenuates bluer lines more; recovered kinematics
  track the truth) are internal trend checks, not a measured redshift, rotation
  curve or metallicity. Rotation-curve form: Courteau (1997AJ....114.2402C).

## Portfolio extensions — also synthetic / illustrative

* **High-z galaxy science case** (`run_science_case_highz_galaxy.py`) is a
  **synthetic science-recovery test**: a known truth cube pushed through the model
  and recovered. It is *not* a real galaxy observation, redshift measurement, or
  flux calibration.
* **Systematics scans** (`systematics.py`) are **controlled single-parameter
  perturbation sanity checks**, *not* a real instrument tolerance budget; the
  perturbation amplitudes are engineering estimates.
* **Sky / telluric residuals** (`sky_telluric.py`) are a controlled
  inject-then-imperfectly-remove experiment, *not* a real reduction pipeline; the
  SkyCalc radiance→electrons scaling is an engineering estimate.
* **Reconstruction limitations** (`cube_reconstruction.py`) are illustrative:
  weighted/optimal extraction can **over-count** flux shared between closely
  packed IFU traces (flux-conservation ratio > 1 on filled fields), the
  trace-offset reconstructor is deliberately biased, and the covariance proxy is
  an illustrative spectral autocorrelation, not a calibrated covariance matrix.

## What is intentionally *not* validated

* **Absolute photometry / throughput** — the AB-magnitude photon budget is an
  engineering estimate (single scalar throughput, band-centre approximation).
* **Detector cosmetics** — read/dark/flat/bad-pixel parameters are H2RG/H4RG-like
  proxies, not a measured device. 1/f noise, reference pixels, persistence,
  inter-pixel capacitance and up-the-ramp sampling are not modelled.
* **Slicer geometry** — the rectangular image-slicer remap is a synthetic proxy,
  not a real slicer's optical layout, diffraction or field distortion.
* **Atmosphere / AO residuals** — PSFs are seeing/AO/diffraction proxies; there is
  no wavefront-error or time-variable AO residual model.
* **Wavelength solution realism** — a single linear order with optional
  polynomial perturbations; no echelle orders, no real arc-lamp calibration.
* **Public-conditioned sky/throughput is not an ETC** — the `public_conditioned`
  mode reads *real* SVO / ESO SkyCalc / CALSPEC data, but the way it is used is a
  demonstrator, not a calibrated exposure-time calculator:
  - the **sky-background** electron level is a physically-*motivated* unit
    conversion of SkyCalc radiance, scaled by an `engineering_estimate`
    (`sky_scaling`); it is **not** a validated background prediction;
  - the **filter weighting** applies the SVO bandpass shape, but the overall
    throughput is still a scalar `engineering_estimate` (no real optics/QE curve);
  - the reported **SNR** propagates Poisson + dark + read-noise variance only —
    flat-field response error and saturation are applied to the data but treated
    as calibratable systematics and are **not** in the variance, and no
    inter-pixel/spectral covariance is propagated. (This is visible in the
    calibration-sensitivity figure, where flat-field shows ~0 SNR loss.)

For the full non-overclaiming statement and the data-provenance table, see the
[README](../README.md).
