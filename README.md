# AO-Fed-IFU-Spectrograph-Simulation

[![CI](https://github.com/xhvoid/AO-Fed-IFU-Spectrograph-Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/xhvoid/AO-Fed-IFU-Spectrograph-Simulation/actions/workflows/ci.yml)

An end-to-end Python model of an adaptive-optics-fed integral-field
spectrograph. A datacube with known truth is pushed through AO PSF coupling,
an image-slicer IFU, a spectrograph and an infrared detector, then
reconstructed, and every science product is measured against the truth it
came from.

This is a learning and portfolio project, not a calibrated instrument
simulator or exposure-time calculator. Real instruments (HARMONI, MUSE,
NIRSpec) only anchor parameter ranges, and every output is labelled as
`direct_public_data`, `literature_derived`, `engineering_estimate`,
`synthetic_proxy` or `internal_validation`.

![High-z galaxy recovered products](figures/highz_recovered_products.png)

*A rotating Hα galaxy at z = 2.3, observed in the K band against the real ESO
SkyCalc sky and recovered from the simulated detector frame. Top: line flux,
velocity field and line SNR. Bottom right: the aperture spectrum before
(orange) and after sky subtraction, with a Gaussian line fit.*

## What it models

```
source cube -> AO PSF coupling -> image slicer -> spectrograph LSF -> IR detector
  -> calibration -> cube reconstruction -> sky subtraction -> flux / velocity / SNR / R
```

| Stage | Module | What is modelled |
|-------|--------|------------------|
| Source | [`source_cube.py`](src/ao_fed_ifu/source_cube.py), [`galaxy_kinematics.py`](src/ao_fed_ifu/galaxy_kinematics.py) | Truth cube with an AB-magnitude photon budget; Sérsic disc with a rotation curve and redshifted nebular lines |
| Atmosphere and sky | [`data_sources.py`](src/ao_fed_ifu/data_sources.py), [`sky_telluric.py`](src/ao_fed_ifu/sky_telluric.py) | ESO SkyCalc telluric absorption and OH sky emission, SVO Ks filter, CALSPEC continuum |
| AO PSF | [`psf_coupling.py`](src/ao_fed_ifu/psf_coupling.py), [`ao_psf_physics.py`](src/ao_fed_ifu/ao_psf_physics.py) | Strehl-weighted diffraction core + seeing halo, Maréchal Strehl vs wavelength, encircled energy per spaxel |
| IFU | [`ifu_mapping.py`](src/ao_fed_ifu/ifu_mapping.py) | Image slicer laying slices along a pseudo-slit; cross-talk between neighbouring traces |
| Spectrograph | [`spectrograph_model.py`](src/ao_fed_ifu/spectrograph_model.py) | Wavelength solution, Gaussian LSF, resolving power and sampling |
| Detector | [`detector_model.py`](src/ao_fed_ifu/detector_model.py), [`detector_systematics.py`](src/ao_fed_ifu/detector_systematics.py) | Poisson, read and dark noise, flat field, bad pixels, saturation; up-the-ramp sampling, nonlinearity, IPC |
| Reduction | [`cube_reconstruction.py`](src/ao_fed_ifu/cube_reconstruction.py) | Detector-to-cube inversion with propagated variance; forward-modelled sky subtraction |
| Diagnostics | [`diagnostics.py`](src/ao_fed_ifu/diagnostics.py) | Line flux, velocity, SNR, recovered R, Gaussian line fits, aperture covariance |
| Trade studies | [`systematics.py`](src/ao_fed_ifu/systematics.py), [`observing_strategy.py`](src/ao_fed_ifu/observing_strategy.py), [`throughput_budget.py`](src/ao_fed_ifu/throughput_budget.py) | Calibration-error scans, dithered coadds with cosmic-ray rejection, throughput cascade |

## Results

- **High-z recovery.** Against a noise-free run through the same reduction,
  line flux is recovered to 1.2% (median over the aperture spaxels) and 0.2%
  (Gaussian fit to the aperture spectrum), with a median velocity bias of
  0.05 km/s. 65% of spaxels yield a velocity at SNR ≥ 3.
- **Choosing the reconstruction.** Each spaxel lands on one detector row, so
  deconvolving the slicer cross-talk (a joint least-squares fit) has almost
  no redundant data. On the high-z case it cuts the median spaxel SNR from 4.9
  to 2.0, and one bad pixel can blank a run of neighbouring spaxels (coverage
  falls from 88% to 43%). Slicer pipelines resample rather than deconvolve, so
  the default is a direct read-out with the cross-talk left as a small spatial
  blur. The deblend stays available as `science_extraction="deblend"`, and
  [notebook 04](notebooks/04_cube_reconstruction_noise_snr.ipynb) compares
  the two.
- **Calibration sensitivity.** A one-pixel wavelength zero-point error biases
  velocities by 58 km/s (Δv = cΔλ/λ). A one-row trace error biases line flux
  by 1.2%. A sky-model scale error ε leaves exactly ε of the sky in the
  reduced cube; at ε = 10% it biases line flux by −9%
  ([systematics](figures/systematics_metric_trends.png),
  [sky residuals](figures/sky_residual_science_impact.png)).
- **AO across the band.** For 150 nm of residual wavefront error on an 8 m
  telescope, Strehl rises from 0.41 at 1.0 µm to 0.86 at 2.4 µm while the
  diffraction core grows from 26 to 63 mas
  ([PSF grid](figures/ao_psf_wavelength_grid.png)).
- **Throughput.** The Ks-band cascade from atmosphere to detector QE delivers
  15% end to end. The band-averaged filter (0.55) and the IFU slicer (0.70)
  are the largest single losses
  ([waterfall](figures/throughput_budget_waterfall.png)).
- **Observing strategy.** Coadding six dithered exposures raises the line SNR
  from 8.4 to 19.5 (log slope 0.47 against the ideal 0.5), and a one-sided
  sigma clip rejects cosmic rays at their 1% injected rate
  ([coadd gain](figures/coadd_snr_gain.png)).

## Quick start

CPU only, Python ≥ 3.10, offline after installation.

```bash
git clone https://github.com/xhvoid/AO-Fed-IFU-Spectrograph-Simulation.git
cd AO-Fed-IFU-Spectrograph-Simulation
python -m pip install -e ".[test]"

python examples/run_toy_ifu_demo.py               # end-to-end toy run with a metrics summary (~1 s)
python examples/run_science_case_highz_galaxy.py  # the high-z case above (~15 s)
ao-ifu validate                                   # 7 physics trend checks
```

```python
from ao_fed_ifu.integration import make_highz_galaxy_config, run_pipeline

result = run_pipeline(make_highz_galaxy_config(redshift=2.3))
print(result.metrics["median_snr"], result.metrics["line_moment_detected_spaxel_fraction"])
```

The `ao-ifu` command also regenerates each report (`ao-ifu systematics`,
`ao-ifu ao-psf`, `ao-ifu throughput`, ...); see
[docs/model_details.md](docs/model_details.md#command-line-interface).

## Validation

- **335 automated checks**: 317 unit tests and 18 gates on the committed
  results. CI lints the code, runs everything on Python 3.10 and 3.12,
  regenerates the science reports from scratch, and fails if a committed
  figure was produced by different code (each summary records a hash of the
  source).
- **Physics trend checks**: flux conservation through the slicer map
  (ratio 1 to 12 decimal places), SNR ∝ √t in the photon limit (slope 0.501),
  recovered R tracks configured R, Δv = cΔλ/λ, and encircled energy rises with
  Strehl.
- **Scope**: the checks test the model against first-principles physics,
  not against real instrument data. [docs/validation.md](docs/validation.md)
  states exactly what is and is not validated.

## Data and provenance

Small samples of real public data ship in [`data/public_cache/`](data/public_cache/)
so that tests and examples run offline: SVO 2MASS J/H/Ks filter curves, an ESO
SkyCalc sky radiance and transmission model, and a CALSPEC flux standard.
Without the cache, every loader falls back to a labelled synthetic proxy.
[docs/references.md](docs/references.md) maps each literature and data
reference to the module that uses it.

## Limitations

- **Not an ETC.** A single scalar instrument throughput is used, and the sky
  level is a physically motivated but uncalibrated conversion of SkyCalc
  radiance.
- **Slicer.** The geometry is a rectangular proxy with one detector row per
  spaxel: no slicer optics, diffraction or field distortion.
- **AO.** The PSF is a core-plus-halo proxy: no phase screens and no
  time-variable residuals.
- **Detector.** Noise uses H2RG-like scalar parameters. Nonlinearity, IPC and
  correlated noise are illustrative extensions, and persistence is not modelled.
- **Sky subtraction.** The sky model is assumed to be perfectly known, apart
  from a deliberate scale error used in experiments.

The full list is in [docs/model_details.md](docs/model_details.md#limitations-intentionally-not-modelled).

## Repository guide

| Path | Contents |
|------|----------|
| [`docs/architecture.md`](docs/architecture.md) | Pipeline diagram, module responsibilities, unit and axis conventions |
| [`docs/model_details.md`](docs/model_details.md) | Run modes, CLI, extension modules, regenerating figures, full limitations |
| [`notebooks/`](notebooks/) | Six walk-throughs with saved outputs: source cube and photon budget, AO PSF sampling, slicer remapping, reconstruction and SNR, calibration sensitivity, public-data conditioning |
| [`figures/`](figures/) | Every committed figure and table, regenerated by `examples/run_*.py` and indexed in [`artifact_manifest.csv`](figures/artifact_manifest.csv) |

A related [Astrophotonics study](https://github.com/xhvoid/Astrophotonics)
(coherent fibre coupling and radial-velocity measurement) is maintained as a
separate project.

## Author, citation and license

Created by **Xu Han**. To cite it, use [`CITATION.cff`](CITATION.cff):
Han, Xu. *AO-Fed-IFU-Spectrograph-Simulation* (v0.1.0).

[MIT](LICENSE) © 2026 Xu Han.
