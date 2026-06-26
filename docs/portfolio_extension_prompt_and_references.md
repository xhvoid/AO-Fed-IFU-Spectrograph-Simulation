# Portfolio Extension Prompt and References

This file contains a complete step-by-step prompt for extending the
AO-Fed-IFU-Spectrograph-Simulation repository as a self-learning / portfolio
project, plus a compact list of downloadable public data sources and literature
references.

Copy the full prompt below into Claude Code.

---

## Step-by-step prompt

```text
You are taking over the AO-Fed-IFU-Spectrograph-Simulation repo. The goal is to
extend it from a public-conditioned AO-fed IFU demonstrator into a richer
self-learning / portfolio project. Implement all 7 extension directions below,
but keep the project boundary strict:

- This is a learning / portfolio demonstrator.
- It is NOT a calibrated observatory simulator.
- It is NOT an exposure-time calculator.
- It does NOT reproduce HARMONI, MUSE, NIRSpec, or any real instrument.
- It does NOT claim validated performance predictions.
- It must remain CPU-only, lightweight, inspectable, and runnable offline after
  public-data caches are present.
- Do not import or modify any old Shack-Hartmann AO repository.

Global requirements:
- Every generated CSV / JSON table must include `source_class` or
  `source_classes_used`.
- Valid provenance classes are exactly:
  `direct_public_data`, `literature_derived`, `engineering_estimate`,
  `synthetic_proxy`, `internal_validation`.
- Public-data fetch scripts may use the network, but ordinary tests, examples,
  and CI must not require network access.
- Commit only small curated ASCII/cache samples. Do not commit large raw FITS,
  VOTable, or temporary downloads.
- Keep all non-overclaiming language in README/docs.
- Prefer small additions over large rewrites. Reuse existing modules whenever
  possible.

0. Initial audit and baseline
1. Run `git status`.
2. Read:
   - README.md
   - docs/references.md
   - docs/validation.md
   - src/ao_fed_ifu/data_sources.py
   - src/ao_fed_ifu/integration.py
   - src/ao_fed_ifu/diagnostics.py
   - src/ao_fed_ifu/validation.py
   - examples/run_public_data_figures.py
   - tests/test_public_data.py
3. Run:
   - `pytest -q`
   - `python examples/run_toy_ifu_demo.py`
   - `python examples/run_public_anchor_overview.py`
   - `python examples/run_public_data_figures.py`
4. If baseline tests fail, fix the baseline first before adding features.

1. Full science case: high-z emission-line galaxy recovery
Goal: add a complete science story from synthetic high-z galaxy to recovered
line flux / velocity / SNR products.

Implementation:
1. Add `examples/run_science_case_highz_galaxy.py`.
2. Add or extend a high-z galaxy preset:
   - `source_kind="highz_galaxy"` or `make_highz_galaxy_config()`
   - support rest-frame H-alpha 0.6563 um, [O III] 0.5007 um, [O II] 0.3727 um
   - compute observed wavelength from redshift
   - default redshift should place the line in H or K band
   - use extended galaxy spatial profile, velocity gradient, and intrinsic line
     dispersion proxy
   - use public-conditioned data: SVO filter, SkyCalc telluric/transmission/
     radiance, CALSPEC or optional source continuum
3. Output to `figures/generated/`:
   - `highz_truth_cube_summary.png`
   - `highz_detector_frame.png`
   - `highz_recovered_products.png`
   - `highz_recovery_bias.png`
   - `highz_science_metrics.csv`
   - `highz_science_summary.json`
4. `highz_science_metrics.csv` must include:
   `metric,value,source_class,component,notes`.
5. Summary JSON must include:
   - config
   - line name, rest wavelength, redshift, observed wavelength
   - provenance log
   - public/synthetic component counts
   - limitations
6. Update README with a science-case section and representative figure.
7. Update docs/validation.md to state that this is a synthetic science-recovery
   test, not real galaxy validation.

Tests:
1. Add `tests/test_science_case.py`.
2. Test observed wavelength lies inside the configured wavelength range.
3. Test the core generation function runs with a tiny config.
4. Test CSV rows include valid `source_class`.
5. Test summary JSON includes a non-overclaiming note.

2. Calibration / systematics sensitivity report
Goal: turn the existing calibration sensitivity figure into a fuller
instrument-systematics portfolio module.

Implementation:
1. Add `src/ao_fed_ifu/systematics.py` or extend `diagnostics.py`.
2. Support perturbations:
   - wavelength zero-point offset
   - wavelength stretch
   - trace row offset
   - cross-talk sigma
   - read noise
   - flat-field residual
   - bad-pixel fraction
   - sky-subtraction scale residual
   - telluric-correction residual
3. Add `examples/run_systematics_report.py`.
4. Output:
   - `systematics_bias_matrix.png`
   - `systematics_waterfall_spectrum.png`
   - `systematics_metric_trends.png`
   - `systematics_metrics.csv`
   - `systematics_summary.json`
5. Metrics should include:
   - median velocity bias in km/s
   - median line-flux bias fraction
   - median SNR loss fraction
   - recovered R bias
   - cross-talk leakage fraction
6. Provenance:
   - `internal_validation` for controlled trend checks
   - `engineering_estimate` for assumed perturbation amplitudes
   - `synthetic_proxy` for simulated instrument internals
7. Update docs/validation.md:
   - these are controlled perturbation sanity checks
   - not a real tolerance budget

Tests:
1. Add `tests/test_systematics.py`.
2. Test wavelength offset velocity bias is close to `c * d_lambda / lambda`.
3. Test higher read noise lowers SNR.
4. Test higher cross-talk increases leakage.
5. Test systematics CSV includes valid provenance.

3. Sky subtraction / telluric correction module
Goal: make ESO SkyCalc public data a clear technical highlight.

Implementation:
1. Add `src/ao_fed_ifu/sky_telluric.py`.
2. Implement:
   - `build_sky_background_spectrum(wavelength_grid, sky_model, config)`
   - `subtract_sky(cube_or_detector, sky_model, scale_error=0.0)`
   - `apply_telluric_absorption(cube, transmission)`
   - `correct_telluric(cube, transmission, transmission_error=0.0, floor=...)`
   - `sky_residual_metrics(...)`
3. Support public SkyCalc cache and synthetic fallback.
4. Add `examples/run_sky_telluric_residuals.py`.
5. Output:
   - `sky_telluric_inputs.png`
   - `sky_subtraction_residuals.png`
   - `telluric_correction_effect.png`
   - `sky_residual_science_impact.png`
   - `sky_telluric_metrics.csv`
   - `sky_telluric_summary.json`
6. Metrics should include:
   - sky scale error
   - residual sky fraction
   - line-flux bias fraction
   - velocity bias km/s
   - SNR loss fraction
   - telluric minimum transmission
7. Documentation must state:
   - SkyCalc radiance/transmission is `direct_public_data` if cached
   - sky residual scaling is `engineering_estimate`
   - correction is a synthetic controlled experiment, not a real reduction
     pipeline

Tests:
1. Empty cache fallback returns `synthetic_proxy`.
2. Public cache returns `direct_public_data`.
3. Nonzero sky scale error gives nonzero residual metrics.
4. CSV provenance is valid.

4. Multi-mode comparison dashboard
Goal: make a one-page comparison of major modes and outputs.

Implementation:
1. Add `examples/run_mode_comparison_dashboard.py`.
2. Compare:
   - toy
   - instrument_inspired
   - public_conditioned
   - highz_galaxy
   - optionally public_conditioned variants with different Strehl / exposure / R
3. Output:
   - `mode_comparison_dashboard.png`
   - `mode_comparison_metrics.csv`
   - `mode_comparison_summary.json`
4. Dashboard should include:
   - SNR map thumbnails
   - extracted spectra
   - detector size / runtime
   - median SNR
   - recovered R
   - PSF EE1 / EE3x3
   - sky_to_source_ratio
   - filter_weighted_throughput
   - provenance class counts
5. CSV rows:
   `mode,metric,value,source_class,notes`.
6. Update README with a comparison-dashboard section.

Tests:
1. Run comparison function with tiny configs.
2. Test CSV includes `source_class`.
3. Test every mode has at least one metric row.
4. Test summary JSON includes mode list and provenance counts.

5. Reconstruction limitations
Goal: keep the controlled inverse mapping, but show reconstruction choices and
limitations more explicitly.

Implementation:
1. Extend `cube_reconstruction.py`:
   - keep `reconstruct_cube_nearest`
   - keep `reconstruct_cube_weighted`
   - add `reconstruct_cube_trace_offset(...)`
   - add `reconstruct_cube_with_bad_pixel_mask(...)`
   - add `estimate_covariance_proxy(...)` or `spectral_correlation_proxy(...)`
   - add `compare_reconstruction_methods(...)`
2. Add `examples/run_reconstruction_limitations.py`.
3. Output:
   - `reconstruction_method_comparison.png`
   - `reconstruction_residual_maps.png`
   - `reconstruction_covariance_proxy.png`
   - `reconstruction_metrics.csv`
   - `reconstruction_summary.json`
4. Compare:
   - nearest
   - weighted
   - weighted + bad-pixel mask
   - trace-offset biased
   - sky-subtracted if sky_telluric.py is implemented
5. Metrics:
   - flux conservation ratio
   - line-flux bias
   - velocity bias
   - median SNR
   - bad-pixel masked fraction
   - covariance proxy / spectral correlation proxy
6. Documentation:
   - not a real pipeline
   - covariance proxy is illustrative only

Tests:
1. No-noise no-offset nearest remains exact.
2. Trace offset introduces nonzero bias.
3. Bad-pixel mask prevents NaN/inf output.
4. CSV provenance is valid.

6. Configurable CLI
Goal: wrap examples in a portfolio-quality CLI.

Implementation:
1. Add console script in pyproject.toml:
   `ao-ifu = ao_fed_ifu.cli:main`
2. Add `src/ao_fed_ifu/cli.py`.
3. CLI subcommands:
   - `ao-ifu run --mode toy|instrument_inspired|public_conditioned|highz --out figures/generated`
   - `ao-ifu public-data --out figures/generated`
   - `ao-ifu compare --out figures/generated`
   - `ao-ifu systematics --out figures/generated`
   - `ao-ifu sky-telluric --out figures/generated`
   - `ao-ifu reconstruct-demo --out figures/generated`
   - `ao-ifu validate`
4. Requirements:
   - no network access
   - output path configurable
   - default small/fast runs
   - print generated artifact paths
   - reasonable exit codes
5. Update README CLI usage.
6. Do not remove existing examples.

Tests:
1. Use subprocess or `main(args)` for:
   - `ao-ifu validate`
   - `ao-ifu run --mode toy --out tmp`
   - `ao-ifu public-data --out tmp --quick`
2. Test generated CSV includes `source_class`.
3. Test invalid mode exits nonzero or raises SystemExit.

7. Lightweight interactive dashboard
Goal: optional portfolio demo interface.

Preferred implementation:
1. Add optional dependency:
   `dashboard = ["streamlit>=1.30"]`
2. Add `apps/interactive_dashboard.py` or
   `examples/run_interactive_dashboard.py`.
3. Features:
   - select mode: toy / public_conditioned / highz
   - sliders: exposure_s, Strehl, resolving_power_R, spaxel_scale_mas,
     sky_scaling
   - display line flux map, velocity map, SNR map, extracted spectrum,
     provenance counts, key metrics table
   - visible note: learning/portfolio demonstrator, not calibrated simulator
4. If Streamlit is missing, print a friendly message:
   `pip install -e .[dashboard]`.
5. Do not make CI depend on Streamlit.

Tests:
1. Dashboard module imports safely or gracefully skips if Streamlit is missing.
2. Test metric formatting helpers.
3. No browser-launch test required.

8. Downloadable data and references documentation
Goal: create one clear data/reference document for portfolio reviewers.

Add `docs/downloadable_data_and_references.md` with:

A. Downloadable public data
1. SVO Filter Profile Service
   - URL: https://svo2.cab.inta-csic.es/theory/fps/
   - Direct examples:
     http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.J
     http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.H
     http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.Ks
   - Use: J/H/K/Ks filter transmission
   - Provenance: direct_public_data if cached
   - Note: cite SVO FPS papers / acknowledgements

2. ESO SkyCalc / Cerro Paranal Sky Model
   - URL: https://www.eso.org/sci/software/pipelines/skytools/skymodel
   - Use: sky radiance and telluric transmission
   - Fetch method: skycalc_cli or web application
   - Provenance: direct_public_data if cached
   - Note: radiance-to-electrons conversion in this repo remains an
     engineering_estimate

3. STScI CALSPEC
   - Overview:
     https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec
   - Archive pattern:
     https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/
   - Current example:
     p041c_stis_001.fits
   - Use: source continuum / flux-standard spectrum
   - Provenance: direct_public_data if cached

4. SDSS DR17 optical spectra
   - URL: https://www.sdss4.org/dr17/spectro/
   - Use: optional public galaxy/star spectra for source templates
   - Note: optical spectra may not overlap K band without redshifting or
     synthetic transformation
   - Do not commit large FITS batches

5. PHOENIX / Goettingen Spectral Library
   - URL: https://phoenix.astro.physik.uni-goettingen.de/
   - Use: optional stellar templates
   - Provenance: direct_public_data if cached, literature reference for
     synthetic stellar spectra

B. Literature / conceptual references
Include links and one-line purpose for:
   - HARMONI overview: https://arxiv.org/abs/2103.11215
   - HARMONI image slicer: https://arxiv.org/abs/1807.06313
   - HARMONI tolerance/performance inspiration: https://arxiv.org/abs/2410.01581
   - MUSE design: https://arxiv.org/abs/astro-ph/0606329
   - MUSE pipeline: https://arxiv.org/abs/2006.08638
   - BlueMUSE pipeline: https://arxiv.org/abs/2209.06022
   - Noll et al. sky model: https://arxiv.org/abs/1205.2003
   - Molecfit: https://arxiv.org/abs/1501.07239
   - Skycorr: https://arxiv.org/abs/1405.3679
   - ESO ETC 2.0: https://arxiv.org/abs/2407.02972
   - PypeIt: https://arxiv.org/abs/2005.06505
   - PypeIt docs: https://pypeit.readthedocs.io/en/latest/
   - Horne 1986 optimal extraction:
     https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract
   - MOSFIRE H2RG detector: https://arxiv.org/abs/1208.0314
   - JWST/NIRSpec detector noise: https://arxiv.org/abs/2208.12686
   - Rauscher detector noise generator: https://arxiv.org/abs/1509.06264

C. Data policy
   - curated small ASCII samples may be committed
   - raw FITS / VOTable / large downloads should not be committed
   - network fetch scripts are developer tools, not CI/runtime requirements
   - include provider attribution and terms note

Update README.md and docs/references.md to link to
docs/downloadable_data_and_references.md.

9. Artifact and provenance consolidation
1. Ensure all examples write to `figures/generated/` by default.
2. Ensure every generated table has `source_class`.
3. Extend `artifact_manifest.csv` to include new artifacts:
   - high-z science artifacts
   - systematics artifacts
   - sky/telluric artifacts
   - mode comparison artifacts
   - reconstruction limitation artifacts
   - public-data artifacts
4. Required manifest columns:
   `artifact_path, artifact_type, generated_by, source_class,
   source_classes_used, inputs_used, notes`
5. `source_class` must be one valid primary class.
6. `source_classes_used` must be semicolon-separated valid classes only.
7. Add or update tests for manifest coverage.

10. Final validation
Run:
   - `pytest -q`
   - `python examples/run_toy_ifu_demo.py`
   - `python examples/run_public_anchor_overview.py`
   - `python examples/run_public_data_figures.py`
   - `python examples/run_science_case_highz_galaxy.py`
   - `python examples/run_systematics_report.py`
   - `python examples/run_sky_telluric_residuals.py`
   - `python examples/run_mode_comparison_dashboard.py`
   - `python examples/run_reconstruction_limitations.py`
   - `ao-ifu validate`
   - `ao-ifu run --mode toy --out /tmp/ao_ifu_cli_test`
   - `ao-ifu public-data --out /tmp/ao_ifu_cli_test --quick`

If full scripts are too slow for CI, add `--quick` mode and use quick mode in
tests.

Final response should report:
   - files added
   - files modified
   - new modules and examples
   - generated figures/tables
   - public data used and provenance
   - tests run and results
   - limitations still intentionally not modeled
   - confirmation that this is still not a calibrated ETC, not a real
     instrument reproduction, and not a validated performance prediction
```

---

## Downloadable data and references

| Type | Link | Use |
|---|---|---|
| SVO Filter Profile Service | https://svo2.cab.inta-csic.es/theory/fps/ | J/H/K/Ks filter transmission curves. |
| SVO 2MASS J | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.J | Direct example for `J.dat`. |
| SVO 2MASS H | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.H | Direct example for `H.dat`. |
| SVO 2MASS Ks | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.Ks | Direct example for `K.dat`. |
| ESO SkyCalc / Cerro Paranal Sky Model | https://www.eso.org/sci/software/pipelines/skytools/skymodel | Sky radiance and telluric transmission. |
| STScI CALSPEC overview | https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec | Flux-standard / source-continuum spectra. |
| CALSPEC archive | https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/ | Direct FITS archive, e.g. `p041c_stis_001.fits`. |
| SDSS DR17 spectra | https://www.sdss4.org/dr17/spectro/ | Optional public galaxy/star spectra for future source templates. |
| PHOENIX / Goettingen Spectral Library | https://phoenix.astro.physik.uni-goettingen.de/ | Optional stellar templates. |
| HARMONI overview | https://arxiv.org/abs/2103.11215 | AO-fed IFU context and mode anchors. |
| HARMONI image slicer | https://arxiv.org/abs/1807.06313 | Image slicer / pseudo-slit concept. |
| HARMONI tolerance inspiration | https://arxiv.org/abs/2410.01581 | Calibration/systematics sensitivity language. |
| MUSE design | https://arxiv.org/abs/astro-ph/0606329 | IFU architecture context. |
| MUSE pipeline | https://arxiv.org/abs/2006.08638 | Cube reconstruction / pipeline concept. |
| BlueMUSE pipeline | https://arxiv.org/abs/2209.06022 | LSF, wavelength calibration, synthetic cube prep. |
| Noll et al. sky model | https://arxiv.org/abs/1205.2003 | Paranal sky-model physics. |
| Molecfit | https://arxiv.org/abs/1501.07239 | Telluric correction context. |
| Skycorr | https://arxiv.org/abs/1405.3679 | Sky-line variability / sky residual context. |
| ESO ETC 2.0 | https://arxiv.org/abs/2407.02972 | ETC/instrument-package context only. |
| PypeIt paper | https://arxiv.org/abs/2005.06505 | Spectroscopic reduction vocabulary. |
| PypeIt docs | https://pypeit.readthedocs.io/en/latest/ | Practical reduction vocabulary. |
| Horne 1986 optimal extraction | https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract | Optimal extraction / variance weighting. |
| MOSFIRE H2RG detector | https://arxiv.org/abs/1208.0314 | Detector-noise proxy references. |
| JWST/NIRSpec detector noise | https://arxiv.org/abs/2208.12686 | Detector-system noise context. |
| Rauscher detector noise generator | https://arxiv.org/abs/1509.06264 | Realistic detector-noise vocabulary. |

## Data policy

- Curated small ASCII samples may be committed.
- Raw FITS / VOTable / large downloads should not be committed.
- Network fetch scripts are developer tools, not CI/runtime requirements.
- Include provider attribution and terms notes.
- Public data may be `direct_public_data`, but derived sky/throughput scaling
  may still be `engineering_estimate`.

