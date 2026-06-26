# References and data-source map

This document maps **every reference in the project's data/reference index** to
the module that uses it, and states *how* each is used: as a literature anchor
for a parameter range, as a concept/vocabulary reference, or as the recommended
source for an optional public-data cache.

**Important distinctions**

* These are **anchors and concept references**, not reproductions. Real
  instruments (HARMONI, MUSE, NIRSpec) appear only to justify *plausible
  parameter ranges*; no as-built design, calibration or performance is reproduced.
* **HARMONI/MUSE/NIRSpec are design papers, not datasets** — there is no
  downloadable "HARMONI data". They inform presets in `spectrograph_model.py`,
  `ifu_mapping.py` and `integration.py`.
* **Public *data products*** (SVO filter curves, ESO SkyCalc sky/telluric,
  CALSPEC spectra) are loaded from `data/public_cache/`. Small **real samples are
  bundled** there (fetched by `examples/fetch_public_data.py`), so the
  `public_conditioned` mode reports genuine `direct_public_data`. If the files are
  removed, the loaders in `data_sources.py` fall back to documented synthetic
  proxies (`synthetic_proxy`). Either way, tests and examples run fully offline (a
  hard requirement) — the network is only needed to *re-fetch* the cache.

---

## A. IFU and instrument-mode anchors

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| HARMONI overview | [arXiv:2103.11215](https://arxiv.org/abs/2103.11215) | `spectrograph_model.py`, `integration.py` | ELT AO-IFU context; spaxel-scale, grating/R ranges, HAWAII-4RG complement → preset *anchors* only |
| HARMONI image-slicer preliminary design | [arXiv:1807.06313](https://arxiv.org/abs/1807.06313) | `ifu_mapping.py` | image-slicer concept, field-splitter/relay/slicer/pseudo-slit language |
| HARMONI IR spectrograph tolerance/performance | [arXiv:2410.01581](https://arxiv.org/abs/2410.01581) | `diagnostics.py`, `validation.py` | calibration-sensitivity *inspiration* only (not as-built HARMONI) |
| MUSE design context | [arXiv:astro-ph/0606329](https://arxiv.org/abs/astro-ph/0606329) | `spectrograph_model.py`, `integration.py` | visible IFU architecture, large-cube concept, R~3000 class |
| MUSE pipeline | [arXiv:2006.08638](https://arxiv.org/abs/2006.08638) | `cube_reconstruction.py` | cube-reconstruction/pipeline concepts and validation philosophy |
| BlueMUSE pipeline design | [arXiv:2209.06022](https://arxiv.org/abs/2209.06022) | `spectrograph_model.py`, `cube_reconstruction.py` | LSF handling, wavelength-calibration importance, synthetic-cube preparation |

## B. Sky, telluric and atmospheric background

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| ESO Cerro Paranal Sky Model / SkyCalc | https://www.eso.org/sci/software/pipelines/skytools/skymodel | `data_sources.load_sky_spectrum` | recommended cache source for sky radiance + telluric transmission |
| ESO SkyCalc help (output columns) | https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html | `data_sources.py`, `data/public_cache/README.md` | defines the cache file format (wavelength, radiance, transmission) |
| Noll et al. 2012 (Paranal sky model) | [arXiv:1205.2003](https://arxiv.org/abs/1205.2003) | `data_sources.load_sky_spectrum` | physical components informing the synthetic continuum + OH-line proxy |
| Molecfit method | [arXiv:1501.07239](https://arxiv.org/abs/1501.07239) | `data_sources.py` | telluric-absorption context for the transmission proxy (full Molecfit not implemented) |
| Skycorr | [arXiv:1405.3679](https://arxiv.org/abs/1405.3679) | `data_sources.py` | sky-line variability context for sky-subtraction stress tests |

## C. Filters, wavelength bands, photometric anchors

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| SVO Filter Profile Service | https://svo2.cab.inta-csic.es/theory/fps/ | `data_sources.load_filter_curve` | recommended cache source for J/H/K transmission curves |
| SVO example ID (2MASS.J etc.) | https://svo2.cab.inta-csic.es/theory/fps/index.php?ID=2MASS/2MASS.J | `data/public_cache/README.md` | example filter IDs to cache (verify in browser first) |
| ESO SkyCalc broadband filters / zero points | https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html | `source_cube.photon_budget_from_abmag` (context) | AB/Vega zero-point context for the photon budget |

## D. Source spectra and datacube inputs

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| STScI CALSPEC | https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec | `data_sources.load_source_spectrum` | recommended cache source for flux-standard spectra |
| CALSPEC FITS archive | https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/ | `data/public_cache/README.md` | direct FITS access for small cached examples |
| PHOENIX / Göttingen Spectral Library | https://phoenix.astro.physik.uni-goettingen.de/ | `data_sources.load_source_spectrum` | recommended cache source for synthetic stellar spectra |
| PHOENIX library paper | [arXiv:1303.5632](https://arxiv.org/abs/1303.5632) | `data_sources.py` | wavelength coverage / parameter-space context |
| SDSS optical spectra | https://www.sdss4.org/dr17/spectro/ | `data_sources.py` (optional) | optional galaxy/stellar spectra (less important for NIR) |
| Blackbody / analytic line fallback | — | `data_sources.blackbody_relative`, `source_cube.py` | synthetic_proxy when no cache is present |

## E. IFU remapping and slicer/lenslet references

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| HARMONI image slicer | [arXiv:1807.06313](https://arxiv.org/abs/1807.06313) | `ifu_mapping.py` | primary anchor for the image-slicer-inspired remap and pseudo-slit language |
| GMOS IFU | [arXiv:astro-ph/0202330](https://arxiv.org/abs/astro-ph/0202330) | `ifu_mapping.py` (lenslet note) | lenslet-coupled-fibre IFU concept (deferred lenslet mode) |
| NIRSpec image-slicer concept | [arXiv:physics/0512002](https://arxiv.org/abs/physics/0512002) | `ifu_mapping.py` | image-slicer design motivation |
| MUSE pipeline | [arXiv:2006.08638](https://arxiv.org/abs/2006.08638) | `cube_reconstruction.py` | converting raw IFU measurements into calibrated cubes |
| Rectangular slicer proxy + Gaussian cross-dispersion | — | `ifu_mapping.py` | synthetic_proxy trace table + cross-talk model |

## F. Spectrograph model, extraction, pipeline references

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| ESO ETC 2.0 | [arXiv:2407.02972](https://arxiv.org/abs/2407.02972) | `integration.py`, `spectrograph_model.py` | modern Python ETC/instrument-package *thinking* (ESO ETC not reproduced) |
| PypeIt | [arXiv:2005.06505](https://arxiv.org/abs/2005.06505) | `spectrograph_model.py`, `cube_reconstruction.py` | trace/wavelength/extraction vocabulary |
| PypeIt documentation | https://pypeit.readthedocs.io/en/latest/ | `spectrograph_model.py` | reduction/wavelength-calibration/extraction vocabulary |
| PyWiFeS | [arXiv:1311.2666](https://arxiv.org/abs/1311.2666) | `spectrograph_model.py` | IFU/spectrograph global optical model, wavelength-calibrated 2D spectra |
| Horne 1986 (optimal extraction) | [1986PASP...98..609H](https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract) | `cube_reconstruction.py`, `detector_model.py` | inverse-variance ("optimal") extraction and variance weighting |
| Linear wavelength solution + Gaussian LSF | — | `spectrograph_model.py` | synthetic_proxy dispersion/LSF + controlled wavelength-offset perturbation |

## G. Detector-noise references

| Reference | arXiv / URL | Used in | How |
|-----------|-------------|---------|-----|
| MOSFIRE H2RG performance | [arXiv:1208.0314](https://arxiv.org/abs/1208.0314) | `detector_model.py` | H2RG-like read noise / dark current / sampling *engineering estimate* |
| JWST/NIRSpec detector noise | [arXiv:2208.12686](https://arxiv.org/abs/2208.12686) | `detector_model.py` | H2RG detector-system concepts (dark, read/total noise) |
| Rauscher H1RG/H2RG/H4RG noise generator | [arXiv:1509.06264](https://arxiv.org/abs/1509.06264) | `detector_model.py` | noise-component vocabulary (white/1f/column/picture-frame); only scalar terms modelled |
| Scalar Poisson+read+dark+flat+bad-pixel model | — | `detector_model.py` | engineering_estimate; no real detector cosmetics |

## H. Validation checks and acceptance tests

| Check | Used in | Provenance |
|-------|---------|------------|
| Flux conservation (noiseless remap+reconstruction) | `validation.check_flux_conservation` | internal_validation |
| SNR ~ sqrt(t) in a fixed noise regime | `validation.check_snr_sqrt_exposure` | internal_validation |
| Resolving power: line width scales with `lambda/R` | `validation.check_resolving_power_scaling` | internal_validation |
| Wavelength-offset → velocity bias `dv = c d_lambda/lambda` | `validation.check_wavelength_offset_velocity_bias` | internal_validation |
| Cross-talk trend (more sigma → more leakage) | `validation.check_crosstalk_increases_leakage` | internal_validation |
| AO PSF coupling (higher Strehl/smaller FWHM → higher EE) | `validation.check_ao_psf_improves_ee` | internal_validation |
| Read noise reduces SNR for a faint source | `validation.check_read_noise_reduces_snr` | internal_validation |

Other: the Sersic profile uses the **Ciotti & Bertin (1999)** b_n approximation
(`source_cube.create_extended_galaxy_cube`).

---

## Public-data cache (bundled, refreshable)

Small real samples are committed under
[`data/public_cache/`](../data/public_cache/README.md):

* `J.dat`, `H.dat`, `K.dat` — SVO 2MASS J/H/Ks curves (`wavelength_um  transmission`)
* `skycalc.dat` — ESO SkyCalc output (`wavelength_um  radiance  transmission`)
* `source.dat` — a CALSPEC standard (`wavelength_um  flux`)

`python examples/run_public_anchor_overview.py` (or the `public_conditioned`
pipeline mode) then reports `direct_public_data` for these. To refresh or extend
the cache:

```bash
pip install -e .[data]
python examples/fetch_public_data.py
```

### How the public data enters the model (not just the log)

In `public_conditioned` mode ([`integration._apply_public_conditioning`](../src/ao_fed_ifu/integration.py)):

* **SVO Ks transmission** → a wavelength-dependent filter bandpass multiplying the
  source (metric `filter_weighted_throughput`).
* **SkyCalc transmission** → telluric absorption of the source spectrum.
* **SkyCalc radiance** → a sky-background electron contribution added to the
  detector budget via `radiance · area · exposure · throughput · filter · dλ ·
  Ω_spaxel` (metrics `sky_background_electrons`, `sky_to_source_ratio`). The unit
  conversion is physical; its absolute normalisation is an `engineering_estimate`
  (`sky_scaling`), not a calibrated ETC.
* **CALSPEC spectrum** → the source continuum shape.

`examples/run_public_data_figures.py` writes, offline, the public-input overview,
a public-vs-synthetic comparison, a PSF-sampling grid, a calibration-sensitivity
matrix, plus `public_data_metrics.csv`, `public_data_summary.json` and
`artifact_manifest.csv` (every row carries a `source_class`).
