# Downloadable data and references

A single, reviewer-facing list of the public data and literature this project
draws on, and how each is used.  See [`references.md`](references.md) for the
module-by-module mapping, and the [README](../README.md) provenance table for the
non-overclaiming boundary.

**Boundary:** this is a learning/portfolio demonstrator. Public data is used as
*anchors* and *conditioning*; the instrument internals are synthetic proxies, the
sky/throughput scalings are engineering estimates, and nothing here is a
calibrated ETC, an instrument reproduction, or a validated performance prediction.

## A. Downloadable public data

| # | Source | URL | Use here | Provenance |
|---|--------|-----|----------|------------|
| 1 | SVO Filter Profile Service | https://svo2.cab.inta-csic.es/theory/fps/ | J/H/K/Ks filter transmission; **used** as a wavelength-dependent bandpass | `direct_public_data` if cached |
| | — 2MASS J | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.J | example for `J.dat` | |
| | — 2MASS H | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.H | example for `H.dat` | |
| | — 2MASS Ks | http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=2MASS/2MASS.Ks | example for `K.dat` | |
| 2 | ESO SkyCalc / Cerro Paranal Sky Model | https://www.eso.org/sci/software/pipelines/skytools/skymodel | sky radiance (→ sky-background electrons) and telluric transmission; **used** in public-conditioning and the sky/telluric module | `direct_public_data` if cached; radiance→electrons scaling is `engineering_estimate` |
| 3 | STScI CALSPEC | https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec | flux-standard / source-continuum spectrum (`p041c_stis_001.fits`); **used** as the source continuum shape | `direct_public_data` if cached |
| | — CALSPEC archive | https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/ | direct FITS archive | |
| 4 | SDSS DR17 optical spectra | https://www.sdss4.org/dr17/spectro/ | **optional, not currently used** — optical spectra need redshifting/transform to overlap the K band; do not commit large FITS batches | would be `direct_public_data` if used |
| 5 | PHOENIX / Göttingen Spectral Library | https://phoenix.astro.physik.uni-goettingen.de/ | **optional, not currently bundled** — alternative stellar templates | `direct_public_data` if cached, else literature reference |

The bundled samples and the fetch tool are described in
[`data/public_cache/README.md`](../data/public_cache/README.md):

```bash
pip install -e .[data]
python examples/fetch_public_data.py    # refresh SVO / SkyCalc / CALSPEC samples
```

### Emission-line rest wavelengths (high-z science case)

The high-z galaxy preset places a redshifted nebular line in the band.  The
rest-frame **vacuum** wavelengths (`Halpha` 0.65646, `Hbeta` 0.48627, `[O III]`
0.50082, `[O II]` 0.37271 µm) are standard atomic constants (`literature_derived`)
from line lists such as the NIST Atomic Spectra Database
(https://www.nist.gov/pml/atomic-spectra-database), on one consistent (vacuum)
convention so relative line spacings are correct.

## B. Literature / conceptual references

| Reference | arXiv / link | Purpose |
|-----------|--------------|---------|
| HARMONI overview | [2103.11215](https://arxiv.org/abs/2103.11215) | AO-fed IFU context and mode anchors |
| HARMONI image slicer | [1807.06313](https://arxiv.org/abs/1807.06313) | image-slicer / pseudo-slit concept |
| HARMONI tolerance/performance | [2410.01581](https://arxiv.org/abs/2410.01581) | calibration/systematics-sensitivity language |
| MUSE design | [astro-ph/0606329](https://arxiv.org/abs/astro-ph/0606329) | IFU architecture context |
| MUSE pipeline | [2006.08638](https://arxiv.org/abs/2006.08638) | cube reconstruction / pipeline concept |
| BlueMUSE pipeline | [2209.06022](https://arxiv.org/abs/2209.06022) | LSF, wavelength calibration, synthetic cubes |
| Noll et al. sky model | [1205.2003](https://arxiv.org/abs/1205.2003) | Paranal sky-model physics |
| Molecfit | [1501.07239](https://arxiv.org/abs/1501.07239) | telluric-correction context |
| Skycorr | [1405.3679](https://arxiv.org/abs/1405.3679) | sky-line variability / residual context |
| ESO ETC 2.0 | [2407.02972](https://arxiv.org/abs/2407.02972) | ETC/instrument-package context only |
| PypeIt | [2005.06505](https://arxiv.org/abs/2005.06505) | spectroscopic reduction vocabulary |
| PypeIt docs | https://pypeit.readthedocs.io/en/latest/ | practical reduction vocabulary |
| PyWiFeS | [1311.2666](https://arxiv.org/abs/1311.2666) | IFU/spectrograph global optical model |
| Horne 1986 optimal extraction | [1986PASP...98..609H](https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract) | optimal extraction / variance weighting |
| MOSFIRE H2RG detector | [1208.0314](https://arxiv.org/abs/1208.0314) | detector-noise proxy |
| JWST/NIRSpec detector noise | [2208.12686](https://arxiv.org/abs/2208.12686) | detector-system noise context |
| Rauscher detector noise generator | [1509.06264](https://arxiv.org/abs/1509.06264) | detector-noise vocabulary |
| Ciotti & Bertin 1999 | (Sérsic b_n approximation) | extended-galaxy spatial profile |

## C. Data policy

* Curated **small ASCII samples may be committed** (`data/public_cache/*.dat`).
* Raw **FITS / VOTable / large downloads must not be committed** (git-ignored).
* Network **fetch scripts are developer tools**, not CI/runtime requirements; the
  tests, examples and CLI run fully offline.
* Include **provider attribution** and consult provider terms (SVO, ESO, STScI).
* Public data may be `direct_public_data`, but derived sky/throughput scalings
  remain `engineering_estimate`.
