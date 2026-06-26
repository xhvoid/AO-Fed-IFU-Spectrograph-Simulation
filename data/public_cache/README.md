# Public-data cache

This folder holds small public datasets used to *condition* the simulation on
real measurements.  The loaders in [`data_sources.py`](../../src/ao_fed_ifu/data_sources.py)
read these files if present (provenance `direct_public_data`) and otherwise fall
back to documented synthetic proxies (`synthetic_proxy`).  **Tests and examples
never touch the network** — they read these committed files or use the fallback.

## What is bundled here

These small samples were fetched by
[`examples/fetch_public_data.py`](../../examples/fetch_public_data.py) and are
committed so the `public_conditioned` pipeline mode and
`examples/run_public_anchor_overview.py` demonstrate real `direct_public_data`:

| File | Source | Content |
|------|--------|---------|
| `J.dat`, `H.dat`, `K.dat` | [SVO FPS](https://svo2.cab.inta-csic.es/theory/fps/) 2MASS J/H/Ks | filter transmission |
| `skycalc.dat` | [ESO SkyCalc](https://www.eso.org/sci/software/pipelines/skytools/skymodel) (airmass 1.0, 1.9–2.5 µm) | sky radiance + telluric transmission |
| `source.dat` | [STScI CALSPEC](https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/) standard `p041c_stis_001` | flux-standard spectrum |

Each file carries a `#` provenance header. To refresh or extend them:

```bash
pip install -e .[data]          # astropy + skycalc_cli + certifi
python examples/fetch_public_data.py
```

**Attribution / terms.** These are public products of the SVO Filter Profile
Service, the ESO Cerro Paranal Sky Model, and STScI CALSPEC; please cite those
services if you use the data. Only small samples are redistributed here for
demonstration; see each provider for full datasets and usage terms.

Provenance for anything placed here is `direct_public_data`.

## Expected file formats

| File              | Columns (whitespace-separated ASCII)                 | Source |
|-------------------|------------------------------------------------------|--------|
| `J.dat`, `H.dat`, `K.dat` | `wavelength_um  transmission`                | [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/) |
| `skycalc.dat`     | `wavelength_um  radiance  transmission`              | [ESO SkyCalc / Cerro Paranal Sky Model](https://www.eso.org/sci/software/pipelines/skytools/skymodel) |
| `source.dat`      | `wavelength_um  flux` (any flux units; peak-normalised on load) | [CALSPEC](https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec) / [PHOENIX](https://phoenix.astro.physik.uni-goettingen.de/) |

### Notes

* **Filter curves** — e.g. the SVO ID pattern `2MASS/2MASS.J` (and `.H`, `.Ks`).
  Export an ASCII profile, convert the wavelength column to micrometres, and save
  as `J.dat` etc.
* **Sky / telluric** — SkyCalc outputs sky emission *radiance*
  (ph/s/m²/µm/arcsec²) and atmospheric *transmission*; keep three columns as
  above. See the [SkyCalc help page](https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html)
  for the column definitions.
* **Source spectrum** — any stellar/flux-standard spectrum resampled to two
  columns. The loader peak-normalises it and uses it only as a continuum shape;
  absolute scaling comes from the AB-magnitude photon budget.

These files are git-ignored by default (see [`.gitignore`](../../.gitignore)) so
large downloads are not accidentally committed.
