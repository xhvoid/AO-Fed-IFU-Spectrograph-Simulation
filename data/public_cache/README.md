# Optional public-data cache

This folder is an **optional** local cache of small public datasets used to
*condition* the simulation on real measurements.  Nothing here is required: when
a file is absent, the loaders in [`data_sources.py`](../../src/ao_fed_ifu/data_sources.py)
fall back to documented synthetic proxies and record the substitution in the run
provenance.  Tests and examples never touch the network.

Cache only **small** files, and verify the IDs/URLs in a browser before
downloading.  Provenance for anything placed here is `direct_public_data`.

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
