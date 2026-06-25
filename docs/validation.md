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

For the full non-overclaiming statement and the data-provenance table, see the
[README](../README.md).
