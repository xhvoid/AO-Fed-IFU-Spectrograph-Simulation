"""Tests for PSF proxies and spaxel encircled-energy coupling."""

import numpy as np
import pytest

from ao_fed_ifu import psf_coupling as psf


def test_psfs_are_normalised_over_the_whole_plane():
    """Unit flux over the plane: a big grid sums to 1, a small grid to the
    fraction it encloses -- never renormalised to 1 by fiat."""
    seeing = psf.seeing_gaussian_psf(500.0, grid_mas=20.0, n_pix=301)   # +-3 FWHM
    ao = psf.ao_corrected_psf_proxy(0.4, 2.2, 8.0, 600.0, grid_mas=10.0, n_pix=601)
    for p in (seeing, ao):
        assert np.sum(p) == pytest.approx(1.0, abs=1e-6)
        assert np.all(p >= 0)
    # the Airy wings fall as 1/x^3, so the enclosed energy converges slowly:
    # ~ 1 - 2/(pi x_max) with x_max = pi D theta_max / lambda
    airy = psf.airy_psf_proxy(2.2, 8.0, grid_mas=2.0, n_pix=2001)
    x_max = np.pi * 8.0 * (1000.0 * 2.0 / psf._RAD_TO_MAS) / 2.2e-6
    assert np.sum(airy) == pytest.approx(1.0, abs=0.01)
    assert np.sum(airy) == pytest.approx(1.0 - 2.0 / (np.pi * x_max), abs=3e-3)
    assert np.all(airy >= 0)
    # truncation is honest: a 9-spaxel kernel of a halo-dominated PSF carries
    # only the energy inside 90 mas, identical to the crop of a big grid
    big = psf.ao_corrected_psf_proxy(0.5, 2.166, 39.0, 600.0, grid_mas=10.0, n_pix=161)
    small = psf.ao_corrected_psf_proxy(0.5, 2.166, 39.0, 600.0, grid_mas=10.0, n_pix=9)
    c = 161 // 2
    assert np.sum(small) == pytest.approx(np.sum(big[c - 4 : c + 5, c - 4 : c + 5]), rel=1e-12)
    assert 0.45 < np.sum(small) < 0.55
    # and the Strehl is the core's energy fraction: a grid holding the core
    # but none of the halo sums to ~S
    core_only = psf.ao_corrected_psf_proxy(0.5, 2.166, 39.0, 60000.0, grid_mas=10.0, n_pix=9)
    assert np.sum(core_only) == pytest.approx(0.5, abs=0.01)


def test_gaussian_pixels_are_exact_area_integrals():
    from scipy.special import erf

    sigma, pixel = 0.3, 1.0  # deliberately much narrower than one pixel
    image = psf._integrated_gaussian_psf(sigma, pixel, n_pix=7)
    centre = image.shape[0] // 2
    expected_central = erf(pixel / (2 * np.sqrt(2) * sigma)) ** 2
    expected_square = erf(image.shape[0] * pixel / (2 * np.sqrt(2) * sigma)) ** 2
    assert image[centre, centre] == pytest.approx(expected_central, abs=2e-15)
    assert image.sum() == pytest.approx(expected_square, abs=2e-15)
    # Centre sampling would assign essentially all weight to the central lattice
    # point; the exact pixel integral retains the finite Gaussian spillover.
    assert image[centre, centre] < 1.0


def test_coarse_gaussian_kernel_matches_fine_area_integration():
    """A fine diagnostic image and the kernel used for spaxel convolution must
    report the same coupling when their pixel boundaries describe the same cells."""
    common = dict(strehl=0.5, lambda_um=2.166, telescope_diameter_m=39.0,
                  seeing_fwhm_mas=600.0)
    coarse = psf.ao_corrected_psf_proxy(**common, grid_mas=10.0, n_pix=7)
    # Five 2-mas pixels tile each 10-mas spaxel exactly while retaining a sample
    # at the optical axis; 35 fine pixels and seven spaxels have the same footprint.
    fine = psf.ao_corrected_psf_proxy(**common, grid_mas=2.0, n_pix=35)
    ee1, ee3, rebinned = psf.encircled_energy_per_spaxel(
        fine, fine_scale_mas=2.0, spaxel_scale_mas=10.0,
    )
    np.testing.assert_allclose(rebinned, coarse, rtol=0, atol=2e-15)
    c = coarse.shape[0] // 2
    assert ee1 == pytest.approx(coarse[c, c], abs=2e-15)
    assert ee3 == pytest.approx(coarse[c - 1:c + 2, c - 1:c + 2].sum(), abs=2e-15)


def test_centred_point_source_uses_analytic_spaxel_integral():
    """Regression: the central spaxel receives the Gaussian *area integral*,
    rather than a centre-sampled value renormalised on a pixel lattice."""
    from functools import partial
    from scipy.special import erf

    strehl, wavelength_um, diameter_m = 0.5, 2.166, 39.0
    seeing_mas, spaxel_mas = 600.0, 10.0
    cube = np.zeros((11, 11, 1))
    cube[5, 5, 0] = 100.0
    builder = partial(psf.ao_corrected_psf_proxy, strehl=strehl,
                      telescope_diameter_m=diameter_m,
                      seeing_fwhm_mas=seeing_mas)
    result = psf.convolve_cube_with_psf(
        cube, np.array([wavelength_um]), builder, spaxel_mas,
    )

    core_sigma = psf.diffraction_fwhm_mas(wavelength_um, diameter_m) / (
        2 * np.sqrt(2 * np.log(2))
    )
    halo_sigma = seeing_mas / (2 * np.sqrt(2 * np.log(2)))
    central_fraction = (
        strehl * erf(spaxel_mas / (2 * np.sqrt(2) * core_sigma)) ** 2
        + (1 - strehl) * erf(spaxel_mas / (2 * np.sqrt(2) * halo_sigma)) ** 2
    )
    assert result[5, 5, 0] == pytest.approx(100.0 * central_fraction, rel=2e-14)


def test_airy_fwhm_matches_diffraction_scale():
    lam, diam = 2.2, 8.0
    # Sample finely so the FWHM estimate is accurate.
    p = psf.airy_psf_proxy(lam, diam, grid_mas=2.0, n_pix=201)
    fwhm = psf.psf_fwhm_mas(p, 2.0)
    expected = psf.diffraction_fwhm_mas(lam, diam)
    assert fwhm == pytest.approx(expected, rel=0.15)


def test_higher_strehl_increases_encircled_energy():
    common = dict(lambda_um=2.2, telescope_diameter_m=8.0, seeing_fwhm_mas=600.0,
                  grid_mas=4.0, n_pix=161)
    lo = psf.ao_corrected_psf_proxy(strehl=0.05, **common)
    hi = psf.ao_corrected_psf_proxy(strehl=0.7, **common)
    ee1_lo, ee3_lo, _ = psf.encircled_energy_per_spaxel(lo, 4.0, 20.0)
    ee1_hi, ee3_hi, _ = psf.encircled_energy_per_spaxel(hi, 4.0, 20.0)
    assert ee1_hi > ee1_lo
    assert ee3_hi > ee3_lo
    for ee in (ee1_lo, ee3_lo, ee1_hi, ee3_hi):
        assert 0.0 <= ee <= 1.0


def test_convolution_conserves_flux_when_the_field_contains_the_psf():
    from functools import partial

    cube = np.zeros((11, 11, 3))
    cube[5, 5, :] = 100.0  # a point source in every plane (well inside the field)
    wl = np.array([2.0, 2.2, 2.4])
    # a PSF that is compact on the 220 mas field: 30 mas seeing, 8 m telescope
    builder = partial(psf.ao_corrected_psf_proxy, strehl=0.3,
                      telescope_diameter_m=8.0, seeing_fwhm_mas=30.0)
    out = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=20.0)
    for il in range(3):
        assert out[:, :, il].sum() == pytest.approx(cube[:, :, il].sum(), rel=1e-3)


def test_convolution_loses_exactly_the_out_of_field_halo():
    """Regression for the truncated-kernel artifact: a 400 mas halo on a
    220 mas field must lose the light that leaves the field, not have it
    folded back into the core by a renormalised small kernel."""
    from functools import partial

    cube = np.zeros((11, 11, 1))
    cube[5, 5, 0] = 100.0
    wl = np.array([2.2])
    builder = partial(psf.ao_corrected_psf_proxy, strehl=0.3,
                      telescope_diameter_m=8.0, seeing_fwhm_mas=400.0)
    out = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=20.0)
    # independent reference: the PSF energy inside the 11x11 field, from a
    # grid that contains the whole halo
    big = psf.ao_corrected_psf_proxy(0.3, 2.2, 8.0, 400.0, grid_mas=20.0, n_pix=301)
    c = 301 // 2
    in_field = np.sum(big[c - 5 : c + 6, c - 5 : c + 6])
    assert 0.3 < in_field < 0.7
    assert out.sum() == pytest.approx(100.0 * in_field, rel=1e-9)
    # the default kernel reaches every field pixel: enlarging it changes nothing
    bigger = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=20.0, kernel_n_pix=61)
    np.testing.assert_allclose(bigger, out, rtol=1e-12)
    # an explicit 9-spaxel kernel now discards in-field light instead of
    # inventing throughput
    small = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=20.0, kernel_n_pix=9)
    assert small.sum() < out.sum()


def test_convolution_flux_accounting_at_field_edges():
    """An edge-filling source must keep its flux with conserve_flux=True, and a
    plain zero-boundary convolution must visibly lose flux at the edges."""
    from functools import partial

    # A uniform field touches all four edges -- the hardest case for flux loss.
    cube = np.ones((12, 12, 1)) * 50.0
    wl = np.array([2.2])
    # A broad, low-Strehl PSF spreads a lot of light, exaggerating any edge loss.
    builder = partial(psf.ao_corrected_psf_proxy, strehl=0.1,
                      telescope_diameter_m=8.0, seeing_fwhm_mas=800.0)

    conserved = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=40.0,
                                           kernel_n_pix=15, conserve_flux=True)
    raw = psf.convolve_cube_with_psf(cube, wl, builder, spaxel_scale_mas=40.0,
                                     kernel_n_pix=15, conserve_flux=False)
    in_flux = cube.sum()
    # Edge-normalised convolution conserves the field flux to high accuracy.
    assert conserved.sum() == pytest.approx(in_flux, rel=1e-3)
    # The raw zero-boundary convolution loses a non-trivial fraction at the edges.
    assert raw.sum() < 0.95 * in_flux


def test_metrics_flags_undersampling():
    # A narrow diffraction PSF sampled by a coarse spaxel -> undersampled.
    p = psf.airy_psf_proxy(1.0, 39.0, grid_mas=1.0, n_pix=201)  # tiny core (ELT)
    metrics = psf.psf_metrics(p, fine_scale_mas=1.0, spaxel_scale_mas=50.0)
    assert metrics.undersampled is True
    # A well-sampled case is not flagged.
    p2 = psf.seeing_gaussian_psf(800.0, grid_mas=10.0, n_pix=201)
    metrics2 = psf.psf_metrics(p2, fine_scale_mas=10.0, spaxel_scale_mas=50.0)
    assert metrics2.undersampled is False


def test_centroid_shift_small_for_centred_psf():
    p = psf.seeing_gaussian_psf(300.0, grid_mas=10.0, n_pix=99)
    m = psf.psf_metrics(p, 10.0, 30.0)
    assert m.centroid_shift_mas < 1.0  # symmetric PSF is centred


def test_airy_pixels_are_integrated_not_centre_sampled():
    # 10 mas pixels at 1 um on 39 m (lambda/D ~ 5.3 mas): centre sampling summed
    # to ~3 (a point source ~3x brighter). Pixel integrals stay a bounded
    # enclosed fraction and agree with a finely sampled grid binned to 10 mas.
    coarse = psf.airy_psf_proxy(1.0, 39.0, grid_mas=10.0, n_pix=41)
    assert 0.9 < coarse.sum() <= 1.0
    fine = psf.airy_psf_proxy(1.0, 39.0, grid_mas=10.0 / 21, n_pix=41 * 21)   # odd x odd stays centred
    binned = fine.reshape(41, 21, 41, 21).sum(axis=(1, 3))
    assert np.allclose(coarse, binned, rtol=0.02, atol=1e-5)
