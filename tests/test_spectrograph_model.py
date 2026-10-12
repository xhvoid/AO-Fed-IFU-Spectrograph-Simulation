"""Spectrograph wavelength solution and LSF conservation."""

import numpy as np
import pytest

from ao_fed_ifu.spectrograph_model import SpectrographConfig


def test_lsf_conserves_narrow_feature_on_irregular_native_grid():
    # An interior line of area 1e-6 between closely spaced irregular knots must
    # survive the LSF: point-sampling the interpolant would miss it entirely.
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=3000.0)
    wavelength = np.array([2.0, 2.099999, 2.1, 2.100001, 2.2, 2.3])
    density = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with_lsf = spectro.integrate_spectral_density(wavelength, density, apply_lsf=True)
    without = spectro.integrate_spectral_density(wavelength, density, apply_lsf=False)
    assert without.sum() == pytest.approx(1e-6, rel=1e-6)
    assert with_lsf.sum() == pytest.approx(1e-6, rel=1e-6)


@pytest.mark.parametrize("fraction_of_fine_cell", [0.0, 0.3, 0.5, 0.8])
def test_irregular_grid_lsf_matches_analytic_profile_and_centroid(fraction_of_fine_cell):
    # A line narrower than the internal fine cell must keep its centroid (no
    # snapping to fine nodes): compare with the exact Gaussian bin integrals.
    from scipy.special import ndtr

    from ao_fed_ifu.spectrograph_model import wavelength_bin_edges

    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=3000.0)
    grid = spectro.wavelength_grid_um()
    disp = spectro.dispersion_um_per_pix
    mu = 2.1 + fraction_of_fine_cell * disp / 16
    wavelength = np.array([2.0, mu - 1e-6, mu, mu + 1e-6, 2.2, 2.3])
    density = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    out = spectro.integrate_spectral_density(wavelength, density)
    edges = wavelength_bin_edges(grid)
    sigma = spectro.lsf_sigma_pix * disp
    reference = 1e-6 * (ndtr((edges[1:] - mu) / sigma) - ndtr((edges[:-1] - mu) / sigma))
    centroid = np.sum(grid * out) / out.sum()
    assert abs(centroid - np.sum(grid * reference) / reference.sum()) < 1e-3 * disp
    assert np.max(np.abs(out - reference)) < 2e-3 * reference.max()


def test_wavelength_offset_keeps_detector_geometry_and_dispersion():
    base = SpectrographConfig(lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=3000.0)
    shifted = base.with_wavelength_offset(0.001)
    assert shifted.n_detector_pixels == base.n_detector_pixels
    assert shifted.dispersion_um_per_pix == pytest.approx(base.dispersion_um_per_pix, rel=1e-12)
    assert np.allclose(shifted.wavelength_grid_um() - base.wavelength_grid_um(), 0.001)


def test_dispersion_stretch_is_the_requested_fraction():
    base = SpectrographConfig(lambda_min_um=1.95, lambda_max_um=2.45, resolving_power_R=3000.0)
    stretched = base.with_dispersion_stretch(0.01)
    assert stretched.n_detector_pixels == base.n_detector_pixels
    assert stretched.dispersion_um_per_pix / base.dispersion_um_per_pix == pytest.approx(1.01, rel=1e-12)


def test_detector_pixels_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="detector_pixels"):
        SpectrographConfig(detector_pixels=0)


@pytest.mark.parametrize("width_um", [1e-6, 1e-4, 1e-2])
def test_uniform_grid_with_nonzero_endpoints_conserves_area(width_um):
    # A uniform native grid whose density is nonzero at its endpoints (a step to
    # zero) must not gain flux from ramps outside its support (it used to double
    # for a 1e-6 um interval).
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    wavelength = np.linspace(2.1, 2.1 + width_um, 2 if width_um < 1e-3 else 101)
    density = np.ones_like(wavelength)
    without = spectro.integrate_spectral_density(wavelength, density, apply_lsf=False)
    with_lsf = spectro.integrate_spectral_density(wavelength, density)
    assert without.sum() == pytest.approx(width_um, rel=1e-6)
    assert with_lsf.sum() == pytest.approx(width_um, rel=1e-6)


def test_uniform_grid_endpoint_step_keeps_its_centroid():
    # An asymmetric single-cell feature (1 -> 0.2): compare with the exact
    # Gaussian-convolved, bin-integrated linear segment.
    from numpy.polynomial.legendre import leggauss
    from scipy.special import ndtr

    from ao_fed_ifu.spectrograph_model import wavelength_bin_edges

    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    a, b, fa, fb = 2.1, 2.1 + 2e-5, 1.0, 0.2
    out = spectro.integrate_spectral_density(np.array([a, b]), np.array([fa, fb]))
    grid = spectro.wavelength_grid_um()
    edges = wavelength_bin_edges(grid)
    sigma = spectro.lsf_sigma_pix * spectro.dispersion_um_per_pix
    nodes, weights = leggauss(64)
    y = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    f = fa + (fb - fa) * (y - a) / (b - a)
    kernel = ndtr((edges[1:, None] - y) / sigma) - ndtr((edges[:-1, None] - y) / sigma)
    reference = 0.5 * (b - a) * (kernel * (f * weights)).sum(axis=1)
    assert out.sum() == pytest.approx(reference.sum(), rel=1e-6)
    centroid = np.sum(grid * out) / out.sum()
    assert abs(centroid - np.sum(grid * reference) / reference.sum()) < 1e-3 * spectro.dispersion_um_per_pix
