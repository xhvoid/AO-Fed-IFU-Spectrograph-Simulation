"""Tests for cube reconstruction (inverse mapping) and flux conservation."""

import numpy as np
import pytest

from ao_fed_ifu import cube_reconstruction as recon
from ao_fed_ifu.detector_model import DetectorConfig, expose
from ao_fed_ifu.ifu_mapping import IFUConfig, build_slicer_mapping, forward_project_cube
from ao_fed_ifu.source_cube import SourceCubeConfig, create_emission_line_cube, scale_cube_to_total_electrons
from ao_fed_ifu.spectrograph_model import SpectrographConfig


def _pipeline_setup(nx=6, ny=5, cross_talk=0.0):
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    cfg = SourceCubeConfig(nx=nx, ny=ny, nlambda=spectro.n_detector_pixels,
                           lambda_min_um=2.0, lambda_max_um=2.3)
    ifu = IFUConfig(nx_spaxel=nx, ny_spaxel=ny, cross_talk_sigma_pix=cross_talk)
    cube = create_emission_line_cube(cfg, line_flux_rel=5.0, velocity_gradient_kms_per_spaxel=10.0)
    cube = scale_cube_to_total_electrons(cube, 1.0e5)
    mapping = build_slicer_mapping(ifu, spectro, cfg.wavelength_grid_um)
    return cfg, cube, mapping


def test_nearest_reconstruction_is_exact_inverse():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    assert rebuilt.shape == cube.shape
    # With no LSF/cross-talk the round trip is exact.
    assert np.allclose(rebuilt, cube, rtol=1e-9, atol=1e-6)


def test_flux_conservation_ratio_unity_noiseless():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    rebuilt = recon.reconstruct_cube_nearest(frame, mapping)
    fc = recon.flux_conservation(cube, rebuilt)
    assert fc["ratio"] == pytest.approx(1.0, abs=1e-9)


def test_weighted_equals_nearest_when_no_crosstalk():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.0)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    var = np.ones_like(frame)
    nearest = recon.reconstruct_cube_nearest(frame, mapping)
    weighted = recon.reconstruct_cube_weighted(frame, var, mapping, profile_sigma_pix=0.0)
    assert np.allclose(nearest, weighted)


def test_extract_1d_spectrum_sums_aperture():
    cfg, cube, mapping = _pipeline_setup()
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    mask[1:3, 1:3] = True
    spec = recon.extract_1d_spectrum(cube, mask)
    assert spec.shape == (cfg.nlambda,)
    assert np.allclose(spec, cube[1:3, 1:3, :].sum(axis=(0, 1)))


def test_bad_voxels_excluded_from_conservation():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    cfgd = DetectorConfig(read_noise_e=0.0, dark_current_e_s_pix=0.0, bad_pixel_fraction=0.05,
                          flat_field_sigma=0.0)
    out = expose(frame, cfgd, exposure_s=1.0, rng=0)
    rebuilt = recon.reconstruct_cube_nearest(out.image_e, mapping)
    bad_vox = recon.bad_voxel_mask(out.bad_pixel_mask, mapping)
    fc = recon.flux_conservation(cube, rebuilt, bad_voxels=bad_vox)
    # Excluding the masked voxels, the (Poisson-only) flux is conserved on average.
    assert fc["ratio"] == pytest.approx(1.0, abs=0.05)


def test_variance_cube_shape_matches_cube():
    cfg, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    out = expose(frame, DetectorConfig(), exposure_s=1.0, rng=0)
    var_cube = recon.estimate_variance_cube(out.variance_e2, mapping)
    assert var_cube.shape == cube.shape
    assert np.all(var_cube > 0)


def test_weighted_variance_matches_horne_formula():
    # For a constant detector variance V0 and a normalised profile P, the Horne
    # variance is Var(f) = 1 / sum(P^2 / V0) = V0 / sum(P^2) at every voxel.
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    v0 = 25.0
    var_frame = np.full_like(frame, v0)
    flux, var = recon.reconstruct_cube_weighted_with_variance(frame, var_frame, mapping)

    sigma = mapping.ifu.cross_talk_sigma_pix
    half = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-half, half + 1)
    profile = np.exp(-0.5 * (offsets / sigma) ** 2)
    profile = profile / profile.sum()
    expected = v0 / np.sum(profile**2)

    assert var.shape == cube.shape
    # Interior spaxels see the full window; detector-edge traces use a truncated
    # and renormalised profile, so check a central spaxel exactly and only assert
    # the cube-wide variance envelope.
    assert var[2, 3, 5] == pytest.approx(expected, rel=1e-9)
    assert np.all(var > v0)  # the wider window always carries more noise than one pixel


def test_weighted_variance_equals_nearest_for_sigma_zero():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.0)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=False)
    out = expose(frame, DetectorConfig(), exposure_s=1.0, rng=0)
    flux, var_w = recon.reconstruct_cube_weighted_with_variance(
        out.image_e, out.variance_e2, mapping, profile_sigma_pix=0.0
    )
    var_n = recon.estimate_variance_cube(out.variance_e2, mapping)
    # Single-row window: the weighted estimator degenerates to nearest exactly.
    assert np.allclose(var_w, var_n)
    assert np.allclose(flux, recon.reconstruct_cube_nearest(out.image_e, mapping))


def test_estimate_variance_cube_weighted_matches_combined():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=True, apply_cross_talk=True)
    out = expose(frame, DetectorConfig(), exposure_s=1.0, rng=0)
    _, var_combined = recon.reconstruct_cube_weighted_with_variance(
        out.image_e, out.variance_e2, mapping
    )
    var_standalone = recon.estimate_variance_cube_weighted(out.variance_e2, mapping)
    # Both paths evaluate the denominator with the identical sum(P^2 * (1/V))
    # expression, so the result is bit-for-bit identical, not merely close.
    assert np.array_equal(var_combined, var_standalone)


def test_estimate_variance_cube_weighted_bit_identical_random():
    # Reproduces the review scenario (dense window, arbitrary finite variance):
    # the stand-alone variance path must be bit-identical to the flux path's.
    cfg, cube, mapping = _pipeline_setup(cross_talk=1.0)
    n_rows, n_cols = mapping.det_shape
    rng = np.random.default_rng(0)
    det = rng.normal(100.0, 10.0, (n_rows, n_cols))
    var = rng.uniform(5.0, 20.0, (n_rows, n_cols))
    _, var_combined = recon.reconstruct_cube_weighted_with_variance(det, var, mapping)
    var_standalone = recon.estimate_variance_cube_weighted(var, mapping)
    assert np.array_equal(var_combined, var_standalone)


def test_weighted_variance_infinite_when_window_all_bad():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    var_frame = np.full_like(frame, np.inf)  # every pixel bad
    flux, var = recon.reconstruct_cube_weighted_with_variance(frame, var_frame, mapping)
    assert np.all(np.isinf(var))
    assert np.all(flux == 0.0)


def test_compare_methods_carries_matching_variance():
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=True, apply_cross_talk=True)
    out = expose(frame, DetectorConfig(bad_pixel_fraction=0.01), exposure_s=1.0, rng=0)
    methods = recon.compare_reconstruction_methods(
        out.image_e, out.variance_e2, out.bad_pixel_mask, mapping, cube
    )
    for name, entry in methods.items():
        assert entry["variance_cube"].shape == cube.shape, name
    # The weighted variance is the Horne propagation, not the nearest look-up.
    finite_w = np.isfinite(methods["weighted"]["variance_cube"])
    finite_n = np.isfinite(methods["nearest"]["variance_cube"])
    both = finite_w & finite_n
    assert np.median(methods["weighted"]["variance_cube"][both]) > np.median(
        methods["nearest"]["variance_cube"][both]
    )


# ---------------------------------------------------------------------------
# Detector-boundary handling for the weighted (Horne) extraction: off-detector
# window rows are discarded (not clipped onto the edge row), with the profile
# renormalised over the surviving rows.
# ---------------------------------------------------------------------------

def _edge_mapping(nx=4, ny=4, cross_talk=0.6):
    """Mapping whose first and last spaxels sit on the detector edge rows.

    ``slice_gap_pix=0`` removes the trailing gap so the last spaxel lands on
    the final detector row, giving a genuine bottom-edge trace to test.
    """
    spectro = SpectrographConfig(lambda_min_um=2.0, lambda_max_um=2.3, resolving_power_R=2000.0)
    ifu = IFUConfig(nx_spaxel=nx, ny_spaxel=ny, cross_talk_sigma_pix=cross_talk, slice_gap_pix=0)
    grid = np.linspace(2.0, 2.3, spectro.n_detector_pixels)
    return build_slicer_mapping(ifu, spectro, grid)


def _renorm_valid_profile(r0, offsets, full_profile, n_rows):
    """Independent reference: profile subset on-detector, renormalised to 1."""
    raw = r0 + offsets
    valid = (raw >= 0) & (raw < n_rows)
    p = full_profile[valid]
    return p / p.sum()


def test_weighted_edge_traces_discard_offdetector_rows():
    mapping = _edge_mapping()
    n_rows, n_cols = mapping.det_shape
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row = mapping.det_row_of_spaxel.reshape(ny, nx)
    assert det_row.min() == 0 and det_row.max() == n_rows - 1  # both edges present

    # Constant detector + variance => Horne flux = C/sum(p^2), var = V/sum(p^2),
    # where p is the (edge-truncated, renormalised) profile for that trace.
    C, V = 7.0, 4.0
    det = np.full((n_rows, n_cols), C)
    var = np.full((n_rows, n_cols), V)
    flux, fvar = recon.reconstruct_cube_weighted_with_variance(det, var, mapping)

    sigma = mapping.ifu.cross_talk_sigma_pix
    half = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-half, half + 1)
    full = np.exp(-0.5 * (offsets / sigma) ** 2)
    full = full / full.sum()

    # An interior trace (whole window on-detector) is unchanged by the edge fix.
    iy_int, ix_int = 1, 1
    assert det_row[iy_int, ix_int] - half >= 0 and det_row[iy_int, ix_int] + half < n_rows
    s2_full = np.sum(full**2)
    assert flux[iy_int, ix_int, 0] == pytest.approx(C / s2_full, rel=1e-9)
    assert fvar[iy_int, ix_int, 0] == pytest.approx(V / s2_full, rel=1e-9)

    # Top-edge (r0=0) and bottom-edge (r0=n_rows-1) traces: only the on-detector
    # rows contribute, so flux/variance match the truncated-profile reference and
    # DIFFER from the interior value (clipping would have made them equal).
    for iy, ix in (tuple(np.argwhere(det_row == 0)[0]),
                   tuple(np.argwhere(det_row == n_rows - 1)[0])):
        r0 = int(det_row[iy, ix])
        p = _renorm_valid_profile(r0, offsets, full, n_rows)
        s2 = float(np.sum(p**2))
        assert p.size == half + 1  # exactly the on-detector half of the window
        assert flux[iy, ix, 0] == pytest.approx(C / s2, rel=1e-9)
        assert fvar[iy, ix, 0] == pytest.approx(V / s2, rel=1e-9)
        assert not np.isclose(flux[iy, ix, 0], C / s2_full)  # not the interior value


def test_weighted_edge_row_not_duplicated():
    # Put signal in ONLY the top edge row; a boundary trace must count it once.
    mapping = _edge_mapping()
    n_rows, n_cols = mapping.det_shape
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row = mapping.det_row_of_spaxel.reshape(ny, nx)
    iy, ix = tuple(np.argwhere(det_row == 0)[0])

    A = 100.0
    det = np.zeros((n_rows, n_cols))
    det[0, :] = A                       # only the edge row carries flux
    var = np.ones((n_rows, n_cols))
    flux, _ = recon.reconstruct_cube_weighted_with_variance(det, var, mapping)

    sigma = mapping.ifu.cross_talk_sigma_pix
    half = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-half, half + 1)
    full = np.exp(-0.5 * (offsets / sigma) ** 2)
    full = full / full.sum()
    p = _renorm_valid_profile(0, offsets, full, n_rows)   # rows 0..half
    # Only row 0 is non-zero: numer = p[0]*A, denom = sum(p^2).
    expected = p[0] * A / float(np.sum(p**2))
    assert flux[iy, ix, 0] == pytest.approx(expected, rel=1e-9)
    # The clipped (buggy) estimator would read row 0 at every non-positive offset,
    # summing several profile lobes of the same pixel -> a strictly larger numer.
    buggy_numer = np.sum(full[offsets <= 0]) * A
    assert buggy_numer > p[0] * A
