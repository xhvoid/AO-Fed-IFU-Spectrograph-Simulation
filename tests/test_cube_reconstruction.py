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


def _dense_design_matrix(mapping, profile_sigma_pix, lambda_index=0):
    """Independent reference: dense ``A[row, spaxel]`` of one detector column."""
    from ao_fed_ifu.ifu_mapping import cross_talk_kernel

    offsets, kernel = cross_talk_kernel(profile_sigma_pix)
    n_rows = mapping.det_shape[0]
    n_spax = mapping.det_row_of_spaxel.size
    shift = int(mapping.lambda_row_shift()[lambda_index])
    A = np.zeros((n_rows, n_spax))
    for sid in range(n_spax):
        for off, k in zip(offsets, kernel):
            r = int(mapping.det_row_of_spaxel[sid]) + shift + int(off)
            if 0 <= r < n_rows:
                A[r, sid] += k
    return A


def test_weighted_extraction_matches_dense_least_squares():
    """flux = (A^T W A)^-1 A^T W d and var = diag((A^T W A)^-1), checked
    against a dense NumPy solve built independently of the banded code."""
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    rng = np.random.default_rng(3)
    det = frame + rng.normal(0.0, 3.0, frame.shape)
    var = rng.uniform(4.0, 16.0, frame.shape)
    flux, fvar = recon.reconstruct_cube_weighted_with_variance(det, var, mapping)
    ny, nx = cube.shape[:2]
    for il in (0, 7):
        col = mapping.col_of_lambda[il]
        A = _dense_design_matrix(mapping, 0.6, il)
        W = np.diag(1.0 / var[:, col])
        cov = np.linalg.inv(A.T @ W @ A)
        f = cov @ A.T @ W @ det[:, col]
        np.testing.assert_allclose(flux[:, :, il].ravel(), f, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(fvar[:, :, il].ravel(), np.diag(cov), rtol=1e-7)


def test_weighted_extraction_inverts_crosstalk_exactly_and_does_not_double_count():
    """Noiseless: the joint estimator is the exact inverse of the cross-talk
    blur (ratio 1, every voxel), while the per-trace Horne window amplifies a
    uniform field by 1/sum(P^2) ~ 2.02 -- the bias it is kept to demonstrate."""
    from ao_fed_ifu.ifu_mapping import cross_talk_kernel

    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    var = np.full_like(frame, 25.0)
    flux, _ = recon.reconstruct_cube_weighted_with_variance(frame, var, mapping)
    # exact data-only inverse (no interpolation prior or ridge)
    np.testing.assert_allclose(flux, cube, rtol=1e-8, atol=1e-8)
    assert recon.flux_conservation(cube, flux)["ratio"] == pytest.approx(1.0, abs=1e-9)

    # Uniform field on a gap-free slicer (every detector row lit except the
    # two outer edges): the joint estimate is exactly uniform, the per-trace
    # Horne window returns 1/sum(P^2) for every trace whose whole window is
    # on lit rows.
    gapless = _edge_mapping(nx=6, ny=6)
    ny, nx = gapless.ifu.ny_spaxel, gapless.ifu.nx_spaxel
    uniform = np.full((ny, nx, gapless.det_shape[1]), 100.0)
    uframe = forward_project_cube(uniform, gapless, apply_lsf=False, apply_cross_talk=True)
    uvar = np.full_like(uframe, 25.0)
    joint, _ = recon.reconstruct_cube_weighted_with_variance(uframe, uvar, gapless)
    horne, _ = recon.reconstruct_cube_horne_per_trace_with_variance(uframe, uvar, gapless)
    offsets, kernel = cross_talk_kernel(0.6)
    radius = int(offsets.max())
    rows = gapless.det_row_of_spaxel.reshape(ny, nx)
    lit = (rows >= 2 * radius) & (rows <= gapless.det_shape[0] - 1 - 2 * radius)
    np.testing.assert_allclose(joint, 100.0, rtol=1e-8)
    np.testing.assert_allclose(
        horne[lit] / 100.0, 1.0 / np.sum(kernel**2), rtol=1e-9
    )
    assert 1.0 / np.sum(kernel**2) > 2.0


def test_weighted_variance_and_aperture_covariance_match_monte_carlo():
    cfg, cube, mapping = _pipeline_setup(nx=6, ny=6, cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    sigma = 4.0
    var = np.full_like(frame, sigma**2)
    _, fvar, model = recon.reconstruct_cube_joint_with_model(frame, var, mapping)
    rng = np.random.default_rng(11)
    n_draws = 400
    draws = np.stack([
        recon.reconstruct_cube_weighted(frame + rng.normal(0.0, sigma, frame.shape), var, mapping)
        for _ in range(n_draws)
    ])
    empirical = draws.var(axis=0, ddof=1)
    # per-voxel variance: pooled ratio within Monte-Carlo error
    assert float(np.mean(empirical / fvar)) == pytest.approx(1.0, abs=0.03)
    # aperture sum: the exact propagation (with spatial covariance) matches the
    # scatter of the summed draws; the independent-spaxel sum does not, because
    # deblended neighbours are anti-correlated
    aperture = np.zeros(cube.shape[:2], dtype=bool)
    aperture[1:5, 1:5] = True
    ap_draws = draws[:, aperture, :].sum(axis=1)
    exact = model.aperture_variance(aperture)
    diagonal = fvar[aperture, :].sum(axis=0)
    assert float(np.mean(ap_draws.var(axis=0, ddof=1) / exact)) == pytest.approx(1.0, abs=0.05)
    assert np.all(exact < 0.7 * diagonal)
    np.testing.assert_allclose(model.aperture_variance_diagonal(aperture), diagonal, rtol=1e-9)


def test_horne_per_trace_variance_matches_horne_formula():
    # For a constant detector variance V0 and a normalised profile P, the
    # per-trace Horne variance is Var(f) = 1 / sum(P^2 / V0) = V0 / sum(P^2).
    cfg, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    v0 = 25.0
    var_frame = np.full_like(frame, v0)
    flux, var = recon.reconstruct_cube_horne_per_trace_with_variance(frame, var_frame, mapping)

    sigma = mapping.ifu.cross_talk_sigma_pix
    half = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-half, half + 1)
    profile = np.exp(-0.5 * (offsets / sigma) ** 2)
    profile = profile / profile.sum()
    expected = v0 / np.sum(profile**2)

    assert var.shape == cube.shape
    assert var[2, 3, 5] == pytest.approx(expected, rel=1e-9)
    assert np.all(var > v0)


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


@pytest.mark.parametrize("masked_value", [np.nan, np.inf])
def test_standalone_weighted_variance_can_match_nonfinite_data_mask(masked_value):
    _, cube, mapping = _pipeline_setup(cross_talk=0.6)
    frame = forward_project_cube(cube, mapping)
    var = np.full_like(frame, 5.0)
    row = mapping.det_row_of_spaxel[3]
    frame[row, 2] = masked_value
    _, combined = recon.reconstruct_cube_weighted_with_variance(frame, var, mapping)
    standalone = recon.estimate_variance_cube_weighted(var, mapping, detector_e=frame)
    unaware = recon.estimate_variance_cube_weighted(var, mapping)
    assert np.array_equal(combined, standalone)
    assert not np.array_equal(combined, unaware)


def test_standalone_weighted_variance_rejects_mismatched_data_shape():
    _, cube, mapping = _pipeline_setup()
    frame = forward_project_cube(cube, mapping)
    with pytest.raises(ValueError, match="shape"):
        recon.estimate_variance_cube_weighted(np.ones_like(frame), mapping,
                                              detector_e=frame[:-1])


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
        unmeasured = ~np.isfinite(entry["variance_cube"])
        expected_measured = recon.flux_conservation(
            cube, entry["cube"], bad_voxels=unmeasured
        )["ratio"]
        assert entry["measured_flux_conservation_ratio"] == expected_measured
        assert entry["measured_voxel_fraction"] == pytest.approx(np.mean(~unmeasured))
    # The weighted variance is the Horne propagation, not the nearest look-up.
    finite_w = np.isfinite(methods["weighted"]["variance_cube"])
    finite_n = np.isfinite(methods["nearest"]["variance_cube"])
    both = finite_w & finite_n
    assert np.median(methods["weighted"]["variance_cube"][both]) > np.median(
        methods["nearest"]["variance_cube"][both]
    )


# ---------------------------------------------------------------------------
# Detector-boundary handling.  Joint estimator: off-detector kernel taps are
# simply absent from the model (the forward blur loses that light), so an edge
# trace's flux is estimated from its observed rows.  Per-trace Horne
# demonstrator: off-detector window rows are discarded (not clipped onto the
# edge row), with the profile renormalised over the surviving rows.
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


def test_joint_edge_traces_recover_truth_from_observed_rows():
    mapping = _edge_mapping()
    n_rows, n_cols = mapping.det_shape
    ny, nx = mapping.ifu.ny_spaxel, mapping.ifu.nx_spaxel
    det_row = mapping.det_row_of_spaxel.reshape(ny, nx)
    assert det_row.min() == 0 and det_row.max() == n_rows - 1  # both edges present
    rng = np.random.default_rng(5)
    cube = rng.uniform(20.0, 80.0, (ny, nx, n_cols))
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    # the forward blur genuinely loses edge light off the array ...
    assert frame.sum() < cube.sum()
    var = np.full_like(frame, 9.0)
    flux, fvar = recon.reconstruct_cube_weighted_with_variance(frame, var, mapping)
    # ... yet the least-squares model of the observed rows recovers the truth,
    # edge traces included.
    np.testing.assert_allclose(flux, cube, rtol=1e-8, atol=1e-8)
    # dense reference with the truncated design matrix agrees
    A = _dense_design_matrix(mapping, 0.6, 0)
    cov = np.linalg.inv(A.T @ A / 9.0)
    np.testing.assert_allclose(fvar[:, :, 0].ravel(), np.diag(cov), rtol=1e-7)


def test_horne_per_trace_edge_traces_discard_offdetector_rows():
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
    flux, fvar = recon.reconstruct_cube_horne_per_trace_with_variance(det, var, mapping)

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


def test_horne_per_trace_edge_row_not_duplicated():
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
    flux, _ = recon.reconstruct_cube_horne_per_trace_with_variance(det, var, mapping)

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


def _one_column_mapping(nx=4, cross_talk=0.6):
    spectro = SpectrographConfig(
        lambda_min_um=2.0, lambda_max_um=2.01, resolving_power_R=100.0,
        pixels_per_resolution_element=2.0,
    )
    assert spectro.n_detector_pixels == 1
    ifu = IFUConfig(
        nx_spaxel=nx, ny_spaxel=1, cross_talk_sigma_pix=cross_talk,
        slice_gap_pix=2,
    )
    return build_slicer_mapping(ifu, spectro, spectro.wavelength_grid_um())


def test_identifiable_bad_pixel_deblend_is_unbiased_and_covariance_is_data_only():
    """A missing trace-centre row need not destroy the measurement: for a short
    trace block the side taps retain full rank.  The data-only WLS solution must
    recover a sharp, nonuniform feature without a smoothness-prior imprint, and
    both voxel and aperture covariance must equal the dense inverse Gram matrix."""
    mapping = _one_column_mapping(nx=4)
    cube = np.array([0.0, 2.0, 100.0, 4.0]).reshape(1, 4, 1)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    var = np.full_like(frame, 7.0)
    var[1, 0] = np.inf
    frame[1, 0] = np.nan  # a common bad-pixel representation must also be harmless

    flux, fvar, model = recon.reconstruct_cube_joint_with_model(frame, var, mapping)
    A = _dense_design_matrix(mapping, 0.6, 0)
    good = np.isfinite(var[:, 0])
    W = np.diag(1.0 / var[good, 0])
    normal = A[good].T @ W @ A[good]
    cov = np.linalg.inv(normal)
    expected = cov @ A[good].T @ W @ frame[good, 0]

    assert np.linalg.matrix_rank(A[good]) == cube.size
    assert np.all(np.isfinite(fvar))
    np.testing.assert_allclose(flux.ravel(), expected, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(flux, cube, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(fvar.ravel(), np.diag(cov), rtol=2e-9)

    aperture = np.array([[True, False, True, True]])
    u = aperture.ravel().astype(float)
    assert model.aperture_variance(aperture)[0] == pytest.approx(u @ cov @ u, rel=2e-9)
    assert model.aperture_variance_diagonal(aperture)[0] == pytest.approx(
        np.sum(np.diag(cov)[aperture.ravel()]), rel=2e-9
    )


def test_ill_conditioned_bad_pixel_deblend_masks_noisy_neighbours_without_bias():
    """In a long adjacent run one missing central row produces a near-null mode.
    Coefficients above the declared variance-amplification threshold must be
    masked, not made to look precise by a spatial interpolation constraint; every
    coefficient still reported measured must retain the sharp-source truth."""
    mapping = _one_column_mapping(nx=20)
    cube = np.zeros((1, 20, 1))
    cube[0, 11, 0] = 100.0
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    var = np.ones_like(frame)
    var[10, 0] = np.inf

    flux, fvar = recon.reconstruct_cube_weighted_with_variance(frame, var, mapping)
    unmeasured = np.isinf(fvar)
    measured = ~unmeasured

    assert unmeasured[0, 10, 0]       # the trace centred on the bad row
    assert unmeasured[0, 11, 0]       # its bright, near-null-linked neighbour
    assert unmeasured.sum() > 1       # do not hide the coupling behind a prior
    assert measured.any()             # distant, identifiable traces survive
    assert np.all(flux[unmeasured] == 0.0)
    np.testing.assert_allclose(flux[measured], cube[measured], rtol=2e-4, atol=2e-4)


def test_rank_deficient_bad_pixel_column_masks_all_null_mode_participants():
    """A boundary bad-pixel cluster creates null modes whose coefficients decay
    rapidly but remain non-zero across the connected trace run.  Even the weak
    tail is not individually identifiable and must not be labelled measured.
    The pseudo-inverse is an internal solve only: every null-mode participant is
    returned as unmeasured, with no invented certainty."""
    mapping = _one_column_mapping(nx=20)
    cube = np.arange(1.0, 21.0).reshape(1, 20, 1)
    frame = forward_project_cube(cube, mapping, apply_lsf=False, apply_cross_talk=True)
    var = np.ones_like(frame)
    var[:3, 0] = np.inf

    flux, fvar, model = recon.reconstruct_cube_joint_with_model(frame, var, mapping)
    A = _dense_design_matrix(mapping, 0.6, 0)
    assert np.linalg.matrix_rank(A[3:]) < cube.size

    assert np.all(flux == 0.0)
    assert np.all(np.isinf(fvar))
    assert model.covariance_overrides.keys() == {0}
    assert np.isinf(model.aperture_variance(np.ones((1, 20), dtype=bool))[0])
