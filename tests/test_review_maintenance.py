"""Focused regression checks for example estimator pairing and noise proxies."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from ao_fed_ifu import cube_reconstruction as recon


def test_notebook_four_pairs_joint_flux_with_matching_variance():
    path = Path(__file__).resolve().parents[1] / "notebooks/04_cube_reconstruction_noise_snr.ipynb"
    notebook = json.loads(path.read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    tree = ast.parse(code)
    paired = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "estimate_variance_cube"
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "reconstruct_cube_weighted_with_variance"):
            paired.append(node)
    assert len(paired) == 1
    assert isinstance(paired[0].targets[0], ast.Tuple)
    assert [name.id for name in paired[0].targets[0].elts] == ["rebuilt", "var_cube"]
    # The extraction must consume the CALIBRATED readout (bias/dark/response
    # removed), not the raw detector frame.
    calibrated = [node.targets[0].id for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                  and getattr(node.value.func, "id", None) == "calibrate_readout"]
    assert len(calibrated) == 1
    args = paired[0].value.args
    assert [ast.unparse(a) for a in args[:2]] == [f"{calibrated[0]}.image_e", f"{calibrated[0]}.variance_e2"]
    # Line flux shares the variance mask used by the SNR map.
    flux_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and getattr(node.func, "attr", None) == "line_flux_map"]
    assert flux_calls and all(
        any(k.arg == "variance_cube" and ast.unparse(k.value) == "var_cube" for k in call.keywords)
        for call in flux_calls)


def test_notebook_four_executes(tmp_path, monkeypatch):
    # Run the notebook's code from a scratch tree so its figures never land in
    # the repository; it must execute end to end with the current API.
    import matplotlib

    matplotlib.use("Agg")
    path = Path(__file__).resolve().parents[1] / "notebooks/04_cube_reconstruction_noise_snr.ipynb"
    notebook = json.loads(path.read_text())
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")
    namespace: dict = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    assert np.isfinite(namespace["snr"]).any()
    assert np.array_equal(np.isnan(namespace["lf"]), np.isnan(namespace["snr"]))


def test_strong_smooth_line_is_not_mistaken_for_noise_correlation():
    rng = np.random.default_rng(39)
    channel = np.arange(2048)
    truth = np.broadcast_to(1000 + 300 * np.exp(-0.5 * ((channel - 900) / 75)**2),
                            (4, 4, channel.size))
    cube = truth + rng.normal(size=truth.shape)
    assert recon.spectral_correlation_proxy(cube) > 0.99
    residual_rho = recon.spectral_noise_correlation_proxy(cube, reference_cube=truth)
    assert abs(residual_rho) < 0.025
    acov = recon.estimate_covariance_proxy(cube, reference_cube=truth)
    assert acov[0] == pytest.approx(1.0)
    assert acov[1] == pytest.approx(residual_rho)
    assert np.max(np.abs(acov[1:])) < 0.025


def test_noise_proxy_recovers_correlated_residuals():
    rng = np.random.default_rng(49)
    noise = rng.normal(size=(4, 4, 4096))
    for i in range(1, noise.shape[-1]):
        noise[:, :, i] += 0.65 * noise[:, :, i - 1]
    assert recon.spectral_noise_correlation_proxy(
        noise, reference_cube=np.zeros_like(noise)
    ) == pytest.approx(0.65, abs=0.02)


def test_line_free_detrending_does_not_fit_or_include_the_line():
    rng = np.random.default_rng(64)
    channel = np.arange(1000)
    cube = (40 + 0.1 * channel + rng.normal(size=(3, 3, 1000)))
    mask = (channel < 450) | (channel > 550)
    with_line = cube.copy()
    with_line[:, :, ~mask] += 1.e8
    rho = recon.spectral_noise_correlation_proxy(cube, line_free_mask=mask)
    assert recon.spectral_noise_correlation_proxy(with_line, line_free_mask=mask) == rho
    assert abs(rho) < 0.04


def test_excluded_channels_never_become_artificial_neighbours():
    rng = np.random.default_rng(72)
    cube = rng.normal(size=(2, 2, 100))
    valid = np.arange(100) % 2 == 0
    reference = np.zeros_like(cube)
    assert np.isnan(recon.spectral_noise_correlation_proxy(
        cube, reference_cube=reference, valid_mask=valid
    ))
    acov = recon.estimate_covariance_proxy(cube, reference_cube=reference, valid_mask=valid)
    assert np.isnan(acov[1])
    assert np.isfinite(acov[2])


def test_noise_proxy_requires_a_signal_removal_definition():
    with pytest.raises(ValueError, match="reference_cube or line_free_mask"):
        recon.spectral_noise_correlation_proxy(np.zeros((1, 1, 10)))
