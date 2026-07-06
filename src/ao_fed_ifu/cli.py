"""Command-line interface: ``ao-ifu`` (a portfolio-quality wrapper).

Subcommands wrap the same code the example scripts use.  Everything is offline,
CPU-only and writes to a configurable output directory (default
``figures``).  ``run`` and ``validate`` use the package directly; the
figure-suite subcommands load the matching ``examples/run_*.py`` generator (so
the CLI works from a source checkout without duplicating their plotting code).

Examples:
    ao-ifu validate
    ao-ifu run --mode toy --out figures
    ao-ifu run --mode highz --out figures
    ao-ifu public-data --out figures --quick
    ao-ifu compare --out figures
    ao-ifu systematics --out figures
    ao-ifu sky-telluric --out figures
    ao-ifu reconstruct-demo --out figures
    ao-ifu benchmark --out figures
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

DEFAULT_OUT = "figures"
RUN_MODES = ("toy", "instrument_inspired", "public_conditioned", "highz")


def _load_example(name: str):
    """Import an ``examples/<name>.py`` figure generator from the source tree.

    The figure-suite subcommands reuse the example plotting code, which lives
    outside the installed package.  ``run`` and ``validate`` are package-native
    and do not call this; the figure suites require a **source checkout** (the
    repository's ``examples/`` directory).  Raises a clear ``SystemExit`` if the
    generator cannot be found.
    """
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    path = examples_dir / f"{name}.py"
    if not path.exists():
        raise SystemExit(
            f"figure-suite generator '{name}' not found at {path}.\n"
            "The figure-suite subcommands (public-data, compare, systematics, "
            "sky-telluric, reconstruct-demo, throughput, ao-psf, observing, detector, "
            "geometry, sky-variability, galaxy, benchmark) require running 'ao-ifu' from "
            "a source checkout containing examples/. The core 'run' and 'validate' "
            "subcommands work from any install."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _print_paths(title: str, paths) -> None:
    print(title)
    for key, p in (paths.items() if isinstance(paths, dict) else enumerate(paths)):
        print(f"  {key} -> {p}")


def _cmd_run(args) -> int:
    # Fully package-native (no dependency on examples/): every mode runs the
    # pipeline and writes standard artifacts. The richer high-z *science* figures
    # live in examples/run_science_case_highz_galaxy.py.
    from .integration import (
        make_highz_galaxy_config, make_instrument_inspired_config,
        make_public_conditioned_config, make_toy_config, run_pipeline, write_artifacts,
    )

    out = Path(args.out)
    builders = {
        "toy": make_toy_config,
        "instrument_inspired": make_instrument_inspired_config,
        "public_conditioned": make_public_conditioned_config,
        "highz": make_highz_galaxy_config,
    }
    if args.mode not in builders:
        raise SystemExit(f"unknown --mode {args.mode!r}; choose from {RUN_MODES}")
    cfg = builders[args.mode]()
    if args.quick:
        cfg.nx = cfg.ny = 10
        cfg.resolving_power_R = min(cfg.resolving_power_R, 2000.0)
    result = run_pipeline(cfg)
    paths = write_artifacts(result, out, prefix=args.mode)
    _print_paths(f"{args.mode} artifacts:", paths)
    return 0


def _cmd_example(example_name: str, fn_name: str):
    def runner(args) -> int:
        mod = _load_example(example_name)
        fn = getattr(mod, fn_name)
        paths = fn(Path(args.out), quick=args.quick)
        _print_paths(f"{example_name} artifacts:", paths)
        return 0

    return runner


def _cmd_validate(args) -> int:
    from .validation import run_all_validations

    results = run_all_validations()
    n_pass = sum(1 for r in results if r.passed)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    print(f"{n_pass}/{len(results)} internal validation checks passed")
    return 0 if n_pass == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ao-ifu", description="AO-fed IFU spectrograph demonstrator CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_out(p, quick=True):
        p.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: figures)")
        if quick:
            p.add_argument("--quick", action="store_true", help="small/fast run")

    p_run = sub.add_parser("run", help="run one pipeline mode and write artifacts")
    p_run.add_argument("--mode", default="toy", choices=RUN_MODES)
    add_out(p_run)
    p_run.set_defaults(func=_cmd_run)

    for cmd, example, help_text in [
        ("public-data", "run_public_data_figures", "public-data figures + tables"),
        ("compare", "run_mode_comparison_dashboard", "mode comparison dashboard"),
        ("systematics", "run_systematics_report", "instrument-systematics report"),
        ("sky-telluric", "run_sky_telluric_residuals", "sky/telluric residual experiment"),
        ("reconstruct-demo", "run_reconstruction_limitations", "reconstruction-method comparison"),
        ("throughput", "run_throughput_budget_report", "throughput budget waterfall"),
        ("ao-psf", "run_ao_psf_wavelength_report", "wavelength-dependent AO PSF report"),
        ("observing", "run_observing_strategy_report", "multi-exposure dither/coadd report"),
        ("detector", "run_detector_systematics_report", "detector systematics report"),
        ("geometry", "run_geometry_systematics_report", "spectrograph/IFU geometry report"),
        ("sky-variability", "run_sky_variability_report", "sky/telluric variability report"),
        ("galaxy", "run_galaxy_source_report", "high-z galaxy source + recovery report"),
        ("benchmark", "run_benchmark_validation_report", "benchmark trend-validation report"),
    ]:
        p = sub.add_parser(cmd, help=help_text)
        add_out(p)
        fn = "generate_all" if example == "run_public_data_figures" else "generate"
        p.set_defaults(func=_cmd_example(example, fn))

    p_val = sub.add_parser("validate", help="run internal validation checks")
    p_val.set_defaults(func=_cmd_validate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
