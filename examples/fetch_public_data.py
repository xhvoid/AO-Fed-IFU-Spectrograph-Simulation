#!/usr/bin/env python3
"""Fetch small public-data samples into ``data/public_cache/``.

This populates the optional cache with **real** public data so the pipeline can
report ``direct_public_data`` instead of synthetic proxies:

    * SVO Filter Profile Service  -> J.dat, H.dat, K.dat   (2MASS J/H/Ks curves)
    * ESO SkyCalc                 -> skycalc.dat           (sky radiance + telluric)
    * STScI CALSPEC               -> source.dat            (a flux-standard spectrum)

Design notes
------------
* This is a **one-time developer tool**, not part of the runtime path: the
  package's loaders only read the resulting ``*.dat`` files and never call this
  script.  Tests and examples still run fully offline.
* It needs network access, and the FITS sources (SkyCalc, CALSPEC) additionally
  need ``astropy`` (``pip install astropy skycalc_cli``).  Each source is fetched
  independently; a failure in one does not stop the others.
* Files are written as small whitespace-separated ASCII with a provenance header,
  in exactly the column format the loaders in ``data_sources.py`` expect.

Run:
    python examples/fetch_public_data.py
"""

from __future__ import annotations

import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[1] / "data" / "public_cache"
USER_AGENT = "AO-Fed-IFU-Spectrograph-Simulation/0.1 (portfolio project)"

# Build an SSL context with a real CA bundle: system Python often cannot verify
# HTTPS archive certificates (e.g. STScI/CALSPEC) without certifi's bundle.
try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _get(url: str, timeout: int = 60) -> bytes:
    """HTTP(S) GET with a descriptive User-Agent and a verified CA bundle."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = _SSL_CTX if url.lower().startswith("https") else None
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()


# SkyCalc wavelength sampling (nm): >= 4 samples per detector pixel of the
# finest simulated spectrograph (0.217 nm at R = 4000) and finer than the
# ~0.1 nm intrinsic width of OH airglow lines.
SKYCALC_WDELTA_NM = 0.05


def _write_dat(path: Path, header: str, columns: np.ndarray, fmt: str = "%.18e") -> None:
    """Write a 2-D array to ``path`` with a ``#`` provenance header."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lines = [f"# {line}" for line in header.strip().splitlines()]
    np.savetxt(path, columns, header="\n".join(lines), comments="", fmt=fmt)


# ---------------------------------------------------------------------------
# A. SVO filter curves (no extra dependencies -- VOTable is plain XML).
# ---------------------------------------------------------------------------
def fetch_svo() -> bool:
    """Download 2MASS J/H/Ks transmission curves from the SVO FPS."""
    svo_ids = {"J": "2MASS/2MASS.J", "H": "2MASS/2MASS.H", "K": "2MASS/2MASS.Ks"}
    ok = True
    for band, svo_id in svo_ids.items():
        try:
            url = f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={svo_id}"
            xml = _get(url).decode("utf-8", errors="replace")
            # The TABLEDATA holds <TR><TD>wavelength_AA</TD><TD>transmission</TD></TR>.
            tds = re.findall(r"<TD>([^<]+)</TD>", xml)
            vals = [float(t) for t in tds]
            wl_aa = np.array(vals[0::2])
            trans = np.array(vals[1::2])
            wl_um = wl_aa * 1e-4  # Angstrom -> micrometre
            order = np.argsort(wl_um)
            _write_dat(
                CACHE / f"{band}.dat",
                f"SVO Filter Profile Service  ID={svo_id}\n"
                f"source: http://svo2.cab.inta-csic.es/theory/fps/?ID={svo_id}\n"
                f"provenance: direct_public_data\n"
                f"columns: wavelength_um  transmission",
                np.column_stack([wl_um[order], trans[order]]),
            )
            print(f"  [SVO] {band}: {wl_um.size} points -> {CACHE / (band + '.dat')}")
        except Exception as e:  # noqa: BLE001 -- report and continue
            ok = False
            print(f"  [SVO] {band}: FAILED ({type(e).__name__}: {e})")
    return ok


# ---------------------------------------------------------------------------
# D. CALSPEC flux standard (needs astropy to read the FITS binary table).
# ---------------------------------------------------------------------------
def fetch_calspec() -> bool:
    """Download a CALSPEC flux-standard spectrum and cache it as source.dat."""
    try:
        from astropy.io import fits  # noqa: PLC0415 -- optional, dev-only
    except ImportError:
        print("  [CALSPEC] SKIPPED -- astropy not installed (pip install astropy)")
        return False

    base = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/"
    try:
        # Find a STIS flux standard in the directory listing (version-robust).
        index = _get(base, timeout=60).decode("utf-8", errors="replace")
        candidates = re.findall(r'href="([a-z0-9_]+_stis(?:nic)?_\d+\.fits)"', index)
        if not candidates:
            print("  [CALSPEC] no STIS standard found in archive listing")
            return False
        # Prefer a solar-analog / G-type standard if present, else the first.
        pick = next((c for c in candidates if c.startswith(("p041", "p177", "sun"))), candidates[0])
        raw = _get(base + pick, timeout=120)
        tmp = CACHE / "_calspec_tmp.fits"
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(raw)
        with fits.open(tmp) as hdul:
            data = hdul[1].data
            wl_aa = np.asarray(data["WAVELENGTH"], dtype=float)
            flux = np.asarray(data["FLUX"], dtype=float)
        tmp.unlink(missing_ok=True)

        wl_um = wl_aa * 1e-4
        # Trim to the optical-NIR range and downsample to keep the file small.
        keep = (wl_um >= 0.30) & (wl_um <= 2.60)
        wl_um, flux = wl_um[keep], flux[keep]
        if wl_um.size > 4000:
            idx = np.linspace(0, wl_um.size - 1, 4000).astype(int)
            wl_um, flux = wl_um[idx], flux[idx]
        _write_dat(
            CACHE / "source.dat",
            f"STScI CALSPEC flux standard  file={pick}\n"
            f"source: {base}{pick}\n"
            f"provenance: direct_public_data\n"
            f"columns: wavelength_um  flux_erg_s_cm2_A (trimmed 0.30-2.60 um, downsampled)",
            np.column_stack([wl_um, flux]),
        )
        print(f"  [CALSPEC] {pick}: {wl_um.size} points -> {CACHE / 'source.dat'}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [CALSPEC] FAILED ({type(e).__name__}: {e})")
        return False


# ---------------------------------------------------------------------------
# B. ESO SkyCalc sky model (needs skycalc_cli + astropy; POSTs to ESO service).
# ---------------------------------------------------------------------------
def fetch_skycalc() -> bool:
    """Query ESO SkyCalc for a default sky model and cache it as skycalc.dat."""
    try:
        from astropy.io import fits  # noqa: PLC0415
        from skycalc_cli.skycalc import SkyModel  # noqa: PLC0415
    except ImportError:
        print("  [SkyCalc] SKIPPED -- needs 'pip install skycalc_cli astropy'")
        return False

    try:
        sm = SkyModel()
        # Near-IR request: airmass 1.0, default Paranal conditions, 1.9-2.5 um.
        # ``wdelta`` is ESO's wavelength SAMPLING step in nm (see the SkyCalc
        # CLI help).  It must resolve the finest science grid this repository
        # simulates -- the R = 4000 high-z case has 0.217 nm detector pixels and
        # a 0.54 nm instrumental FWHM, and OH airglow lines are intrinsically
        # ~0.1 nm wide -- so SKYCALC_WDELTA_NM (0.05 nm, >= 4 samples per pixel)
        # is used.  The previous 1.0 nm cache, interpolated onto that grid,
        # mis-stated the OH comb by ~200 % rms and the transmission by 0.14 rms.
        params = {
            "airmass": 1.0,
            "wmin": 1900.0,   # nm
            "wmax": 2500.0,   # nm
            "wgrid_mode": "fixed_wavelength_step",
            "wdelta": SKYCALC_WDELTA_NM,
        }
        sm.callwith(params)
        tmp = CACHE / "_skycalc_tmp.fits"
        CACHE.mkdir(parents=True, exist_ok=True)
        sm.write(str(tmp))
        with fits.open(tmp) as hdul:
            data = hdul[1].data
            wl_nm = np.asarray(data["lam"], dtype=float)
            trans = np.asarray(data["trans"], dtype=float)
            flux = np.asarray(data["flux"], dtype=float)  # sky emission radiance
        tmp.unlink(missing_ok=True)
        wl_um = wl_nm * 1e-3
        _write_dat(
            CACHE / "skycalc.dat",
            "ESO Cerro Paranal Sky Model (SkyCalc)\n"
            "source: https://www.eso.org/sci/software/pipelines/skytools/skymodel\n"
            "provenance: direct_public_data\n"
            f"airmass 1.0, {params['wmin']:.0f}-{params['wmax']:.0f} nm, "
            f"wdelta {SKYCALC_WDELTA_NM} nm (fixed_wavelength_step)\n"
            "columns: wavelength_um  radiance  transmission",
            np.column_stack([wl_um, flux, trans]),
            fmt="%.8g",
        )
        print(f"  [SkyCalc] {wl_um.size} points -> {CACHE / 'skycalc.dat'}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [SkyCalc] FAILED ({type(e).__name__}: {e})")
        return False


def main() -> int:
    print("Fetching public-data samples into", CACHE)
    results = {
        "SVO filters": fetch_svo(),
        "CALSPEC source": fetch_calspec(),
        "ESO SkyCalc": fetch_skycalc(),
    }
    print("-" * 60)
    for name, ok in results.items():
        print(f"  {name:18s}: {'OK' if ok else 'not fetched (see message above)'}")
    print("Re-run examples/run_public_anchor_overview.py to see updated provenance.")
    # Success if at least one source was fetched.
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
