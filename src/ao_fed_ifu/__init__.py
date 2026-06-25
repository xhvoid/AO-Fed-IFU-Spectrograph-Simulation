"""AO-fed integral-field spectrograph (IFU) simulation -- a compact demonstrator.

This package is an *engineering-informed* toy model of the data path of an
adaptive-optics (AO) fed integral-field spectrograph:

    source datacube  ->  AO PSF coupling  ->  IFU image-slicer remapping
    ->  spectral dispersion  ->  detector noise  ->  cube reconstruction
    ->  science diagnostics  ->  internal validation

It is **not** a reproduction of HARMONI, MUSE, NIRSpec or any real instrument,
and it makes no calibrated performance predictions.  Real-instrument names are
used only as *inspiration anchors* for plausible parameter ranges, and every
quantitative output is tagged with a provenance label (see ``data_sources``).

----------------------------------------------------------------------
Conventions used throughout the package (read this once, then the code
reads cleanly):

Units
    wavelength            micrometres            ``um``
    angular scale         milli-arcseconds       ``mas``
    telescope diameter    metres                 ``m``
    exposure time         seconds                ``s``
    detected signal       photo-electrons        ``e-``  (after throughput)
    read/dark noise       electrons              ``e-``

Array axis order
    A science datacube ``I(x, y, lambda)`` is stored as a NumPy array with
    shape ``(ny, nx, nlambda)`` and is indexed ``cube[iy, ix, il]``.
    * axis 0 = ``iy`` : spatial row    (detector cross-dispersion family)
    * axis 1 = ``ix`` : spatial column (along-slit family)
    * axis 2 = ``il`` : wavelength bin (dispersion direction)

    A detector frame is a 2-D array ``frame[det_y, det_x]`` where ``det_x`` is
    the dispersion (wavelength) axis and ``det_y`` is the cross-dispersion axis
    that stacks the image-slicer slices.

These two conventions are the contract between modules; if you keep them in
mind the forward model (cube -> detector) and the inverse model
(detector -> cube) are straightforward to follow.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Xu Han"

# A single source of truth for the speed of light, used by the velocity
# diagnostics (Doppler shift) and the wavelength-calibration validation checks.
SPEED_OF_LIGHT_KM_S: float = 299_792.458  # km / s, exact by SI definition

__all__ = ["__version__", "__author__", "SPEED_OF_LIGHT_KM_S"]
