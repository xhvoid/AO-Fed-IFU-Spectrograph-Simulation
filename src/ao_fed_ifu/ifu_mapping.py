"""Image-slicer IFU remapping: from a datacube to a 2-D detector frame.

An integral-field unit (IFU) has to fit a 2-D field *and* a spectral axis onto a
2-D detector.  An **image slicer** does this by cutting the field into thin
strips ("slices"), re-arranging the slices end-to-end into a long **pseudo-slit**,
and feeding that slit to the spectrograph.  Each point on the slit is then
dispersed into a spectrum across the detector.

Our model (a deliberate ``synthetic_proxy`` -- not a real slicer geometry):

    * The field has ``nx_spaxel x ny_spaxel`` spaxels.
    * Slices run along x; a slice is ``slice_width_spaxels`` spaxel-rows tall.
    * On the detector, **each spaxel maps to exactly one detector row** (the
      cross-dispersion axis), and **wavelength maps to detector columns** (the
      dispersion axis) via the spectrograph's wavelength solution.
    * Slices are stacked along detector rows with a small gap between them.

Because every spaxel owns a unique detector row and every wavelength a unique
column, the forward map is a clean rearrangement of flux: with the LSF and
cross-talk turned off it conserves total electrons *exactly*, which makes the
inverse map (``cube_reconstruction``) and the flux-conservation validation
straightforward.

**Cross-talk** -- the unwanted leakage of light between neighbouring traces --
is modelled as a Gaussian blur along the detector-row (cross-dispersion) axis
with width ``cross_talk_sigma_pix``.  Larger sigma leaks more signal from a
spaxel into its neighbours, which the cross-talk diagnostic then measures.

Detector frame convention: ``frame[det_y, det_x]`` with ``det_x`` = dispersion
(wavelength) and ``det_y`` = cross-dispersion (spaxel/slice stacking).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .spectrograph_model import SpectrographConfig


@dataclass
class IFUConfig:
    """Geometry of the image-slicer IFU.

    Parameters
    ----------
    nx_spaxel, ny_spaxel
        Field size in spaxels (x = along-slice, y = across-slices).
    spaxel_scale_mas
        Angular spaxel size in mas (carried for bookkeeping / metadata).
    slicer_mode
        ``"image_slicer"`` (implemented) or ``"lenslet"`` (deferred -- raises).
    slice_width_spaxels
        Number of spaxel-rows that make up one slice strip.  Default 1: each
        spaxel-row is its own slice (the simplest, most common toy choice).
    cross_talk_sigma_pix
        Gaussian sigma (in detector rows) of the cross-dispersion leakage
        between adjacent traces.  0 = no cross-talk.
    slice_gap_pix
        Empty detector rows inserted between consecutive slice blocks, which
        suppresses leakage *between* slices relative to *within* a slice.
    """

    nx_spaxel: int = 16
    ny_spaxel: int = 16
    spaxel_scale_mas: float = 10.0
    slicer_mode: str = "image_slicer"
    slice_width_spaxels: int = 1
    cross_talk_sigma_pix: float = 0.0
    slice_gap_pix: int = 2

    def __post_init__(self) -> None:
        if self.slicer_mode not in ("image_slicer", "lenslet"):
            raise ValueError(f"Unknown slicer_mode {self.slicer_mode!r}")
        if self.slicer_mode == "lenslet":
            raise NotImplementedError(
                "Lenslet mode is intentionally deferred; use slicer_mode='image_slicer'. "
                "See the project roadmap -- image-slicer mode is the first milestone."
            )
        if self.slice_width_spaxels < 1:
            raise ValueError("slice_width_spaxels must be >= 1")

    @property
    def n_spaxel(self) -> int:
        """Total number of spaxels in the field."""
        return self.nx_spaxel * self.ny_spaxel

    @property
    def n_slices(self) -> int:
        """Number of slices the field is cut into."""
        return int(np.ceil(self.ny_spaxel / self.slice_width_spaxels))


@dataclass
class SlicerMapping:
    """Forward/inverse bookkeeping between spaxels and detector pixels.

    Built once per (IFU, spectrograph) pair by :func:`build_slicer_mapping`.
    Stores, for every spaxel, the detector row it lands on, plus the column the
    each wavelength lands on, so that both the forward projection and the inverse
    reconstruction are pure look-ups.

    Arrays indexed by ``spaxel_id = iy * nx_spaxel + ix``:
        ``ix_of_spaxel, iy_of_spaxel`` -- spatial indices,
        ``slice_of_spaxel``           -- which slice the spaxel belongs to,
        ``slit_y_of_spaxel``          -- position along the pseudo-slit,
        ``det_row_of_spaxel``         -- the detector row it occupies.
    Plus ``col_of_lambda`` -- the detector column for each cube wavelength bin,
    and ``det_shape = (n_rows, n_cols)``.
    """

    ifu: IFUConfig
    spectro: SpectrographConfig
    ix_of_spaxel: np.ndarray
    iy_of_spaxel: np.ndarray
    slice_of_spaxel: np.ndarray
    slit_y_of_spaxel: np.ndarray
    det_row_of_spaxel: np.ndarray
    col_of_lambda: np.ndarray
    det_shape: tuple[int, int]

    def spaxel_id(self, ix: int, iy: int) -> int:
        """Flattened spaxel id for spatial indices ``(ix, iy)``."""
        return int(iy) * self.ifu.nx_spaxel + int(ix)

    def spaxel_trace_table(self) -> pd.DataFrame:
        """One row per spaxel: how each spaxel maps onto the detector.

        Compact (``n_spaxel`` rows) -- this is what reconstruction actually
        needs.  Columns: spaxel_id, ix, iy, slice_id, slit_y, det_y.
        """
        return pd.DataFrame(
            {
                "spaxel_id": np.arange(self.ifu.n_spaxel),
                "ix": self.ix_of_spaxel,
                "iy": self.iy_of_spaxel,
                "slice_id": self.slice_of_spaxel,
                "slit_y": self.slit_y_of_spaxel,
                "det_y": self.det_row_of_spaxel,
            }
        )

    def detector_trace_table(self, lambda_index: np.ndarray | None = None) -> pd.DataFrame:
        """Full per-(spaxel, wavelength) trace table for inspection/plots.

        Columns: spaxel_id, slice_id, slit_y, lambda_index, det_x, det_y.
        This expands to ``n_spaxel * nlambda`` rows, so by default only a coarse
        subset of wavelength indices is included (override with ``lambda_index``).
        """
        nlam = self.col_of_lambda.size
        if lambda_index is None:
            # A handful of representative wavelengths keeps the table readable.
            lambda_index = np.unique(np.linspace(0, nlam - 1, 5).astype(int))
        lambda_index = np.atleast_1d(np.asarray(lambda_index, dtype=int))

        rows = []
        for il in lambda_index:
            det_x = int(self.col_of_lambda[il])
            for sid in range(self.ifu.n_spaxel):
                rows.append(
                    {
                        "spaxel_id": sid,
                        "slice_id": int(self.slice_of_spaxel[sid]),
                        "slit_y": int(self.slit_y_of_spaxel[sid]),
                        "lambda_index": int(il),
                        "det_x": det_x,
                        "det_y": int(self.det_row_of_spaxel[sid]),
                    }
                )
        return pd.DataFrame(rows)


def build_slicer_mapping(
    ifu_config: IFUConfig,
    spectrograph_config: SpectrographConfig,
    cube_wavelength_grid_um: np.ndarray,
) -> SlicerMapping:
    """Construct the :class:`SlicerMapping` for an IFU + spectrograph.

    The detector geometry is derived as follows:

    * **Rows (cross-dispersion).**  Within a slice, the along-slit position is
      ``row_within_slice * nx_spaxel + ix`` so each slice occupies
      ``slice_width_spaxels * nx_spaxel`` contiguous rows.  Slices are stacked
      with ``slice_gap_pix`` empty rows between them.
    * **Columns (dispersion).**  Each cube wavelength bin is placed at the
      detector column nearest to ``wavelength_to_detector_x(lambda)``.  When the
      cube's wavelength grid matches the spectrograph's (the recommended setup)
      this is the identity ``col = lambda_index``, giving exact invertibility.
    """
    nx, ny = ifu_config.nx_spaxel, ifu_config.ny_spaxel
    sw = ifu_config.slice_width_spaxels
    slit_len_per_slice = sw * nx
    slice_pitch = slit_len_per_slice + ifu_config.slice_gap_pix

    ix_of_spaxel = np.empty(ifu_config.n_spaxel, dtype=int)
    iy_of_spaxel = np.empty(ifu_config.n_spaxel, dtype=int)
    slice_of_spaxel = np.empty(ifu_config.n_spaxel, dtype=int)
    slit_y_of_spaxel = np.empty(ifu_config.n_spaxel, dtype=int)
    det_row_of_spaxel = np.empty(ifu_config.n_spaxel, dtype=int)

    for iy in range(ny):
        for ix in range(nx):
            sid = iy * nx + ix
            slice_id = iy // sw
            row_within_slice = iy % sw
            slit_y = row_within_slice * nx + ix
            det_row = slice_id * slice_pitch + slit_y
            ix_of_spaxel[sid] = ix
            iy_of_spaxel[sid] = iy
            slice_of_spaxel[sid] = slice_id
            slit_y_of_spaxel[sid] = slit_y
            det_row_of_spaxel[sid] = det_row

    n_rows = ifu_config.n_slices * slice_pitch  # includes one trailing gap
    n_cols = spectrograph_config.n_detector_pixels

    # Column for each cube wavelength bin via the spectrograph wavelength solution.
    cube_wavelength_grid_um = np.asarray(cube_wavelength_grid_um, dtype=float)
    col_float = spectrograph_config.wavelength_to_detector_x(cube_wavelength_grid_um)
    col_of_lambda = np.clip(np.round(col_float).astype(int), 0, n_cols - 1)

    return SlicerMapping(
        ifu=ifu_config,
        spectro=spectrograph_config,
        ix_of_spaxel=ix_of_spaxel,
        iy_of_spaxel=iy_of_spaxel,
        slice_of_spaxel=slice_of_spaxel,
        slit_y_of_spaxel=slit_y_of_spaxel,
        det_row_of_spaxel=det_row_of_spaxel,
        col_of_lambda=col_of_lambda,
        det_shape=(n_rows, n_cols),
    )


def forward_project_cube(
    cube: np.ndarray,
    mapping: SlicerMapping,
    apply_lsf: bool = True,
    apply_cross_talk: bool = True,
) -> np.ndarray:
    """Project a datacube onto a noiseless detector electron frame.

    Parameters
    ----------
    cube
        Input datacube in electrons, shape ``(ny, nx, nlambda)``.
    mapping
        The :class:`SlicerMapping` describing where each spaxel/wavelength goes.
    apply_lsf
        If True, convolve along the dispersion axis with the spectrograph LSF
        (blurs the spectrum to the instrument's resolution).
    apply_cross_talk
        If True, apply Gaussian cross-dispersion leakage between adjacent rows.

    Returns
    -------
    detector_electrons
        2-D array ``frame[det_y, det_x]`` of noiseless electrons.

    Notes
    -----
    With both ``apply_lsf`` and ``apply_cross_talk`` False this is an exact
    rearrangement of flux (sum preserved to machine precision); the validation
    flux-conservation check relies on that.
    """
    ny, nx, nlam = cube.shape
    n_rows, n_cols = mapping.det_shape
    frame = np.zeros((n_rows, n_cols), dtype=float)

    # Scatter each spaxel's spectrum into its detector row.  np.add.at handles the
    # (rare) case where several wavelength bins share a column after rounding.
    cols = mapping.col_of_lambda
    for iy in range(ny):
        for ix in range(nx):
            sid = iy * nx + ix
            det_row = mapping.det_row_of_spaxel[sid]
            np.add.at(frame[det_row], cols, cube[iy, ix, :])

    if apply_lsf:
        frame = mapping.spectro.apply_lsf(frame, axis=1)  # along dispersion (columns)
    if apply_cross_talk and mapping.ifu.cross_talk_sigma_pix > 0:
        frame = apply_cross_dispersion_crosstalk(frame, mapping.ifu.cross_talk_sigma_pix)
    return frame


def apply_cross_dispersion_crosstalk(frame: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Gaussian leakage between adjacent detector rows (cross-dispersion).

    Models the finite cross-dispersion width of each trace: a fraction of every
    spaxel's light spills into neighbouring rows.  Implemented as a Gaussian blur
    along axis 0 (rows).  ``mode="nearest"`` keeps interior flux conserved; the
    trend (more sigma -> more leakage) is what matters for the diagnostics.
    """
    if sigma_pix <= 0:
        return frame
    return gaussian_filter1d(frame, sigma=sigma_pix, axis=0, mode="nearest")


def image_slicer_map(
    cube: np.ndarray,
    ifu_config: IFUConfig,
    spectrograph_config: SpectrographConfig,
    cube_wavelength_grid_um: np.ndarray,
    apply_lsf: bool = True,
    apply_cross_talk: bool = True,
) -> tuple[np.ndarray, SlicerMapping]:
    """Convenience: build the mapping and forward-project a cube in one call.

    Returns ``(detector_electrons, mapping)``.  The mapping is needed later by
    :mod:`ao_fed_ifu.cube_reconstruction` to invert the projection.
    """
    mapping = build_slicer_mapping(ifu_config, spectrograph_config, cube_wavelength_grid_um)
    frame = forward_project_cube(
        cube, mapping, apply_lsf=apply_lsf, apply_cross_talk=apply_cross_talk
    )
    return frame, mapping
