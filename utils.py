"""Helpers for jwst_survey.py: repository paths, instrument constants, names, and geometry.

Contents
--------
* Repository layout: ``REPO_ROOT``, ``DATA_DIR``, ``APT_DIR``, ``CAL_DIR``, ``OUTPUT_DIR``,
  ``cal_file_paths``, ``output_path``.
* Instrument constants, read from ``config/<instrument>.yaml`` by ``load_instrument``
  and exposed as ``NIRCAM_*`` names.
* Name helpers: ``nircam_filter``, ``nircam_channel``, ``normalize_detector``.
* Sky-geometry helpers: ``unit_mean``, ``polygon_center_rotation``, ``parse_s_region``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord

# ======================================================================================
# Repository layout
# ======================================================================================
#
#   <repo>/config/            instrument constants (YAML)
#   <repo>/data/apt/          APT files (.aptx)
#   <repo>/data/cal_files/    one subdirectory of stage-2 cal files per program/epoch
#   <repo>/outputs/           generated tables and figures
#
# Set the environment variable JWST_SURVEY_DATA (and/or JWST_SURVEY_OUTPUT) to keep the
# data or outputs elsewhere, e.g. on an external drive. Both are read once, at import.

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = Path(os.environ.get("JWST_SURVEY_DATA", REPO_ROOT / "data")).expanduser()
APT_DIR = DATA_DIR / "apt"
CAL_DIR = DATA_DIR / "cal_files"
OUTPUT_DIR = Path(os.environ.get("JWST_SURVEY_OUTPUT", REPO_ROOT / "outputs")).expanduser()


def cal_file_paths(subdir, pattern="*_cal*.fits"):
    """Return the sorted paths of the stage-2 cal files for one program/epoch.

    Cal files are expected in one subdirectory per program or epoch under
    ``CAL_DIR`` (by default ``<repo>/data/cal_files/``; see ``JWST_SURVEY_DATA``).
    A relative ``subdir`` is interpreted relative to ``CAL_DIR``; an absolute path
    (or one starting with ``~``) is used as given, so files stored elsewhere can
    be read without moving them.

    Parameters
    ----------
    subdir : str or pathlib.Path
        Name of the subdirectory of ``CAL_DIR`` (e.g. "jw080600_epoch1_v1.0e_cal"),
        or an absolute path to a directory of cal files.
    pattern : str, optional
        Shell-style filename pattern, matched within ``subdir`` only (not in its
        subdirectories). The default, "*_cal*.fits", matches pipeline products
        such as "*_cal.fits" and modified versions such as
        "*_cal_bkgsub_tweak.fits".

    Returns
    -------
    list of pathlib.Path
        Matching files, sorted alphabetically by full path. For standard JWST
        filenames this groups the files by observation, visit, exposure, and
        detector.

    Raises
    ------
    FileNotFoundError
        If no file matches, for example because the subdirectory name is
        misspelled or the pattern does not fit the file names. An error is raised
        instead of returning an empty list so that a wrong path cannot silently
        produce an empty survey.

    Examples
    --------
    >>> paths = cal_file_paths("jw080600_epoch1_v1.0e_cal")
    >>> exposures = exposures_from_cal(paths, epoch="Y1E1")
    """
    folder = Path(subdir).expanduser()
    if not folder.is_absolute():
        folder = CAL_DIR / folder
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    return paths


def output_path(*parts):
    """Return a path inside the output directory, creating its parent directories.

    The path is ``OUTPUT_DIR`` joined with ``parts``, where ``OUTPUT_DIR`` is
    ``<repo>/outputs/`` by default or the value of the environment variable
    ``JWST_SURVEY_OUTPUT``. All directories leading to the final component are
    created if they do not exist, so the result can be passed directly to a
    writer such as ``DataFrame.to_csv``. The final component itself is not
    created: it is treated as a file name.

    Parameters
    ----------
    *parts : str or pathlib.Path
        Successive path components below ``OUTPUT_DIR``; the last one is the file
        name (e.g. ``"jets_epoch1", "depths_Y1E1.csv"``).

    Returns
    -------
    pathlib.Path
        The full output path. Existing files at this path are not modified;
        whether they are overwritten depends on the function that writes to it.

    Examples
    --------
    >>> depths.to_csv(output_path("jets_epoch1", "depths_Y1E1.csv"), index=False)
    # writes <repo>/outputs/jets_epoch1/depths_Y1E1.csv, creating jets_epoch1/ if needed
    """
    path = OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================================================
# Instrument constants (config/<instrument>.yaml)
# ======================================================================================


def load_instrument(name="nircam"):
    """Read the constants of one instrument from ``config/<name>.yaml``.

    Parameters
    ----------
    name : str
        Instrument name; the file ``CONFIG_DIR / f"{name}.yaml"`` is read.

    Returns
    -------
    dict
        The YAML content as nested dictionaries and lists (see the YAML file for
        the keys, units, and sources).
    """
    with open(CONFIG_DIR / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


NIRCAM = load_instrument("nircam")

# Convenience names used throughout jwst_survey.py.
NIRCAM_CHANNEL_DEFAULTS = NIRCAM["channels"]                    # {"SW": {...}, "LW": {...}}
NIRCAM_PSF_FWHM_ARCSEC = NIRCAM["psf_fwhm_arcsec"]              # {filter: FWHM [arcsec]}
NIRCAM_READOUT_PATTERNS = {k: tuple(v) for k, v in NIRCAM["readout_patterns"].items()}
NIRCAM_TFRAME_FULL = float(NIRCAM["tframe_full_s"])             # s
NIRCAM_ACTIVE_PIXELS = int(NIRCAM["active_pixels"])
NIRCAM_SW_DETECTORS = NIRCAM["detectors"]["SW"]                 # {"A": [...], "B": [...]}
NIRCAM_LW_DETECTORS = NIRCAM["detectors"]["LW"]
NIRCAM_REFERENCE_APERTURES = {                                  # {(module, placement): aperture}
    (module, placement): aperture
    for module, placements in NIRCAM["reference_apertures"].items()
    for placement, aperture in placements.items()
}

# Noise-equivalent area of a Gaussian PSF: NEA = 4 pi sigma^2 = pi / (2 ln 2) FWHM^2.
GAUSSIAN_NEA_FACTOR = np.pi / (2.0 * np.log(2.0))  # ~2.266

# Columns of the standard exposure table produced by every builder in jwst_survey.py.
EXPOSURE_COLUMNS = [
    "time", "ra", "dec", "rotation", "footprint_side_deg", "filter", "channel", "detector",
    "readpatt", "ngroups", "nframes", "nints", "tgroup", "photmjsr", "pixar_sr", "sky_mjysr", "epoch",
]

# ======================================================================================
# Name helpers
# ======================================================================================

_FILTER_RE = re.compile(r"^F\d{3}[WMN]2?$")


def nircam_filter(filter_kw, pupil_kw):
    """Effective NIRCam filter: a filter in the pupil wheel (e.g. F162M behind F150W2) wins."""
    pupil_kw = (pupil_kw or "").strip().upper()
    return pupil_kw if _FILTER_RE.match(pupil_kw) else str(filter_kw).strip().upper()


def nircam_channel(detector):
    """'SW' or 'LW' from a detector name such as NRCA1 or NRCALONG / NRCA5."""
    d = str(detector).upper()
    return "LW" if ("LONG" in d or d.endswith("5")) else "SW"


def normalize_detector(detector):
    """Use one naming scheme for LW detectors (headers say NRCALONG, SIAF says NRCA5)."""
    d = str(detector).upper()
    return d.replace("LONG", "5")


# ======================================================================================
# Sky-geometry helpers
# ======================================================================================


def unit_mean(coords):
    """Mean direction of sky coordinates (normalized mean of their unit vectors).

    Unlike averaging RA and Dec, this is correct across RA = 0/360 deg and near the poles.
    """
    xyz = coords.cartesian.xyz.value.reshape(3, -1).mean(axis=1)
    sph = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], representation_type="cartesian").spherical
    return SkyCoord(sph.lon, sph.lat)


def polygon_center_rotation(ra, dec):
    """Center, LightCurveLynx rotation and mean side length of a (near-)square polygon.

    LightCurveLynx's ``rotation`` is the position angle (deg east of north) of the
    footprint's y-axis; for a square detector it is defined modulo 90 degrees.
    Returns (center_ra, center_dec, rotation_deg, side_deg).
    """
    corners = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    center = unit_mean(corners)
    mid = unit_mean(corners[0:2])
    rotation = center.position_angle(mid).deg % 90.0
    side = corners.separation(np.roll(corners, -1)).deg.mean()
    return center.ra.deg, center.dec.deg, rotation, float(side)


def parse_s_region(s_region):
    """(ra, dec) vertex arrays of an S_REGION 'POLYGON [ICRS] ...' string, in degrees."""
    numbers = [float(tok) for tok in str(s_region).split() if re.match(r"^[-+]?\d", tok)]
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError(f"Cannot parse S_REGION: {s_region!r}")
    return np.array(numbers[0::2]), np.array(numbers[1::2])
