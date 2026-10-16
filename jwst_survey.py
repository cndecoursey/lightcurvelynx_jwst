"""LightCurveLynx survey tables for JWST/NIRCam imaging programs, executed or planned.

Every survey is described by one *exposure table*: one row per detector exposure, with
the columns listed in ``utils.EXPOSURE_COLUMNS``. Three builders produce it:

* ``exposures_from_apt``   planned programs: an APT ``.aptx`` file (geometry from pysiaf)
* ``exposures_from_mast``  executed programs: MAST JWST keyword search (no downloads)
* ``exposures_from_cal``   executed programs: stage-2 ``*_cal.fits`` headers (+ sky level)

Calibration quantities a plan cannot contain (PHOTMJSR, gain, sky background) are filled
from a *reference lookup* keyed by (filter, detector), built from executed data with
``reference_lookup``. ``NIRCamObsTable`` turns the exposure table into the inputs of
LightCurveLynx's ``PoissonFluxNoiseModel``; ``build_nircam_surveys`` returns one
``SurveyInfo`` per channel (SW, LW), each with its own single-detector footprint.

Noise inputs derived per row (all in electrons over the full exposure):

* ``exptime``    = NINTS * (NGROUPS - 1) * TGROUP                          [s]
* ``read_noise`` = sqrt(12 (n-1) / (m n (n+1))) * sigma_CDS / sqrt(2)      [e-/pixel/ramp]
                   (Rauscher et al. 2007, PASP 119, 768, Eq. 1; white read noise)
* ``nexposure``  = NINTS
* ``zp``         = PHOTMJSR * PIXAR_SR * 1e15 / (gain * exptime)           [nJy/e-]
* ``sky_bg_e``   = sky [MJy/sr] / PHOTMJSR * gain * exptime                [e-/pixel]
* ``psf_footprint`` = 2.266 (FWHM / pixel scale)^2 (in-flight FWHM)        [pixels]

Instrument constants (gain, read noise, dark current, PSF FWHM, readout patterns, ...)
are read from ``config/nircam.yaml``; repository paths and small helpers are in ``utils.py``.
"""

from __future__ import annotations

import fnmatch
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time

# Repository paths, NIRCam constants (from config/nircam.yaml), and helpers live in utils.py.
# They are imported here so that they are also available as jwst_survey.<name>.
from utils import (
    APT_DIR, CAL_DIR, CONFIG_DIR, DATA_DIR, OUTPUT_DIR, REPO_ROOT, cal_file_paths, output_path,
    EXPOSURE_COLUMNS, GAUSSIAN_NEA_FACTOR, NIRCAM, NIRCAM_ACTIVE_PIXELS, NIRCAM_CHANNEL_DEFAULTS,
    NIRCAM_LW_DETECTORS, NIRCAM_PSF_FWHM_ARCSEC, NIRCAM_READOUT_PATTERNS,
    NIRCAM_REFERENCE_APERTURES, NIRCAM_SW_DETECTORS, NIRCAM_TFRAME_FULL,
    nircam_channel, nircam_filter, normalize_detector,
    parse_s_region, polygon_center_rotation, unit_mean,
)

__all__ = [
    # re-exported from utils.py
    "APT_DIR", "CAL_DIR", "CONFIG_DIR", "DATA_DIR", "OUTPUT_DIR", "REPO_ROOT", "cal_file_paths", "output_path",
    "EXPOSURE_COLUMNS", "GAUSSIAN_NEA_FACTOR", "NIRCAM", "NIRCAM_ACTIVE_PIXELS", "NIRCAM_CHANNEL_DEFAULTS",
    "NIRCAM_LW_DETECTORS", "NIRCAM_PSF_FWHM_ARCSEC", "NIRCAM_READOUT_PATTERNS", "NIRCAM_REFERENCE_APERTURES",
    "NIRCAM_SW_DETECTORS", "NIRCAM_TFRAME_FULL", "nircam_channel", "nircam_filter", "normalize_detector",
    "parse_s_region", "polygon_center_rotation", "unit_mean",
    # defined here
    "nircam_detector_polygons", "siaf_pixel_area_sr",
    "read_apt", "exposures_from_apt", "exposures_from_mast", "read_cal_header", "exposures_from_cal",
    "observed_start_and_v3pa", "reference_lookup", "apply_reference", "ramp_read_noise",
    "nircam_footprint", "build_nircam_surveys", "five_sigma_depth", "compare_exposure_tables", "stack_epochs",
]

# ======================================================================================
# Detector footprints (pysiaf)
# ======================================================================================


def nircam_detector_polygons(ra, dec, v3pa, *, reference_aperture="NRCALL_FULL", detectors=None):
    """Sky corners of each NIRCam detector for a pointing, using pysiaf.

    Parameters
    ----------
    ra, dec : float
        Position (deg) placed at the reference point of ``reference_aperture``.
    v3pa : float
        Position angle of the telescope V3 axis at the reference point (deg).
    detectors : list of str, optional
        Detector names (e.g. ["NRCA1", ..., "NRCB5"]). Default: all ten.

    Returns
    -------
    dict
        detector -> (ra_corners, dec_corners)
    """
    import pysiaf
    from pysiaf.utils import rotations

    siaf = pysiaf.Siaf("NIRCam")
    ref = siaf[reference_aperture]
    att = rotations.attitude(ref.V2Ref, ref.V3Ref, ra, dec, v3pa)
    if detectors is None:
        detectors = sum(NIRCAM_SW_DETECTORS.values(), []) + sum(NIRCAM_LW_DETECTORS.values(), [])
    out = {}
    for det in detectors:
        aper = siaf[f"{det}_FULL"]
        aper.set_attitude_matrix(att)
        x, y = aper.corners("idl")
        out[det] = aper.idl_to_sky(x, y)
    return out


def siaf_pixel_area_sr(detector):
    """Nominal pixel solid angle [sr] of a NIRCam detector from the SIAF plate scales."""
    import pysiaf

    aper = pysiaf.Siaf("NIRCam")[f"{normalize_detector(detector)}_FULL"]
    return (aper.XSciScale * aper.YSciScale) * (np.pi / 180.0 / 3600.0) ** 2


# ======================================================================================
# Builder 1: APT file (planned programs)
# ======================================================================================


def _strip(tag):
    return tag.split("}")[-1]


def _child_text(elem, name, default=None):
    for c in elem:
        if _strip(c.tag) == name:
            return (c.text or "").strip()
    return default


def _ms_to_mjd(ms):
    return Time(datetime.fromtimestamp(int(ms) / 1000.0, timezone.utc)).mjd


def _parse_pcf(pcf):
    """Parse an APT 'StSchedulingPCF' string into (mjd, value, range-or-None) tuples."""
    toks = pcf.split()
    rows, i = [], 0
    while i + 1 < len(toks):
        t, v = toks[i], toks[i + 1]
        i += 2
        rng = None
        if i < len(toks) and ":" in toks[i]:
            vals = [float(x) for x in toks[i].split(":")]
            rng = list(zip(vals[0::2], vals[1::2]))  # one or more (lo, hi) intervals in degrees
            i += 1
        rows.append((_ms_to_mjd(t), float(v), rng))
    if i < len(toks):
        rows.append((_ms_to_mjd(toks[i]), np.nan, None))
    return rows


def _range_midpoint(intervals):
    """Midpoint (deg) of an allowed angle range given as one or more (lo, hi) intervals.

    APT writes a range that wraps through 0/360 deg as two intervals, e.g.
    [(0.0, 0.81), (346.83, 359.9999)]; these are merged before taking the midpoint.
    """
    iv = sorted(intervals)
    if len(iv) == 2 and iv[0][0] <= 1e-3 and iv[1][1] >= 360.0 - 1e-3:
        lo, hi = iv[1][0], iv[0][1] + 360.0
    else:
        lo, hi = iv[0][0], iv[-1][1]
        if len(iv) > 1:
            warnings.warn(f"Disjoint V3PA intervals {iv}; using the span of all of them.")
    return (0.5 * (lo + hi)) % 360.0


def _visit_planner(obs_elem, proposal_id):
    """Scheduling windows and field-of-regard V3PA ranges from the APT Visit Planner data."""
    text = None
    for e in obs_elem.iter():
        if _strip(e.tag) == "ToolValue" and f"Visit Planner:{proposal_id}:" in e.attrib.get("Name", ""):
            text = e.text
    if text is None:
        return [], []
    flag = re.search(r'<StVisitSchedulingWindows UpToDate="(\w+)"', text)
    if flag and flag.group(1).lower() == "false":
        number = _child_text(obs_elem, "Number")
        warnings.warn(f"APT obs {number}: Visit Planner data are flagged out of date; re-run the "
                      "Visit Planner in APT and save the file to refresh the scheduling windows.")
    m = re.search(r'<StVisitSchedulingWindows[^>]*StSchedulingPCF="([^"]+)"', text)
    windows = []
    if m:
        rows = _parse_pcf(m.group(1))
        for (t0, v, _), (t1, _, _) in zip(rows[:-1], rows[1:]):
            if v == 1.0:
                windows.append((t0, t1))
    m = re.search(r'Type="off-normal" StSchedulingPCF="([^"]+)"', text)
    v3pa_ranges = [(t, rng) for t, v, rng in _parse_pcf(m.group(1)) if rng] if m else []
    return windows, v3pa_ranges


def read_apt(aptx_path):
    """Parse the NIRCam imaging observations of an APT ``.aptx`` (or extracted ``.xml``) file.

    Returns a DataFrame with one row per observation: number, label, target, ra, dec,
    module, placement, subpixel positions, primary dither, mosaic size, filter configs,
    orient constraint, scheduling windows and field-of-regard V3PA ranges.
    """
    path = Path(aptx_path)
    if path.suffix == ".aptx":
        with zipfile.ZipFile(path) as z:
            xml_name = [n for n in z.namelist() if n.endswith(".xml")][0]
            root = ET.fromstring(z.read(xml_name))
    else:
        root = ET.parse(path).getroot()

    # Proposal ID: from <ProposalInformation><ProposalID>, zero-padded to the 5 digits used in
    # the Visit Planner keys. A file can hold planner data for several IDs (e.g. "00000" before
    # submission, or the ID of a program it was copied from); only the file's own ID is used.
    proposal_id = None
    for info in root.iter():
        if _strip(info.tag) == "ProposalInformation":
            pid = _child_text(info, "ProposalID")
            if pid and pid.strip().isdigit():
                proposal_id = f"{int(pid):05d}"
            break
    if proposal_id is None:  # no ID yet: most frequent planner ID, preferring a real one over "00000"
        ids = [e.attrib["Name"].split(":")[1] for e in root.iter()
               if _strip(e.tag) == "ToolValue" and e.attrib.get("Name", "").startswith("Visit Planner:")]
        real = [i for i in ids if i != "00000"]
        proposal_id = max(set(real), key=real.count) if real else ("00000" if ids else None)

    targets = {}
    for t in root.iter():
        if _strip(t.tag) == "Target":
            coords = [c for c in t if _strip(c.tag) == "EquatorialCoordinates"]
            if coords:
                c = SkyCoord(coords[0].attrib["Value"], unit=(u.hourangle, u.deg))
                targets[_child_text(t, "TargetID")] = (c.ra.deg, c.dec.deg)

    rows = []
    for obs in root.iter():
        if _strip(obs.tag) != "Observation" or _child_text(obs, "Instrument") != "NIRCAM":
            continue
        tmpl = [e for e in obs.iter() if _strip(e.tag) == "NircamImaging"]
        if not tmpl:
            continue
        tmpl = tmpl[0]
        target_id = _child_text(obs, "TargetID").split(" ", 1)[-1]
        configs = []
        for fc in tmpl.iter():
            if _strip(fc.tag) == "FilterConfig":
                configs.append({
                    "short": _child_text(fc, "ShortFilter"), "long": _child_text(fc, "LongFilter"),
                    "readpatt": _child_text(fc, "ReadoutPattern"),
                    "ngroups": int(_child_text(fc, "Groups")), "nints": int(_child_text(fc, "Integrations")),
                })
        # Orient special requirement: Mode "APA" (aperture PA) or "V3PA"; the midpoint of the
        # allowed range is used (ranges may wrap through 0/360 deg, e.g. 350-10 deg).
        orient_mode, orient_deg = None, np.nan
        for o in obs.iter():
            if _strip(o.tag) == "OrientRange" and o.attrib.get("Mode") in ("APA", "V3PA"):
                lo = float(o.attrib["OrientMin"].split()[0])
                hi = float(o.attrib["OrientMax"].split()[0])
                orient_mode = o.attrib["Mode"]
                orient_deg = (lo + 0.5 * ((hi - lo) % 360.0)) % 360.0
        mosaic = [e for e in obs.iter() if _strip(e.tag) == "MosaicParameters"]
        nrows = int(_child_text(mosaic[0], "Rows", 1)) if mosaic else 1
        ncols = int(_child_text(mosaic[0], "Columns", 1)) if mosaic else 1
        windows, v3pa_ranges = _visit_planner(obs, proposal_id)
        rows.append({
            "obs": int(_child_text(obs, "Number")), "label": _child_text(obs, "Label"),
            "target": target_id, "ra": targets[target_id][0], "dec": targets[target_id][1],
            "module": _child_text(tmpl, "Module"), "placement": _child_text(tmpl, "TargetPlacement"),
            "primary_dither": _child_text(tmpl, "PrimaryDitherType"),
            "subpixel_positions": int(_child_text(tmpl, "SubpixelPositions", 1) or 1),
            "mosaic": (nrows, ncols), "configs": configs,
            "orient_mode": orient_mode, "orient_deg": orient_deg,
            "windows": windows, "v3pa_ranges": v3pa_ranges,
        })
    obs_table = pd.DataFrame(rows)
    obs_table.attrs["proposal_id"] = proposal_id
    obs_table.attrs["same_orient"] = _same_orient_groups(root)
    return obs_table


def _same_orient_groups(root):
    groups = []
    for link in root.iter():
        if _strip(link.tag) == "SameOrientLink":
            nums = [int(re.search(r"Obs (\d+)", c.text).group(1)) for c in link if _strip(c.tag) == "Observation"]
            groups.append(nums)
    return groups


def exposures_from_apt(aptx_path, *, observations, epoch, year_index=0, start_mjd=None, v3pa=None,
                       dither_overhead_s=91.0, filter_overhead_s=333.0, obs_overhead_s=789.0):
    """Exposure table for selected NIRCam imaging observations of an APT file.

    Parameters
    ----------
    observations : list of int
        APT observation numbers that make up the epoch (e.g. [1, 2]).
    epoch : str
        Label for the epoch (used for stacking and comparison).
    year_index : int
        Which annual scheduling window to use when ``start_mjd`` is not given (0 = first).
    start_mjd : float, optional
        Start time of the first observation. Default: middle of the selected window.
    v3pa : float, optional
        V3 position angle (deg) at the reference point. Default: the midpoint of the APT
        orient constraint (Mode V3PA, or Mode APA converted with the reference aperture's
        V3IdlYAngle), also applied to observations in the same SameOrientLink; otherwise
        the middle of the field-of-regard range on the start date (the nominal roll).
    dither_overhead_s, filter_overhead_s, obs_overhead_s : float
        Idle time before an exposure at a new subpixel position, before the first
        exposure of a new filter configuration, and between observations. Defaults are
        the values measured in JETS epoch 1 (program 8060, DEEP8, 7 groups).

    Notes
    -----
    Exposures are ordered filter configuration by configuration, with all subpixel
    positions of a configuration before the next. Subpixel offsets (< 0.5") are not
    applied to the footprints. Primary dithers and mosaics are not supported yet; such
    observations raise NotImplementedError.
    """
    import pysiaf

    apt = read_apt(aptx_path)
    sel = apt[apt["obs"].isin(observations)].sort_values("obs")
    if len(sel) != len(observations):
        raise ValueError(f"Observations not found or not NIRCam imaging: {set(observations) - set(sel['obs'])}")

    siaf = pysiaf.Siaf("NIRCam")

    # Orient constraints as V3PA at each observation's reference point. An explicit constraint
    # on any observation of a SameOrientLink applies to the whole link.
    def _constraint_v3pa(row):
        if row["orient_mode"] == "V3PA":
            return row["orient_deg"]
        if row["orient_mode"] == "APA":
            ref_name = NIRCAM_REFERENCE_APERTURES.get((row["module"], row["placement"]))
            return (row["orient_deg"] - siaf[ref_name].V3IdlYAngle) % 360.0  # APA = V3PA + V3IdlYAngle
        return None

    v3pa_by_obs = {row["obs"]: _constraint_v3pa(row) for _, row in apt.iterrows()}
    for group in apt.attrs["same_orient"]:
        known = [v3pa_by_obs[o] for o in group if v3pa_by_obs.get(o) is not None]
        if known:
            for o in group:
                if v3pa_by_obs.get(o) is None:
                    v3pa_by_obs[o] = known[0]

    t = start_mjd
    rows = []
    for i_obs, (_, ob) in enumerate(sel.iterrows()):
        if i_obs > 0:
            t += obs_overhead_s / 86400.0
        if ob["primary_dither"] not in (None, "", "NONE") or tuple(ob["mosaic"]) != (1, 1):
            raise NotImplementedError(
                f"Obs {ob['obs']}: primary dithers / mosaics need per-exposure offsets "
                "(APT pointing-file export); not supported yet."
            )
        if t is None:
            if not ob["windows"]:
                raise ValueError(f"Obs {ob['obs']}: no scheduling windows in APT; pass start_mjd.")
            w0, w1 = ob["windows"][year_index]
            t = 0.5 * (w0 + w1)

        refname = NIRCAM_REFERENCE_APERTURES.get((ob["module"], ob["placement"]))
        if refname is None:
            raise NotImplementedError(f"No reference aperture for {ob['module']}/{ob['placement']}")

        if v3pa is not None:
            pa_v3 = v3pa
        elif v3pa_by_obs.get(ob["obs"]) is not None:
            pa_v3 = v3pa_by_obs[ob["obs"]]
        else:
            ranges = [(tt, r) for tt, r in ob["v3pa_ranges"] if tt <= t]
            if not ranges:
                raise ValueError(f"Obs {ob['obs']}: cannot determine V3PA; pass v3pa.")
            pa_v3 = _range_midpoint(ranges[-1][1])

        modules = ["A", "B"] if ob["module"] == "ALL" else [ob["module"]]
        sw_dets = sum((NIRCAM_SW_DETECTORS[m] for m in modules), [])
        lw_dets = sum((NIRCAM_LW_DETECTORS[m] for m in modules), [])
        polys = nircam_detector_polygons(ob["ra"], ob["dec"], pa_v3, reference_aperture=refname,
                                         detectors=sw_dets + lw_dets)
        geom = {d: polygon_center_rotation(*polys[d]) for d in polys}

        for i_cfg, cfg in enumerate(ob["configs"]):
            if i_cfg > 0:
                t += filter_overhead_s / 86400.0
            nframes, groupgap = NIRCAM_READOUT_PATTERNS[cfg["readpatt"]]
            tgroup = (nframes + groupgap) * NIRCAM_TFRAME_FULL
            # Clock time of one exposure: all frames read, including skipped ones.
            t_exp = cfg["nints"] * (cfg["ngroups"] * nframes + (cfg["ngroups"] - 1) * groupgap) * NIRCAM_TFRAME_FULL
            for i_pos in range(ob["subpixel_positions"]):
                if i_pos > 0:
                    t += dither_overhead_s / 86400.0
                tmid = t + 0.5 * t_exp / 86400.0
                for det in sw_dets + lw_dets:
                    filt = cfg["short"] if det in sw_dets else cfg["long"]
                    ra_c, dec_c, rot, side = geom[det]
                    rows.append({
                        "time": tmid, "ra": ra_c, "dec": dec_c, "rotation": rot, "footprint_side_deg": side,
                        "filter": filt, "channel": nircam_channel(det), "detector": det,
                        "readpatt": cfg["readpatt"], "ngroups": cfg["ngroups"], "nframes": nframes,
                        "nints": cfg["nints"], "tgroup": tgroup, "photmjsr": np.nan,
                        "pixar_sr": siaf_pixel_area_sr(det), "sky_mjysr": np.nan, "epoch": epoch,
                        "apt_obs": ob["obs"], "v3pa": pa_v3,
                    })
                t += t_exp / 86400.0
    return pd.DataFrame(rows)


# ======================================================================================
# Builder 2: MAST keyword search (executed programs, no downloads)
# ======================================================================================

# Header keyword (MAST column) -> exposure-table column.
_MAST_COLUMNS = {
    "expmid": "time", "filter": "filter_kw", "pupil": "pupil_kw", "detector": "detector",
    "readpatt": "readpatt", "ngroups": "ngroups", "nframes": "nframes", "nints": "nints",
    "tgroup": "tgroup", "photmjsr": "photmjsr", "pixar_sr": "pixar_sr", "s_region": "s_region",
    "targprop": "target", "program": "program", "observtn": "observation", "visit": "visit",
    "filename": "filename",
}


def exposures_from_mast(program, *, target_pattern="*", epoch=None, product="cal"):
    """Exposure table of an executed NIRCam imaging program from the MAST JWST keyword service.

    Run this where MAST is reachable. Header values are used as stored at MAST; the sky
    level is not available from keywords and is left as NaN (fill it from a reference
    lookup or from ``exposures_from_cal``).

    Parameters
    ----------
    program : int or str
        JWST program ID (e.g. 8060).
    target_pattern : str
        Case-sensitive shell-style pattern on the proposal target name (TARGPROP),
        e.g. "NIRCam-Epoch-1*" for JETS year 1, epoch 1.
    epoch : str, optional
        Epoch label for all rows. Default: program_observation_visit.
    product : str
        File suffix to keep when several product levels are listed ("cal").
    """
    from astroquery.mast import Mast

    def query(program_value):
        params = {"columns": "*", "filters": [{"paramName": "program", "values": [program_value]}]}
        result = Mast.service_request("Mast.Jwst.Filtered.Nircam", params)
        out = result.to_pandas() if result is not None else pd.DataFrame()
        out.columns = [c.lower() for c in out.columns]
        return out

    df = query(f"{int(program):05d}")
    if df.empty:
        df = query(str(int(program)))
    print(f"MAST: {len(df)} rows for program {program}")
    if df.empty:
        raise ValueError(f"MAST returned no NIRCam rows for program {program}.")

    if "targprop" in df.columns:
        targets = sorted(df["targprop"].astype(str).unique())
        df = df[df["targprop"].astype(str).apply(lambda s: fnmatch.fnmatchcase(s, target_pattern))]
        print(f"MAST: {len(df)} rows match target pattern {target_pattern!r}")
        if df.empty:
            raise ValueError(f"No target matches {target_pattern!r}; targets in program {program}: {targets}")
    if "filename" in df.columns:
        keep = df["filename"].astype(str).str.contains(f"_{product}")
        if keep.any():
            df = df[keep]
        print(f"MAST: {len(df)} rows after keeping '_{product}' products")
    if "exp_type" in df.columns:
        df = df[df["exp_type"].astype(str) == "NRC_IMAGE"]
        print(f"MAST: {len(df)} rows with EXP_TYPE = NRC_IMAGE")
    df = df.rename(columns={k: v for k, v in _MAST_COLUMNS.items() if k in df.columns})

    # Fallbacks for keywords the service may not index.
    if "time" not in df.columns and {"expstart", "expend"} <= set(df.columns):
        df["time"] = 0.5 * (df["expstart"].astype(float) + df["expend"].astype(float))
    if "s_region" not in df.columns and {"ra_v1", "dec_v1", "pa_v3"} <= set(df.columns):
        df["s_region"] = [
            _s_region_from_v1(det, ra, dec, pa)
            for det, ra, dec, pa in zip(df["detector"], df["ra_v1"].astype(float),
                                        df["dec_v1"].astype(float), df["pa_v3"].astype(float))
        ]
    missing = [c for c in ("time", "s_region", "detector", "filter_kw", "ngroups", "nframes", "nints", "tgroup")
               if c not in df.columns]
    if missing:
        raise KeyError(f"MAST result lacks {missing}; available columns: {sorted(df.columns)}")

    df = df.drop_duplicates(subset=["time", "detector"]).reset_index(drop=True)
    return _finalize_data_rows(df, epoch)


def _s_region_from_v1(detector, ra_v1, dec_v1, pa_v3):
    """S_REGION-style polygon of a detector from the V1 pointing (RA_V1, DEC_V1, PA_V3)."""
    import pysiaf
    from pysiaf.utils import rotations

    aper = pysiaf.Siaf("NIRCam")[f"{normalize_detector(detector)}_FULL"]
    aper.set_attitude_matrix(rotations.attitude(0.0, 0.0, ra_v1, dec_v1, pa_v3))
    ra, dec = aper.idl_to_sky(*aper.corners("idl"))
    return "POLYGON ICRS " + " ".join(f"{r:.8f} {d:.8f}" for r, d in zip(ra, dec))


def _finalize_data_rows(df, epoch):
    """Shared post-processing for MAST and cal-file rows."""
    out = df.copy()
    geom = out["s_region"].apply(lambda s: polygon_center_rotation(*parse_s_region(s)))
    out["ra"] = [g[0] for g in geom]
    out["dec"] = [g[1] for g in geom]
    out["rotation"] = [g[2] for g in geom]
    out["footprint_side_deg"] = [g[3] for g in geom]
    out["filter"] = [nircam_filter(f, p) for f, p in zip(out["filter_kw"], out.get("pupil_kw", [""] * len(out)))]
    out["detector"] = out["detector"].map(normalize_detector)
    out["channel"] = out["detector"].map(nircam_channel)
    for col in ("photmjsr", "pixar_sr", "sky_mjysr"):
        if col not in out.columns:
            out[col] = np.nan
    if "readpatt" not in out.columns:
        out["readpatt"] = ""
    if epoch is not None:
        out["epoch"] = epoch
    elif "epoch" not in out.columns:
        out["epoch"] = (out.get("program", "").astype(str) + "_" + out.get("observation", "").astype(str)
                        + "_" + out.get("visit", "").astype(str))
    for col in ("ngroups", "nframes", "nints"):
        out[col] = out[col].astype(int)
    out["tgroup"] = out["tgroup"].astype(float)
    return out


# ======================================================================================
# Builder 3: stage-2 cal files (executed programs)
# ======================================================================================


def read_cal_header(path, *, measure_sky=True, sky_method="auto"):
    """Exposure-table row from one NIRCam stage-2 ``_cal.fits`` file.

    sky_method
        "sci": clipped median of the good SCI pixels (images that are not background
        subtracted). "var_poisson": sky = median(VAR_POISSON) * gain * (NGROUPS-1) *
        TGROUP / PHOTMJSR, valid also for background-subtracted products, because the
        Poisson variance retains the subtracted background (gain: channel default).
        "auto": "var_poisson" when the SCI median is consistent with zero, else "sci".
        Statistics use the central 1024x1024 pixels, every second pixel.
    """
    cut = (slice(512, 1536, 2), slice(512, 1536, 2))
    with fits.open(path, memmap=True) as hdul:
        h0, h1 = hdul[0].header, hdul["SCI"].header
        row = {
            "time": h0["EXPMID"], "expstart": h0.get("EXPSTART", np.nan), "s_region": h1["S_REGION"],
            "ra_v1": h1.get("RA_V1", np.nan), "dec_v1": h1.get("DEC_V1", np.nan), "pa_v3": h1.get("PA_V3", np.nan),
            "filter_kw": h0["FILTER"],
            "pupil_kw": h0.get("PUPIL", ""), "detector": h0["DETECTOR"], "readpatt": h0.get("READPATT", ""),
            "ngroups": h0["NGROUPS"], "nframes": h0["NFRAMES"], "nints": h0["NINTS"], "tgroup": h0["TGROUP"],
            "photmjsr": h1.get("PHOTMJSR", np.nan), "pixar_sr": h1.get("PIXAR_SR", np.nan),
            "target": h0.get("TARGPROP", ""), "program": h0.get("PROGRAM"), "observation": h0.get("OBSERVTN"),
            "visit": h0.get("VISIT"), "filename": Path(path).name,
        }
        if measure_sky:
            sci = np.asarray(hdul["SCI"].data[cut], dtype=float)
            sci_med = float(sigma_clipped_stats(sci[np.isfinite(sci)], sigma=3.0, maxiters=10)[1])
            sky_var = np.nan
            if "VAR_POISSON" in hdul:
                vp = np.asarray(hdul["VAR_POISSON"].data[cut], dtype=float)
                vp = vp[np.isfinite(vp) & (vp > 0)]
                gain = NIRCAM_CHANNEL_DEFAULTS[nircam_channel(normalize_detector(h0["DETECTOR"]))]["gain"]
                t_ramp = (h0["NGROUPS"] - 1) * h0["TGROUP"]
                sky_var = float(np.median(vp)) * gain * t_ramp / row["photmjsr"] if vp.size else np.nan
            if sky_method == "auto":
                use_var = np.isfinite(sky_var) and abs(sci_med) < 0.1 * sky_var
                sky_method_used = "var_poisson" if use_var else "sci"
            else:
                sky_method_used = sky_method
            row["sky_mjysr"] = sky_var if sky_method_used == "var_poisson" else sci_med
            row["sky_method"] = sky_method_used
    return row


def exposures_from_cal(paths, *, epoch=None, measure_sky=True, sky_method="auto"):
    """Exposure table from stage-2 NIRCam ``_cal.fits`` files (one row per file)."""
    rows = []
    for p in paths:
        try:
            rows.append(read_cal_header(p, measure_sky=measure_sky, sky_method=sky_method))
        except Exception as err:
            warnings.warn(f"Skipping {p}: {err}")
    return _finalize_data_rows(pd.DataFrame(rows), epoch)


def observed_start_and_v3pa(executed, *, reference_aperture="NRCALL_FULL"):
    """Start time and V3PA of an executed visit, for building the matching APT plan.

    Uses the earliest exposure of ``executed`` (columns ``ra_v1``, ``dec_v1``, ``pa_v3`` and
    ``expstart`` or ``time``). Returns (start_mjd, v3pa_at_reference_point), which can be
    passed as ``exposures_from_apt(..., start_mjd=..., v3pa=...)``.
    """
    import pysiaf
    from pysiaf.utils import rotations

    first = executed.sort_values("time").iloc[0]
    if not np.all(np.isfinite([first["ra_v1"], first["dec_v1"], first["pa_v3"]])):
        raise ValueError("Executed table lacks RA_V1 / DEC_V1 / PA_V3.")
    ref = pysiaf.Siaf("NIRCam")[reference_aperture]
    att = rotations.attitude(0.0, 0.0, first["ra_v1"], first["dec_v1"], first["pa_v3"])
    v3pa = float(rotations.posangle(att, ref.V2Ref, ref.V3Ref)) % 360.0
    start = first["expstart"] if "expstart" in executed.columns and np.isfinite(first["expstart"]) else first["time"]
    return float(start), v3pa


# ======================================================================================
# Reference lookup: calibration values a plan does not contain
# ======================================================================================


def reference_lookup(exposures, *, columns=("photmjsr", "sky_mjysr", "pixar_sr")):
    """Median calibration values per (filter, detector) from an executed-data exposure table."""
    cols = [c for c in columns if c in exposures.columns]
    return exposures.groupby(["filter", "detector"])[cols].median().reset_index()


def apply_reference(exposures, lookup, *, by=("filter", "detector"), fallback=("filter",)):
    """Fill NaN calibration columns of ``exposures`` from ``lookup``.

    Values are matched on (filter, detector); where a pair is missing, the median over
    ``fallback`` keys (default: filter) is used. Existing non-NaN values are kept.
    """
    out = exposures.copy()
    value_cols = [c for c in lookup.columns if c not in ("filter", "detector")]
    for keys in (list(by), list(fallback)):
        ref = lookup.groupby(keys)[value_cols].median().reset_index()
        merged = out[keys].merge(ref, on=keys, how="left")
        for c in value_cols:
            if c not in out.columns:
                out[c] = np.nan
            fill = out[c].isna().to_numpy()
            out.loc[fill, c] = merged.loc[fill, c].to_numpy()
    still = [c for c in value_cols if out[c].isna().any()]
    if still:
        warnings.warn(f"Calibration values still missing for {still}; supply them before simulating.")
    return out


# ======================================================================================
# LightCurveLynx ObsTable and survey construction
# ======================================================================================


def ramp_read_noise(read_noise_cds, ngroups, nframes):
    """Effective per-pixel read noise [e-] of one up-the-ramp fitted integration
    (Rauscher et al. 2007, Eq. 1, read-noise term; white read noise)."""
    n = np.asarray(ngroups, dtype=float)
    m = np.asarray(nframes, dtype=float)
    if np.any(n < 2):
        raise ValueError("Up-the-ramp fitting requires NGROUPS >= 2.")
    return np.sqrt(12.0 * (n - 1.0) / (m * n * (n + 1.0))) * np.asarray(read_noise_cds, float) / np.sqrt(2.0)


def _obs_table_class():
    from lightcurvelynx.obstable.obs_table import ObsTable

    class NIRCamObsTable(ObsTable):
        """ObsTable for one NIRCam channel ("SW" or "LW") built from an exposure table."""

        _default_survey_values = {"survey_name": "JWST_NIRCam", "radius": None, "zp_err_mag": 0.0}

        def __init__(self, table, *, channel, psf_fwhm_arcsec=None, **kwargs):
            channel = channel.upper()
            self.channel = channel
            self.psf_fwhm_arcsec = {**NIRCAM_PSF_FWHM_ARCSEC, **(psf_fwhm_arcsec or {})}
            survey = {**NIRCAM_CHANNEL_DEFAULTS[channel], "survey_name": f"JWST_NIRCam_{channel}"}
            survey.update(kwargs.pop("survey_values", {}))
            super().__init__(table, **survey, **kwargs)

        def _derive_noise_columns(self):
            t = self._table
            for col in ("photmjsr", "pixar_sr"):
                if t[col].isna().any():
                    raise ValueError(f"Column {col!r} has NaNs; fill it with apply_reference().")
            gain = self.get_value_per_row("gain")
            pixel_scale = np.sqrt(t["pixar_sr"].to_numpy()) * (180.0 / np.pi) * 3600.0
            if "exptime" not in t.columns:
                self.add_column("exptime", t["nints"] * (t["ngroups"] - 1) * t["tgroup"])
            exptime = t["exptime"].to_numpy(float)
            if "nexposure" not in t.columns:
                self.add_column("nexposure", t["nints"].to_numpy())
            if "read_noise" not in t.columns:
                self.add_column("read_noise", ramp_read_noise(self.survey_values["read_noise_cds"],
                                                              t["ngroups"], t["nframes"]))
            if "zp" not in t.columns:
                self.add_column("zp", t["photmjsr"].to_numpy() * t["pixar_sr"].to_numpy() * 1e15 / (gain * exptime))
            if "sky_bg_e" not in t.columns:
                if t["sky_mjysr"].isna().any():
                    raise ValueError("Column 'sky_mjysr' has NaNs; fill it with apply_reference().")
                sky_e = t["sky_mjysr"].to_numpy() / t["photmjsr"].to_numpy() * gain * exptime
                self.add_column("sky_bg_e", np.clip(sky_e, 0.0, None))
            if "psf_footprint" not in t.columns:
                fwhm = t["filter"].map(self.psf_fwhm_arcsec)
                if fwhm.isna().any():
                    raise ValueError(f"No PSF FWHM for {sorted(set(t.loc[fwhm.isna(), 'filter']))}")
                self.add_column("psf_footprint", GAUSSIAN_NEA_FACTOR * (fwhm.to_numpy() / pixel_scale) ** 2)

    return NIRCamObsTable


def nircam_footprint(exposures, *, active_pixels=NIRCAM_ACTIVE_PIXELS):
    """Square single-detector footprint sized from the exposures' polygon side lengths."""
    from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint

    side = float(np.median(exposures["footprint_side_deg"]))
    return DetectorFootprint.from_sky_rect(side, side, pixel_scale=side * 3600.0 / active_pixels)


def build_nircam_surveys(exposures, passbands, *, err_scale=None, channels=("SW", "LW"), **table_kwargs):
    """One SurveyInfo per NIRCam channel present in ``exposures``.

    ``err_scale`` (e.g. {"SW": 1.3, "LW": 1.2}) multiplies the modeled errors; calibrate
    it so that predicted depths match measured ones.
    """
    from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
    from lightcurvelynx.survey_info import SurveyInfo

    cls = _obs_table_class()
    err_scale = err_scale or {}
    surveys = []
    for ch in channels:
        sub = exposures[exposures["channel"] == ch].reset_index(drop=True)
        if sub.empty:
            continue
        ot = cls(sub, channel=ch, detector_footprint=nircam_footprint(sub), **table_kwargs)
        surveys.append(SurveyInfo(obstable=ot, passbands=passbands,
                                  noise_model=PoissonFluxNoiseModel(err_scale=err_scale.get(ch, 1.0))))
    return surveys


# ======================================================================================
# Diagnostics and post-processing
# ======================================================================================


def five_sigma_depth(survey_info, *, by=("epoch", "filter")):
    """Single-exposure 5-sigma point-source depth (AB) implied by the noise model.

    Returns (per-row table, summary by ``by``). A stack of N overlapping exposures of
    equal depth reaches m5_single + 1.25 log10(N).
    """
    ot = survey_info.obstable
    idx = np.arange(len(ot._table))
    sigma = survey_info.noise_model.compute_flux_error(np.zeros(len(idx)), obs_table=ot, indices=idx)
    rows = ot._table.assign(sigma_njy=sigma, m5_single=31.4 - 2.5 * np.log10(5.0 * sigma))
    summary = (rows.groupby(list(by))
               .agg(m5_single=("m5_single", "median"), sigma_njy=("sigma_njy", "median"), n_rows=("m5_single", "size"))
               .reset_index())
    return rows, summary


def compare_exposure_tables(planned, executed, *, keys=("filter", "detector")):
    """Compare a planned (APT) and an executed (MAST/cal) exposure table.

    Returns a per-(filter, detector) table of exposure counts, mean pointing offset
    [arcsec], rotation difference [deg, mod 90] and mean time difference [days].
    """
    def summarize(df):
        g = df.groupby(list(keys))
        c = g.apply(lambda d: unit_mean(SkyCoord(d["ra"].to_numpy() * u.deg, d["dec"].to_numpy() * u.deg)),
                    include_groups=False)
        return pd.DataFrame({"n": g.size(), "center": c, "rotation": g["rotation"].median(), "time": g["time"].mean()})

    p, e = summarize(planned), summarize(executed)
    j = p.join(e, lsuffix="_plan", rsuffix="_exec", how="outer")
    both = j["center_plan"].notna() & j["center_exec"].notna()
    j["offset_arcsec"] = np.nan
    j.loc[both, "offset_arcsec"] = [a.separation(b).arcsec for a, b in zip(j.loc[both, "center_plan"], j.loc[both, "center_exec"])]
    d = (j["rotation_plan"] - j["rotation_exec"]) % 90.0
    j["d_rotation_deg"] = np.where(d > 45, d - 90, d)
    j["d_time_days"] = j["time_plan"] - j["time_exec"]
    return j.drop(columns=["center_plan", "center_exec"]).reset_index()


def stack_epochs(lightcurve, *, epoch_col="epoch", flux_col="flux"):
    """Inverse-variance combine one object's exposures within each (epoch, filter).

    Simulate with ``obstable_save_cols=["epoch"]``. Approximates photometry on the stacked
    mosaic of each epoch; ignores pixel covariance from drizzling.
    """
    lc = pd.DataFrame(lightcurve)
    lc = lc[np.isfinite(lc[flux_col]) & (lc["fluxerr"] > 0)]
    w = 1.0 / lc["fluxerr"] ** 2
    g = lc.assign(_w=w, _wf=w * lc[flux_col], _wt=w * lc["mjd"]).groupby([epoch_col, "filter"])
    out = g.agg(_w=("_w", "sum"), _wf=("_wf", "sum"), _wt=("_wt", "sum"), nexp=("_w", "size"))
    out[flux_col] = out["_wf"] / out["_w"]
    out["fluxerr"] = 1.0 / np.sqrt(out["_w"])
    out["mjd"] = out["_wt"] / out["_w"]
    return out[["mjd", flux_col, "fluxerr", "nexp"]].reset_index().sort_values("mjd", ignore_index=True)
