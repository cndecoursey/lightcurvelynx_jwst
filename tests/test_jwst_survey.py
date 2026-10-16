"""Tests of jwst_survey.py with the JETS APT file (PID 8060), year 1, epoch 1.

Sections 1-4 use SYNTHETIC PHOTMJSR and sky values (placeholders; real values come from
reference_lookup() applied to executed data). Section 5 compares the APT plan with the
real epoch-1 cal-file headers in tests/data/.

    python tests/test_jwst_survey.py [path/to/file.aptx]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jwst_survey as js  # noqa: E402

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.models.basic_models import ConstantSEDModel
from lightcurvelynx.simulate import simulate_lightcurves

APTX = Path(sys.argv[1]) if len(sys.argv) > 1 else js.APT_DIR / "8060.aptx"

# 0. Angle ranges that wrap through 0/360 deg (as written by the APT Visit Planner).
assert abs(js._range_midpoint([(241.5, 248.8)]) - 245.15) < 1e-9
assert 353.0 < js._range_midpoint([(0.0, 0.810222), (346.832, 359.9999)]) < 354.0

# 1. Plan: epoch 1 of year 1 = APT observations 1 and 2, at a fixed V3PA for reproducibility.
plan = js.exposures_from_apt(APTX, observations=[1, 2], epoch="Y1E1", v3pa=241.0)
assert len(plan) == 2 * 3 * 3 * 10, len(plan)
assert set(plan["filter"]) == {"F115W", "F150W", "F200W", "F277W", "F356W", "F444W"}
assert np.allclose(plan["v3pa"], 241.0, atol=1e-6)
print("Plan:", len(plan), "detector exposures; MJD", round(plan.time.min(), 3), "-", round(plan.time.max(), 3))

# 2. V1-pointing fallback (used when MAST lacks S_REGION) reproduces the direct geometry.
import pysiaf  # noqa: E402
from pysiaf.utils import rotations  # noqa: E402

ob = js.read_apt(APTX).set_index("obs").loc[1]
ref = pysiaf.Siaf("NIRCam")["NRCALL_FULL"]
att = rotations.attitude(ref.V2Ref, ref.V3Ref, ob["ra"], ob["dec"], 241.0)
ra_v1, dec_v1 = rotations.pointing(att, 0.0, 0.0)
pa_v3_at_v1 = rotations.posangle(att, 0.0, 0.0)
direct = js.nircam_detector_polygons(ob["ra"], ob["dec"], 241.0)
for det in ("NRCA1", "NRCB5"):
    poly = js.parse_s_region(js._s_region_from_v1(det, ra_v1, dec_v1, pa_v3_at_v1))
    sep = SkyCoord(poly[0] * u.deg, poly[1] * u.deg).separation(SkyCoord(direct[det][0] * u.deg, direct[det][1] * u.deg))
    assert sep.max().arcsec < 1e-3, sep.arcsec
print("V1-pointing fallback matches direct pysiaf geometry (< 1 mas).")

# 3. Mock 'executed' table: the plan observed 2 days earlier with a 0.3 deg roll error,
#    passed through the same post-processing as MAST / cal rows.
exec_rows = []
for o in (1, 2):
    obr = js.read_apt(APTX).set_index("obs").loc[o]
    polys = js.nircam_detector_polygons(obr["ra"], obr["dec"], 241.3)
    for _, r in plan[plan.apt_obs == o].iterrows():
        ra, dec = polys[r["detector"]]
        exec_rows.append({
            "time": r["time"] - 2.0, "s_region": "POLYGON ICRS " + " ".join(f"{a} {b}" for a, b in zip(ra, dec)),
            "filter_kw": r["filter"], "pupil_kw": "CLEAR",
            "detector": r["detector"].replace("5", "LONG"), "ngroups": 7, "nframes": 8, "nints": 1,
            "tgroup": r["tgroup"], "photmjsr": {"SW": 2.5, "LW": 0.45}[r["channel"]] * 1.02,  # SYNTHETIC
            "pixar_sr": r["pixar_sr"], "sky_mjysr": {"SW": 0.10, "LW": 0.30}[r["channel"]],  # SYNTHETIC
        })
executed = js._finalize_data_rows(pd.DataFrame(exec_rows), epoch="Y1E1")
cmp = js.compare_exposure_tables(plan, executed)
print("\nPlan vs executed (per filter, detector):")
print(cmp[["filter", "detector", "n_plan", "n_exec", "offset_arcsec", "d_rotation_deg", "d_time_days"]].head(6).round(3))
assert (cmp["n_plan"] == cmp["n_exec"]).all()
assert np.allclose(cmp["d_rotation_deg"], -0.3, atol=0.02)
assert np.allclose(cmp["d_time_days"], 2.0)

# 4. Fill the plan's calibration columns from the executed data, build surveys, simulate.
lookup = js.reference_lookup(executed)
plan_filled = js.apply_reference(plan, lookup)
assert not plan_filled[["photmjsr", "sky_mjysr"]].isna().any().any()


def tophat(name, lo, hi):
    w = np.arange(lo - 500, hi + 500, 10.0)
    return Passband(np.column_stack([w, ((w >= lo) & (w <= hi)).astype(float)]), "JWST/NIRCam", name,
                    trim_quantile=None)


passbands = PassbandGroup(given_passbands=[
    tophat("F115W", 10130, 12820), tophat("F150W", 13310, 16680), tophat("F200W", 17550, 22270),
    tophat("F277W", 24160, 31270), tophat("F356W", 31330, 40050), tophat("F444W", 38800, 49860)])
surveys = js.build_nircam_surveys(plan_filled, passbands)

# A source on SW detector A2 of obs 1 (and on LW A5 of obs 1).
ra_a2, dec_a2 = direct["NRCA2"]
src_pos = SkyCoord(ra_a2.mean() * u.deg, dec_a2.mean() * u.deg)
src = ConstantSEDModel(brightness=50.0, t0=0.0, ra=src_pos.ra.deg, dec=src_pos.dec.deg, redshift=0.0, node_label="src")
lc = simulate_lightcurves(src, 1, surveys, obstable_save_cols=["epoch", "detector"],
                          rng=np.random.default_rng(3)).iloc[0]["lightcurve"]
print("\nSource on NRCA2/obs 1 observed by:", lc.groupby(["filter", "detector"]).size().to_dict())
assert set(lc["detector"]) == {"NRCA2", "NRCA5"} and len(lc) == 6 * 3
st = js.stack_epochs(lc)
print("\nStacked epoch photometry:\n", st.round(3))

for s in surveys:
    print(f"\n{s.obstable.channel} single-exposure depth (SYNTHETIC calibration):\n",
          js.five_sigma_depth(s)[1].round(2))
# 5. Real data: the APT plan at the observed start time and roll reproduces the JETS
#    epoch-1 detector positions and times from the cal-file headers.
raw = pd.read_csv(Path(__file__).resolve().parent / "data" / "jets_Y1E1_cal_header_summary.csv")
observed = js._finalize_data_rows(raw.rename(columns={"expmid": "time", "filter": "filter_kw", "pupil": "pupil_kw",
                                                      "observtn": "observation"}), epoch="Y1E1")
start, v3pa_obs = js.observed_start_and_v3pa(observed)
plan_obs = js.exposures_from_apt(APTX, observations=[1, 2], epoch="Y1E1", start_mjd=start, v3pa=v3pa_obs)
cmp_obs = js.compare_exposure_tables(plan_obs, observed)
print(f"\nAPT vs JETS epoch 1: V3PA {v3pa_obs:.3f} deg; detector offsets median "
      f"{cmp_obs.offset_arcsec.median():.3f}\", max {cmp_obs.offset_arcsec.max():.3f}\"; "
      f"time offsets max {cmp_obs.d_time_days.abs().max() * 1440:.2f} min")
assert (cmp_obs["n_plan"] == cmp_obs["n_exec"]).all()
assert cmp_obs["offset_arcsec"].max() < 0.5          # sub-pixel dither offsets are not modeled
assert cmp_obs["d_rotation_deg"].abs().max() < 0.5
assert cmp_obs["d_time_days"].abs().max() * 1440 < 2.0

print("\nALL CHECKS PASSED")
