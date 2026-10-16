"""Compare the JETS epoch-1 cal files with the jwst_survey.py models.

    python scripts/analyze_epoch1.py [path/to/cal_header_summary.csv]

The summary comes from scripts/extract_rows.py. Default: the copy in
outputs/jw080600_epoch1_v1.0e_cal/ if present, otherwise the one in tests/data/.
Results are written to outputs/jets_epoch1/.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jwst_survey as js  # noqa: E402

if len(sys.argv) > 1:
    summary = Path(sys.argv[1])
else:
    summary = js.OUTPUT_DIR / "jw080600_epoch1_v1.0e_cal" / "cal_header_summary.csv"
    if not summary.exists():
        summary = js.REPO_ROOT / "tests" / "data" / "jets_Y1E1_cal_header_summary.csv"
print(f"Using {summary}\n")
raw = pd.read_csv(summary)
df = raw.rename(columns={"expmid": "time", "filter": "filter_kw", "pupil": "pupil_kw", "targprop": "target",
                         "observtn": "observation"})
ex = js._finalize_data_rows(df, epoch="Y1E1")
ex["gain"] = ex["channel"].map({c: v["gain"] for c, v in js.NIRCAM_CHANNEL_DEFAULTS.items()})
T = (ex["ngroups"] - 1) * ex["tgroup"]

# --- 1. Geometry: pysiaf footprint from header pointing (RA_V1, DEC_V1, PA_V3) vs S_REGION.
offs, drot = [], []
for _, r in ex.iterrows():
    pr, pd_ = js.parse_s_region(js._s_region_from_v1(r["detector"], r["ra_v1"], r["dec_v1"], r["pa_v3"]))
    c = js.polygon_center_rotation(pr, pd_)
    offs.append(SkyCoord(c[0]*u.deg, c[1]*u.deg).separation(SkyCoord(r["ra"]*u.deg, r["dec"]*u.deg)).arcsec)
    d = (c[2] - r["rotation"]) % 90; drot.append(d - 90 if d > 45 else d)
print("1. pysiaf vs header S_REGION: center offset median/max = %.3f / %.3f arcsec; rotation diff max = %.4f deg"
      % (np.median(offs), np.max(offs), np.max(np.abs(drot))))
print("   APA (PA_APER) median %.3f deg; PA_V3 median %.3f deg" % (ex["pa_aper"].median(), ex["pa_v3"].median()))

# --- 2. Sequence and timing.
seq = ex.sort_values("time").drop_duplicates("time")[["time", "observation", "filter", "channel", "patt_num", "exposure"]]
sw = ex[ex.channel == "SW"].sort_values("time").drop_duplicates(["time"])
print("\n2. Exposure sequence (SW filter, dither position) in time order:")
print("  ", list(zip(sw["observation"], sw["filter"], sw["patt_num"])))
dt = np.diff(np.sort(ex.drop_duplicates("time")["time"].to_numpy())) * 86400
print("   gaps between exposure midpoints [s]: median %.0f, min %.0f, max %.0f; EFFEXPTM = %.1f s"
      % (np.median(dt), dt.min(), dt.max(), ex["effexptm"].iloc[0]))
# EXPOSURE restarts for each filter configuration, so key on (observation, filter, dither).
ue = ex[ex.channel == "SW"].groupby(["observation", "filter", "patt_num"])["time"].median().sort_values()
gaps = np.diff(ue.to_numpy()) * 86400
print("   per-exposure midpoint spacing [s]: median %.0f, min %.0f, max %.0f (EFFEXPTM %.0f s -> overhead ~%.0f s)"
      % (np.median(gaps), gaps.min(), gaps.max(), ex["effexptm"].iloc[0], np.median(gaps) - ex["effexptm"].iloc[0]))
print("   total span %.2f h; MJD %.4f - %.4f" % ((ex.time.max()-ex.time.min())*24, ex.time.min(), ex.time.max()))

# --- 3. Sky recovered from VAR_POISSON (images are background subtracted).
ex["sky_mjysr"] = ex["var_poisson_med"] * ex["gain"] * T / ex["photmjsr"]
print("\n3. SCI median after bkgsub: |median| < %.4f MJy/sr" % ex["sci_median"].abs().max())
print("   Sky from VAR_POISSON [MJy/sr] and PHOTMJSR per filter:")
print(ex.groupby("filter")[["sky_mjysr", "photmjsr"]].agg(["median", "min", "max"]).round(4).to_string())

# --- 4. Read noise: pipeline VAR_RNOISE vs Rauscher formula with JDox CDS read noise.
rn_eff = js.ramp_read_noise(ex["channel"].map({c: v["read_noise_cds"] for c, v in js.NIRCAM_CHANNEL_DEFAULTS.items()}),
                            ex["ngroups"], ex["nframes"])
pred_var = (rn_eff / T / ex["gain"] * ex["photmjsr"]) ** 2
ex["rnoise_ratio"] = np.sqrt(ex["var_rnoise_med"] / pred_var)
print("\n4. sqrt(VAR_RNOISE_pipeline / VAR_RNOISE_model) per detector (1 = agreement):")
print(ex.groupby("detector")["rnoise_ratio"].median().round(3).to_string())

# --- 5. Depths: executed table and APT plan filled from it.
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
def tophat(n, lo, hi):
    w = np.arange(lo - 500, hi + 500, 10.0)
    return Passband(np.column_stack([w, ((w >= lo) & (w <= hi)).astype(float)]), "JWST/NIRCam", n, trim_quantile=None)
pbs = PassbandGroup(given_passbands=[tophat("F115W", 10130, 12820), tophat("F150W", 13310, 16680),
      tophat("F200W", 17550, 22270), tophat("F277W", 24160, 31270), tophat("F356W", 31330, 40050), tophat("F444W", 38800, 49860)])
lookup = js.reference_lookup(ex)
lookup.to_csv(js.output_path("jets_epoch1", "reference_lookup.csv"), index=False)
start, v3pa_obs = js.observed_start_and_v3pa(ex)
plan = js.exposures_from_apt(js.APT_DIR / "8060.aptx", observations=[1, 2], epoch="Y1E1",
                             start_mjd=start, v3pa=v3pa_obs)
cmp_pos = js.compare_exposure_tables(plan, ex)
print("\n5. APT plan (8060.aptx, observed start and V3PA %.3f deg) vs cal files:" % v3pa_obs)
print("   detector offsets median %.3f\", max %.3f\"; rotation max %.3f deg; time max %.2f min"
      % (cmp_pos.offset_arcsec.median(), cmp_pos.offset_arcsec.max(), cmp_pos.d_rotation_deg.abs().max(),
         cmp_pos.d_time_days.abs().max() * 1440))
plan_f = js.apply_reference(plan, lookup)
cnt = pd.concat({"plan": plan.groupby(["filter", "readpatt", "ngroups", "nints"]).size(),
                 "JETS": ex.groupby(["filter", "readpatt", "ngroups", "nints"]).size()}, axis=1)
print("\n6a. Detector exposures per filter/readout, APT plan vs JETS cal files:\n", cnt.to_string())
rows = []
for label, tab in (("JETS (cal)", ex), ("APT plan", plan_f)):
    for s in js.build_nircam_surveys(tab, pbs, survey_values={}):
        per, summ = js.five_sigma_depth(s, by=("filter",))
        summ["source"] = label; rows.append(summ)
dep = pd.concat(rows)
dep["m5_3dither"] = dep["m5_single"] + 1.25 * np.log10(3)
print("\n6b. Predicted 5-sigma point-source depth (Gaussian NEA, no err_scale):")
print(dep.pivot(index="filter", columns="source", values=["m5_single", "m5_3dither"]).round(2).to_string())
ex.to_csv(js.output_path("jets_epoch1", "exec_jets_Y1E1_cal.csv"), index=False)
plan_f.to_csv(js.output_path("jets_epoch1", "plan_8060_Y1E1.csv"), index=False)
dep.to_csv(js.output_path("jets_epoch1", "depths_Y1E1.csv"), index=False)
