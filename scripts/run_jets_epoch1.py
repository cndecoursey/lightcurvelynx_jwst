"""JETS epoch 1 (year 1): build planned and executed exposure tables and compare them.

Run locally (needs network access to MAST and SVO). Edit the settings below.

    python scripts/run_jets_epoch1.py

Outputs (in outputs/jets_epoch1/):
    plan_8060_Y1E1.parquet         APT-based exposure table (JETS, PID 8060)
    exec_jets_Y1E1_mast.parquet    MAST-based exposure table for JETS (8060, NIRCam-Epoch1*)
    exec_jets_Y1E1_cal.parquet     cal-file-based table (if CAL_SUBDIR is set)
    reference_lookup.csv           PHOTMJSR / sky / pixel area per (filter, detector)
    depths_*.csv                   predicted single-exposure 5-sigma depths
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jwst_survey as js  # noqa: E402

APTX = js.APT_DIR / "8060.aptx"            # JETS APT file
PROGRAM = 8060                             # JETS
TARGET_PATTERN = "NIRCam-Epoch-1*"         # year-1 epoch-1 targets (TARGPROP, e.g. NIRCam-Epoch-1a)
CAL_SUBDIR = "jw080600_epoch1_v1.0e_cal"   # under data/cal_files/; "" to skip the cal files
OUTDIR = js.OUTPUT_DIR / "jets_epoch1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1. Plan from APT (observations 1 and 2 = epoch 1). Time and roll default to the APT
#    scheduling windows; they are replaced by the observed values in step 4 when executed
#    data are available.
plan = js.exposures_from_apt(APTX, observations=[1, 2], epoch="Y1E1", year_index=0)
print(f"APT plan: {len(plan)} detector exposures")

# 2. Executed data from MAST (keywords only, nothing downloaded).
try:
    mast = js.exposures_from_mast(PROGRAM, target_pattern=TARGET_PATTERN, epoch="Y1E1")
    mast.to_parquet(OUTDIR / "exec_jets_Y1E1_mast.parquet")
    print(f"MAST: {len(mast)} detector exposures; filters {sorted(mast['filter'].unique())}")
    print(mast.groupby(["filter", "channel"]).size())
except Exception as err:  # MAST is optional when cal files are available
    print(f"MAST step failed: {err}")
    mast = None

# 3. Optionally, cal files (adds the measured sky level per exposure).
executed = mast
if CAL_SUBDIR:
    cal = js.exposures_from_cal(js.cal_file_paths(CAL_SUBDIR), epoch="Y1E1")
    cal.to_parquet(OUTDIR / "exec_jets_Y1E1_cal.parquet")
    print(f"cal files: {len(cal)} detector exposures")
    executed = cal

if executed is None:
    raise SystemExit("Neither MAST nor cal files provided executed exposures.")

# 4. Comparison. Rebuild the plan at the observed start time and roll, then compare the
#    exposure structure and the per-detector positions and times.
if {"ra_v1", "dec_v1", "pa_v3"} <= set(executed.columns):
    start, v3pa_obs = js.observed_start_and_v3pa(executed)
    plan = js.exposures_from_apt(APTX, observations=[1, 2], epoch="Y1E1", start_mjd=start, v3pa=v3pa_obs)
    pos = js.compare_exposure_tables(plan, executed)
    print(f"\nAPT plan at observed V3PA {v3pa_obs:.3f} deg vs executed: detector offsets median "
          f"{pos.offset_arcsec.median():.3f}\", max {pos.offset_arcsec.max():.3f}\"; "
          f"time offsets max {pos.d_time_days.abs().max() * 1440:.2f} min")
plan.to_parquet(OUTDIR / "plan_8060_Y1E1.parquet")
counts = pd.concat({"plan": plan.groupby(["filter", "readpatt", "ngroups", "nints"]).size(),
                    "executed": executed.groupby(["filter", "readpatt", "ngroups", "nints"]).size()}, axis=1)
print("\nExposures per filter / readout (plan vs executed):\n", counts)

# 5. Calibration values from the executed data, applied to the plan.
lookup = js.reference_lookup(executed)
lookup.to_csv(OUTDIR / "reference_lookup.csv", index=False)
plan_filled = js.apply_reference(plan, lookup)

# 6. Surveys and predicted depths (SVO passbands).
from lightcurvelynx.astro_utils.passbands import PassbandGroup  # noqa: E402

filters = sorted(set(plan_filled["filter"]) | set(executed["filter"]))
passbands = PassbandGroup.from_svo([f"JWST/NIRCam.{f}" for f in filters])
for label, table in (("plan", plan_filled), ("executed", executed)):
    if table[["photmjsr", "sky_mjysr"]].isna().any().any():
        print(f"\n{label}: missing PHOTMJSR or sky values; skipping depth prediction")
        continue
    for s in js.build_nircam_surveys(table, passbands):
        _, summary = js.five_sigma_depth(s)
        summary.to_csv(OUTDIR / f"depths_{label}_{s.obstable.channel}.csv", index=False)
        print(f"\n{label} {s.obstable.channel} single-exposure 5-sigma depths:\n", summary.round(2))
