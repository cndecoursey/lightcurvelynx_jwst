# lightcurvelynx_jwst

Observation tables for [LightCurveLynx](https://lightcurvelynx.readthedocs.io) simulations of
JWST/NIRCam imaging surveys, built from planned programs (APT files) or executed ones (MAST
keywords or stage-2 cal files). Developed for the JADES Extended Transient Survey (JETS, PID 8060).

## Layout

```
jwst_survey.py        Module: exposure-table builders, NIRCam noise model, LightCurveLynx surveys
scripts/              Analysis scripts (run from the repository root)
tests/                Tests; tests/data/ holds small test inputs
data/apt/             APT files (.aptx), committed
data/cal_files/       One subdirectory of stage-2 cal files per program/epoch (not committed)
outputs/              Generated tables and figures (not committed)
```

Data and outputs may live elsewhere: set `JWST_SURVEY_DATA` (replaces `data/`) and/or
`JWST_SURVEY_OUTPUT` (replaces `outputs/`).

## Installation

```bash
conda activate <env with Python >= 3.11>
python -m pip install -r requirements.txt
```

## Usage

```bash
python tests/test_jwst_survey.py                                  # APT geometry, timing, simulation
python scripts/extract_rows.py jw080600_epoch1_v1.0e_cal          # summarize cal files (numpy + astropy only)
python scripts/analyze_epoch1.py                                  # compare cal files with the model
python scripts/run_jets_epoch1.py                                 # APT + MAST + cal files (needs network)
```

```python
import jwst_survey as js
ex   = js.exposures_from_cal(js.cal_file_paths("jw080600_epoch1_v1.0e_cal"), epoch="Y1E1")
start, v3pa = js.observed_start_and_v3pa(ex)       # or omit to use the APT scheduling windows
plan = js.exposures_from_apt(js.APT_DIR / "8060.aptx", observations=[1, 2], epoch="Y1E1",
                             start_mjd=start, v3pa=v3pa)
plan = js.apply_reference(plan, js.reference_lookup(ex))
```

Stage-2 cal files for PID 8060 are available from MAST.
