"""Summarize NIRCam stage-2 cal files: header keywords plus variance medians, one row per file.

Needs only numpy and astropy, so it runs in any Python environment.

    python scripts/extract_rows.py jw080600_epoch1_v1.0e_cal
    python scripts/extract_rows.py /absolute/path/to/cal_dir -o my_summary.csv

The first argument is a subdirectory of data/cal_files/ (or an absolute path). The default
output is outputs/<subdirectory>/cal_header_summary.csv. Statistics use the central
1024x1024 pixels, every second pixel.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("JWST_SURVEY_DATA", REPO_ROOT / "data")).expanduser()
OUTPUT_DIR = Path(os.environ.get("JWST_SURVEY_OUTPUT", REPO_ROOT / "outputs")).expanduser()

H0 = ["TARGPROP", "PROGRAM", "OBSERVTN", "VISIT", "EXPOSURE", "DETECTOR", "FILTER", "PUPIL", "READPATT",
      "NGROUPS", "NFRAMES", "NINTS", "TGROUP", "TFRAME", "GROUPGAP", "EFFEXPTM", "EXPSTART", "EXPMID", "EXPEND",
      "PATT_NUM", "NUMDTHPT", "SUBPXPAT", "APERNAME"]
H1 = ["PHOTMJSR", "PIXAR_SR", "S_REGION", "RA_REF", "DEC_REF", "ROLL_REF", "V3I_YANG", "PA_V3", "PA_APER",
      "RA_V1", "DEC_V1"]
CUT = (slice(512, 1536, 2), slice(512, 1536, 2))


def positive_median(a):
    a = np.asarray(a[CUT], dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if a.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cal_dir", help="subdirectory of data/cal_files/, or an absolute path")
    parser.add_argument("-o", "--output", help="output CSV (default: outputs/<cal_dir>/cal_header_summary.csv)")
    parser.add_argument("--pattern", default="*_cal*.fits")
    args = parser.parse_args()

    cal_dir = Path(args.cal_dir).expanduser()
    if not cal_dir.is_absolute():
        cal_dir = DATA_DIR / "cal_files" / cal_dir
    files = sorted(cal_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} in {cal_dir}")
    out = Path(args.output) if args.output else OUTPUT_DIR / cal_dir.name / "cal_header_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename"] + [k.lower() for k in H0 + H1] + ["sci_median", "var_poisson_med", "var_rnoise_med"])
        for f in files:
            with fits.open(f, memmap=True) as h:
                h0, h1 = h[0].header, h["SCI"].header
                row = [f.name] + [h0.get(k) for k in H0] + [h1.get(k) for k in H1]
                row += [float(np.nanmedian(h["SCI"].data[CUT])),
                        positive_median(h["VAR_POISSON"].data), positive_median(h["VAR_RNOISE"].data)]
            w.writerow(row)
    print(f"Wrote {len(files)} rows to {out}")


if __name__ == "__main__":
    main()
