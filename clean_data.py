"""Clean the raw NASA POWER download and convert it to PV panel output.

Steps (each one is counted in data/cleaned/data_quality_report.json):
  1. Parse the header block and build a timestamp from YEAR, MO, DY, HR (local solar time)
  2. Remove duplicate hours and insert any missing hours so the series is complete
  3. Replace the -999 "missing" sentinel with NaN
  4. Range checks: physically impossible values become NaN
  5. Night hours (clear-sky irradiance = 0): irradiance and clearness index set to 0
  6. Fill short gaps: weather by linear interpolation (<= 6 h), daytime irradiance
     as clear-sky irradiance x interpolated clearness index; filled hours are flagged
  7. Drop days with more than 4 filled daylight hours
  8. Convert irradiance to the output of the small prototype panel (PV model below)
  9. Aggregate to one row per day

PV model (standard NOCT model, horizontal panel):
    T_cell = T_air + (NOCT - 20) / 800 * G
    P      = P_rated * G / 1000 * (1 + gamma * (T_cell - 25)) * derate

Usage:
    python clean_data.py                         # study window
    python clean_data.py --raw data/raw/<file>.csv --out-prefix extended
"""
import argparse
import json

import numpy as np
import pandas as pd
from openpyxl.styles import Font

import config
import data_processing as dp
from fetch_open_data import raw_path

RENAME = {
    "ALLSKY_SFC_SW_DWN": "irradiance_Wm2",
    "CLRSKY_SFC_SW_DWN": "clearsky_Wm2",
    "ALLSKY_KT": "clearness_kt",
    "CLOUD_AMT": "cloud_pct",
    "T2M": config.COL_TEMP,
    "RH2M": "humidity_pct",
    "WS2M": "wind_ms",
    "PRECTOTCORR": "precip_mm",
}
# Physically possible range for each variable; anything outside becomes NaN
VALID_RANGE = {
    "irradiance_Wm2": (0, 1400), "clearsky_Wm2": (0, 1400), "clearness_kt": (0, 1),
    "cloud_pct": (0, 100), config.COL_TEMP: (-10, 55), "humidity_pct": (0, 100),
    "wind_ms": (0, 40), "precip_mm": (0, 150),
}
WEATHER = ["cloud_pct", config.COL_TEMP, "humidity_pct", "wind_ms", "precip_mm"]


def read_raw(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    end = lines.index("-END HEADER-")
    header = [l.strip() for l in lines[1:end] if l.strip()]
    df = pd.read_csv(path, skiprows=end + 1)
    return df, header


def clean(raw, rated_w=config.PANEL_RATED_W):
    q = {"rows_downloaded": len(raw)}
    df = raw.copy()
    df[config.COL_TIME] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY)) + pd.to_timedelta(df.HR, unit="h")
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"]).rename(columns=RENAME)

    # 2. duplicates and missing hours
    q["duplicate_hours_removed"] = int(df.duplicated(config.COL_TIME).sum())
    df = df.drop_duplicates(config.COL_TIME).set_index(config.COL_TIME).sort_index()
    full = pd.date_range(df.index.min().normalize(), df.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    q["missing_hours_inserted"] = int(len(full.difference(df.index)))
    df = df.reindex(full)
    df.index.name = config.COL_TIME
    night = df["clearsky_Wm2"].fillna(0) <= 0

    # 3. sentinel
    sentinel = (df == -999)
    q["sentinel_-999_by_column"] = {c: int(n) for c, n in sentinel.sum().items() if n}
    q["sentinel_-999_at_night_clearness"] = int((sentinel["clearness_kt"] & night).sum())
    df = df.mask(sentinel)

    # 4. range checks
    q["out_of_range_set_to_nan"] = {}
    for col, (lo, hi) in VALID_RANGE.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if bad.any():
            q["out_of_range_set_to_nan"][col] = {"count": int(bad.sum()),
                                                 "values": sorted(set(df.loc[bad, col].round(2).tolist()))[:10]}
        df.loc[bad, col] = np.nan
    over = df["irradiance_Wm2"] > df["clearsky_Wm2"] * 1.1 + 20
    q["irradiance_above_clearsky_set_to_nan"] = int(over.sum())
    df.loc[over, "irradiance_Wm2"] = np.nan

    # 5. night
    df.loc[night, ["irradiance_Wm2", "clearness_kt"]] = 0.0

    # 6. gap filling
    df["imputed"] = 0
    for col in WEATHER + ["clearsky_Wm2"]:
        gaps = df[col].isna()
        df[col] = df[col].interpolate(limit=6, limit_area="inside")
        df.loc[gaps & df[col].notna(), "imputed"] = 1
    gaps = df["irradiance_Wm2"].isna() & ~night
    kt = df["clearness_kt"].where(~night).interpolate(limit=3, limit_area="inside")
    df.loc[gaps, "irradiance_Wm2"] = df.loc[gaps, "clearsky_Wm2"] * kt[gaps]
    df.loc[gaps & df["irradiance_Wm2"].notna(), "imputed"] = 1
    df["clearness_kt"] = df["clearness_kt"].fillna(kt)
    q["hours_imputed"] = int(df["imputed"].sum())

    # 7. drop days with too many filled daylight hours (or still missing)
    bad_hour = (df["imputed"].eq(1) & ~night) | df[["irradiance_Wm2", config.COL_TEMP]].isna().any(axis=1)
    bad_days = bad_hour.groupby(df.index.normalize()).sum()
    drop = bad_days[bad_days > 4].index
    q["days_dropped"] = [d.strftime("%Y-%m-%d") for d in drop]
    df = df[~df.index.normalize().isin(drop)]

    # 8. PV model
    g = df["irradiance_Wm2"]
    df["cell_temp_C"] = df[config.COL_TEMP] + (config.NOCT_C - 20) / 800 * g
    power_w = rated_w * g / 1000 * (1 + config.TEMP_COEFF * (df["cell_temp_C"] - 25)) * config.SYSTEM_DERATE
    df[config.COL_POWER] = (power_w.clip(lower=0) * 1000).round(2)

    df = df.reset_index()
    cols = [config.COL_TIME, config.COL_POWER, "irradiance_Wm2", "clearsky_Wm2", "clearness_kt", "cloud_pct",
            config.COL_TEMP, "cell_temp_C", "humidity_pct", "wind_ms", "precip_mm", "imputed"]
    df = df[cols].round({"irradiance_Wm2": 2, "clearsky_Wm2": 2, "clearness_kt": 3, "cloud_pct": 2,
                         config.COL_TEMP: 2, "cell_temp_C": 2, "humidity_pct": 2, "wind_ms": 2, "precip_mm": 3})
    q["rows_clean"] = len(df)
    q["days_clean"] = int(df[config.COL_TIME].dt.normalize().nunique())
    q["period"] = [df[config.COL_TIME].min().strftime("%Y-%m-%d"), df[config.COL_TIME].max().strftime("%Y-%m-%d")]
    return df, q


DICTIONARY = [
    ("timestamp", "Start of the hour, local solar time (about 15 min behind IST at this longitude)", "-"),
    ("power_mW", "Modelled output of a 5 W panel, mean over the hour (NOCT PV model)", "mW"),
    ("irradiance_Wm2", "Global horizontal irradiance, all sky (NASA ALLSKY_SFC_SW_DWN)", "W/m2"),
    ("clearsky_Wm2", "Irradiance if the sky were clear (CLRSKY_SFC_SW_DWN)", "W/m2"),
    ("clearness_kt", "Clearness index: share of extraterrestrial radiation reaching the ground (0 at night)", "-"),
    ("cloud_pct", "Cloud amount (CLOUD_AMT)", "%"),
    ("temperature_C", "Air temperature at 2 m (T2M)", "C"),
    ("cell_temp_C", "Estimated panel cell temperature", "C"),
    ("humidity_pct", "Relative humidity at 2 m (RH2M)", "%"),
    ("wind_ms", "Wind speed at 2 m (WS2M)", "m/s"),
    ("precip_mm", "Precipitation, corrected (PRECTOTCORR)", "mm/h"),
    ("imputed", "1 if any value in the row was filled during cleaning", "flag"),
    ("energy_Wh (daily)", "Panel energy for the day: sum of hourly power", "Wh"),
    ("irradiation_kWh_m2 (daily)", "Solar energy received per square metre", "kWh/m2"),
    ("clearsky_kWh_m2 (daily)", "Clear-sky irradiation, known in advance from sun position", "kWh/m2"),
    ("cloud_pct (daily)", "Mean cloud amount over daylight hours", "%"),
]


def write_workbook(raw, header, hourly, daily, quality, path=config.WORKBOOK_FILE):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        readme = pd.DataFrame({"Solar energy forecasting dataset": [
            f"Site: {config.SITE_NAME}, Tamil Nadu ({config.LATITUDE_DEG} N, {config.LONGITUDE_DEG} E)",
            f"Period: {quality['period'][0]} to {quality['period'][1]} ({quality['days_clean']} days, hourly)",
            "Source: NASA POWER hourly API, https://power.larc.nasa.gov (CERES SYN1deg + MERRA-2)",
            f"Panel model: {config.PANEL_RATED_W:g} W, NOCT {config.NOCT_C:g} C, "
            f"temp. coeff {config.TEMP_COEFF}, derate {config.SYSTEM_DERATE}",
            "Sheets: raw_hourly (as downloaded), clean_hourly, clean_daily, data_dictionary, cleaning_log",
            "", "Raw file header:", *header]})
        readme.to_excel(xl, sheet_name="readme", index=False)
        raw.to_excel(xl, sheet_name="raw_hourly", index=False)
        hourly.to_excel(xl, sheet_name="clean_hourly", index=False)
        daily.reset_index().to_excel(xl, sheet_name="clean_daily", index=False)
        pd.DataFrame(DICTIONARY, columns=["Column", "Meaning", "Unit"]).to_excel(
            xl, sheet_name="data_dictionary", index=False)
        log = [(k, json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in quality.items()]
        pd.DataFrame(log, columns=["Check", "Result"]).to_excel(xl, sheet_name="cleaning_log", index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 90)
            for c in ws[1]:
                c.font = Font(bold=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(raw_path(config.STUDY_START, config.STUDY_END)))
    ap.add_argument("--out-prefix", default="", help="prefix for output files (e.g. 'extended')")
    ap.add_argument("--no-workbook", action="store_true")
    args = ap.parse_args()

    raw, header = read_raw(args.raw)
    hourly, quality = clean(raw)
    quality["source_file"] = args.raw.replace("\\", "/").split("/")[-1]
    pre = f"{args.out_prefix}_" if args.out_prefix else ""
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    hourly_path = config.CLEAN_DIR / f"{pre}{config.CLEAN_HOURLY_FILE.name}"
    hourly.to_csv(hourly_path, index=False)
    daily = dp.to_daily(dp.load_raw(hourly_path))
    daily.round(4).to_csv(config.CLEAN_DIR / f"{pre}{config.CLEAN_DAILY_FILE.name}")
    (config.CLEAN_DIR / f"{pre}{config.QUALITY_REPORT_FILE.name}").write_text(json.dumps(quality, indent=2))
    if not args.no_workbook and not pre:
        write_workbook(raw, header, hourly, daily, quality)
    print(json.dumps(quality, indent=2))
    print(f"\nClean: {len(hourly)} hourly rows, {len(daily)} days, "
          f"mean {daily['energy_Wh'].mean():.1f} Wh/day")


if __name__ == "__main__":
    main()
