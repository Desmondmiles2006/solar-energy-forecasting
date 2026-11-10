"""Download the raw open dataset: hourly NASA POWER data for the study site.

NASA POWER (https://power.larc.nasa.gov) publishes satellite-derived solar
irradiance (CERES SYN1deg) and weather (MERRA-2) for any point on Earth. The data
is free to use; NASA asks only that it is cited. No API key is needed.

The file is saved exactly as the API returns it (header block included) so the
raw dataset can be submitted unchanged next to the cleaned one.

Usage:
    python fetch_open_data.py                                  # study window from config.py
    python fetch_open_data.py --start 2026-01-01 --end 2026-08-30 --tag extended
"""
import argparse
import urllib.parse
import urllib.request

import config

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   # global horizontal irradiance, all sky (Wh/m^2 per hour)
    "CLRSKY_SFC_SW_DWN",   # same under a clear sky
    "ALLSKY_KT",           # clearness index
    "CLOUD_AMT",           # cloud amount (%)
    "T2M",                 # air temperature at 2 m (C)
    "RH2M",                # relative humidity at 2 m (%)
    "WS2M",                # wind speed at 2 m (m/s)
    "PRECTOTCORR",         # precipitation (mm/hour)
]


def raw_path(start, end, tag=None):
    name = f"nasa_power_hourly_{config.SITE_NAME.lower()}_{start.replace('-', '')}_{end.replace('-', '')}"
    return config.RAW_DIR / f"{name}{'_' + tag if tag else ''}.csv"


def fetch(start=config.STUDY_START, end=config.STUDY_END, lat=config.LATITUDE_DEG,
          lon=config.LONGITUDE_DEG, tag=None):
    query = urllib.parse.urlencode({
        "parameters": ",".join(PARAMETERS), "community": "RE",
        "latitude": lat, "longitude": lon,
        "start": start.replace("-", ""), "end": end.replace("-", ""),
        "format": "CSV", "time-standard": "LST",
    })
    with urllib.request.urlopen(f"{API}?{query}", timeout=180) as resp:
        body = resp.read()
    out = raw_path(start, end, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=config.STUDY_START)
    ap.add_argument("--end", default=config.STUDY_END)
    ap.add_argument("--lat", type=float, default=config.LATITUDE_DEG)
    ap.add_argument("--lon", type=float, default=config.LONGITUDE_DEG)
    ap.add_argument("--tag", help="suffix for the file name, e.g. 'extended'")
    args = ap.parse_args()
    out = fetch(args.start, args.end, args.lat, args.lon, args.tag)
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
