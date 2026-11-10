"""Command-line forecast: prints model accuracy and tomorrow's prediction.

Usage:
    python forecast.py                       # logged data if present, else the cleaned NASA POWER data
    python forecast.py --weather --cloud 60  # add a cloud-cover forecast for tomorrow
    python forecast.py --data my_log.csv --ma-window 5 --test-days 10
"""
import argparse

import pandas as pd

import config
import data_processing as dp
import models as m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="raw CSV (default: logged data, falling back to sample data)")
    ap.add_argument("--ma-window", type=int, default=config.MA_WINDOW)
    ap.add_argument("--test-days", type=int, default=config.BACKTEST_DAYS)
    ap.add_argument("--weather", action="store_true", help="use clear-sky + cloud-cover features")
    ap.add_argument("--cloud", type=float, help="tomorrow's forecast cloud cover (%%) for --weather")
    args = ap.parse_args()

    path = args.data or (config.LOGGED_DATA_FILE if config.LOGGED_DATA_FILE.exists() else config.CLEAN_HOURLY_FILE)
    raw = dp.load_raw(path)
    daily = dp.to_daily(raw)
    print(f"Data: {path}  ({len(daily)} complete days, {daily.index[0]:%d %b} - {daily.index[-1]:%d %b %Y})")
    print(f"Average {daily['energy_Wh'].mean():.1f} Wh/day, total {daily['energy_Wh'].sum():.1f} Wh\n")

    features = dp.FEATURES + (dp.WEATHER_FEATURES if args.weather else [])
    met = m.metrics(m.backtest(daily, args.test_days, args.ma_window, features=features))
    with pd.option_context("display.width", 120):
        print(f"Walk-forward back-test (last {args.test_days} days):")
        print(met.round(2).to_string(), "\n")

    preds, next_date, *_ = m.forecast_tomorrow(daily, args.ma_window, features=features, cloud_forecast=args.cloud)
    best = met.index[0]
    print(f"Forecast for {next_date:%A %d %b %Y}:")
    for name, p in preds.items():
        print(f"  {'*' if name == best else ' '} {name:18s} {p:6.1f} Wh")
    rmse = met.loc[best, "RMSE (Wh)"]
    print(f"\n=> Tomorrow's approximate solar generation: {preds[best]:.1f} Wh "
          f"(likely {max(preds[best] - rmse, 0):.1f}-{preds[best] + rmse:.1f} Wh, best model: {best})")


if __name__ == "__main__":
    main()
