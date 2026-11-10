"""Load raw sensor logs and turn them into daily / hourly tables and model features."""
import numpy as np
import pandas as pd

import config

FEATURES = [
    "lag_1", "lag_2", "lag_3",
    "roll_mean_3", "roll_mean_7", "roll_std_3",
    "peak_lag_1", "sun_hours_lag_1", "day_index",
]
# Known for the target day before it starts: clear-sky irradiation comes from the sun's
# position, cloud cover from a weather forecast (historical runs use the observed value
# as a stand-in for a perfect forecast).
WEATHER_FEATURES = ["clearsky_target", "cloud_target"]


def load_raw(source):
    """Read a raw CSV (path or file-like) and return a clean, time-sorted frame."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    if config.COL_TIME not in df.columns:
        raise ValueError(f"CSV must contain a '{config.COL_TIME}' column")

    df[config.COL_TIME] = pd.to_datetime(df[config.COL_TIME], errors="coerce")
    df = df.dropna(subset=[config.COL_TIME])

    if config.COL_POWER not in df.columns:
        if {config.COL_VOLT, config.COL_CURR} <= set(df.columns):
            df[config.COL_POWER] = df[config.COL_VOLT] * df[config.COL_CURR]
        else:
            raise ValueError(f"CSV needs '{config.COL_POWER}' or both "
                             f"'{config.COL_VOLT}' and '{config.COL_CURR}'")

    for col in (config.COL_VOLT, config.COL_CURR, config.COL_POWER, config.COL_TEMP, config.COL_LUX):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[config.COL_POWER] = df[config.COL_POWER].clip(lower=0).fillna(0)

    df = df.sort_values(config.COL_TIME).drop_duplicates(config.COL_TIME).reset_index(drop=True)
    df["power_W"] = df[config.COL_POWER] / 1000
    df["date"] = df[config.COL_TIME].dt.normalize()
    df["hour"] = df[config.COL_TIME].dt.hour
    return df


def _sample_hours(df):
    """Duration each sample represents (hours). Gaps longer than 3x the typical
    interval (logger unplugged) are capped so they don't inflate energy."""
    dt = df[config.COL_TIME].diff().shift(-1).dt.total_seconds() / 3600
    typical = dt.median() if dt.notna().any() else 1 / 60
    return dt.fillna(typical).clip(upper=3 * typical)


def to_daily(df, rated_w=config.PANEL_RATED_W):
    """Aggregate raw samples into one row per day."""
    df = df.copy()
    df["dt_h"] = _sample_hours(df)
    df["energy_Wh"] = df["power_W"] * df["dt_h"]
    df["sunny_h"] = np.where(df["power_W"] >= config.SUN_THRESHOLD * rated_w, df["dt_h"], 0)

    agg = {
        "energy_Wh": ("energy_Wh", "sum"),
        "peak_W": ("power_W", "max"),
        "mean_daylight_W": ("power_W", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        "sun_hours": ("sunny_h", "sum"),
        "samples": ("power_W", "size"),
    }
    if config.COL_TEMP in df.columns:
        agg["avg_temp_C"] = (config.COL_TEMP, "mean")
        agg["max_temp_C"] = (config.COL_TEMP, "max")
    if config.COL_LUX in df.columns:
        agg["peak_lux"] = (config.COL_LUX, "max")
    if "irradiance_Wm2" in df.columns:                      # open NASA POWER dataset
        df["irr_kWh"] = df["irradiance_Wm2"] * df["dt_h"] / 1000
        df["clr_kWh"] = df["clearsky_Wm2"] * df["dt_h"] / 1000
        df["cloud_day"] = df["cloud_pct"].where(df["clearsky_Wm2"] > 0)
        agg["irradiation_kWh_m2"] = ("irr_kWh", "sum")
        agg["clearsky_kWh_m2"] = ("clr_kWh", "sum")
        agg["cloud_pct"] = ("cloud_day", "mean")
        agg["precip_mm"] = ("precip_mm", "sum")
        agg["humidity_pct"] = ("humidity_pct", "mean")

    daily = df.groupby("date").agg(**agg)
    peak_idx = df.groupby("date")["power_W"].idxmax()
    daily["peak_time"] = df.loc[peak_idx, config.COL_TIME].dt.strftime("%H:%M").values
    daily["capacity_factor_%"] = daily["energy_Wh"] / (rated_w * 24) * 100
    if "irradiation_kWh_m2" in daily.columns:
        daily["clearness"] = daily["irradiation_kWh_m2"] / daily["clearsky_kWh_m2"]

    # Drop partial days (e.g. today while still logging): keep days with >= 60 %
    # of the typical sample count.
    full = daily["samples"] >= 0.6 * daily["samples"].median()
    daily = daily[full]
    daily.index.name = "date"
    return daily


def hourly_matrix(df):
    """Mean power (W) per date x hour-of-day (for heatmaps / profiles)."""
    return df.pivot_table(index="date", columns="hour", values="power_W", aggfunc="mean").reindex(
        columns=range(24)).fillna(0)


def build_features(daily):
    """Features known at the END of each day, used to predict the NEXT day.
    Row t holds the target energy of day t and features computed from days < t."""
    e = daily["energy_Wh"]
    feat = pd.DataFrame(index=daily.index)
    feat["target"] = e
    feat["lag_1"] = e.shift(1)
    feat["lag_2"] = e.shift(2)
    feat["lag_3"] = e.shift(3)
    feat["roll_mean_3"] = e.shift(1).rolling(3, min_periods=3).mean()
    feat["roll_mean_7"] = e.shift(1).rolling(7, min_periods=3).mean()
    feat["roll_std_3"] = e.shift(1).rolling(3, min_periods=3).std()
    feat["peak_lag_1"] = daily["peak_W"].shift(1)
    feat["sun_hours_lag_1"] = daily["sun_hours"].shift(1)
    feat["day_index"] = np.arange(len(daily))
    if "clearsky_kWh_m2" in daily.columns:
        feat["clearsky_target"] = daily["clearsky_kWh_m2"]
        feat["cloud_target"] = daily["cloud_pct"]
    return feat


def next_day_features(daily, features=FEATURES, cloud_forecast=None):
    """The feature row for the day after the last logged day. Weather inputs:
    tomorrow's clear-sky irradiation ~ today's (it changes by <1 % a day) and the
    cloud cover from a weather forecast (default: mean of the last 3 days)."""
    next_date = daily.index[-1] + pd.Timedelta(days=1)
    row = {}
    if "clearsky_kWh_m2" in daily.columns:
        row["clearsky_kWh_m2"] = daily["clearsky_kWh_m2"].iloc[-1]
        row["cloud_pct"] = daily["cloud_pct"].iloc[-3:].mean() if cloud_forecast is None else cloud_forecast
    extended = pd.concat([daily, pd.DataFrame(row, index=[next_date])])
    extended.index.name = "date"
    feats = build_features(extended)
    return feats.iloc[[-1]][features], next_date
