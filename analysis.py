"""Run the full analysis for the report: statistics, back-tests, forecast and figures.

Outputs:
    reports/results.json      every number quoted in the report
    reports/figures/*.png     the graphs used in the report and slides

Usage:
    python analysis.py
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_processing as dp
import models as m

warnings.filterwarnings("ignore")
OUT = config.ROOT / "reports"
FIG = OUT / "figures"
MODEL_COLORS = {"Moving Average": "#2a78d6", "Linear Regression": "#eb6834",
                "Decision Tree": "#1baf7a", "Random Forest": "#eda100"}
BLUE, GREY, INK, MUTED, GRID = "#2a78d6", "#a3a29d", "#0b0b0b", "#52514e", "#e6e5e1"
GRID_KG_PER_KWH = 0.71          # CEA CO2 baseline, Indian grid

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def describe(daily):
    cols = ["energy_Wh", "peak_W", "sun_hours", "irradiation_kWh_m2", "clearness", "cloud_pct",
            "precip_mm", "avg_temp_C", "humidity_pct"]
    d = daily[cols].describe().T
    d["cv_%"] = d["std"] / d["mean"] * 100
    return d.round(3)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw = dp.load_raw(config.CLEAN_HOURLY_FILE)
    daily = dp.to_daily(raw)
    hourly = dp.hourly_matrix(raw)
    e = daily["energy_Wh"]
    base, wx = list(dp.FEATURES), dp.FEATURES + dp.WEATHER_FEATURES
    td = config.BACKTEST_DAYS

    bt_base = m.backtest(daily, td, config.MA_WINDOW, features=base)
    bt_wx = m.backtest(daily, td, config.MA_WINDOW, features=wx)
    met_base, met_wx = m.metrics(bt_base), m.metrics(bt_wx)
    preds, next_date, fitted, x_next, rf_spread = m.forecast_tomorrow(daily, features=base)
    preds_wx, _, fitted_wx, _, _ = m.forecast_tomorrow(daily, features=wx)
    best = met_base.index[0]
    best_wx = met_wx.index[0]
    imp = m.feature_importance(fitted_wx, wx)
    curve = m.hourly_forecast(hourly, preds[best])

    ma_sweep = {w: m.metrics(m.backtest(daily, td, w)).loc["Moving Average", "MAE (Wh)"] for w in range(1, 8)}

    ext = dp.to_daily(dp.load_raw(config.CLEAN_DIR / f"extended_{config.CLEAN_HOURLY_FILE.name}"))
    ext_base = m.metrics(m.backtest(ext, 28, config.MA_WINDOW, features=base))
    ext_wx = m.metrics(m.backtest(ext, 28, config.MA_WINDOW, features=wx))

    quality = json.loads(config.QUALITY_REPORT_FILE.read_text())
    ext_quality = json.loads((config.CLEAN_DIR / f"extended_{config.QUALITY_REPORT_FILE.name}").read_text())
    corr = daily[["energy_Wh", "cloud_pct", "clearness", "precip_mm", "avg_temp_C", "humidity_pct",
                  "sun_hours"]].corr()["energy_Wh"].drop("energy_Wh")

    kwp_scale = 1000 / config.PANEL_RATED_W
    results = {
        "site": config.SITE_NAME, "lat": config.LATITUDE_DEG, "lon": config.LONGITUDE_DEG,
        "period": [daily.index[0].strftime("%Y-%m-%d"), daily.index[-1].strftime("%Y-%m-%d")],
        "days": len(daily), "hours": len(raw), "panel_w": config.PANEL_RATED_W,
        "quality": quality, "extended_quality": ext_quality,
        "energy": {"total_Wh": e.sum(), "mean_Wh": e.mean(), "median_Wh": e.median(), "std_Wh": e.std(),
                   "min_Wh": e.min(), "min_day": e.idxmin().strftime("%Y-%m-%d"),
                   "max_Wh": e.max(), "max_day": e.idxmax().strftime("%Y-%m-%d"),
                   "peak_W": raw["power_W"].max(),
                   "peak_time": raw.loc[raw["power_W"].idxmax(), config.COL_TIME].strftime("%Y-%m-%d %H:%M"),
                   "capacity_factor_pct": daily["capacity_factor_%"].mean(),
                   "mean_irradiation_kWh_m2": daily["irradiation_kWh_m2"].mean(),
                   "mean_clearness": daily["clearness"].mean(), "mean_cloud_pct": daily["cloud_pct"].mean(),
                   "total_rain_mm": daily["precip_mm"].sum(),
                   "days_below_20Wh": int((e < 20).sum())},
        "describe": describe(daily).reset_index().rename(columns={"index": "variable"}).to_dict("records"),
        "corr_with_energy": corr.round(3).to_dict(),
        "autocorr": {f"lag_{l}": round(e.autocorr(l), 3) for l in range(1, 6)},
        "backtest_days": td,
        "metrics_base": met_base.round(3).reset_index().to_dict("records"),
        "metrics_weather": met_wx.round(3).reset_index().to_dict("records"),
        "backtest_base": bt_base.round(3).reset_index().assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        "backtest_weather": bt_wx.round(3).reset_index().assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        "ma_window_sweep": {str(k): round(v, 3) for k, v in ma_sweep.items()},
        "extended": {"days": len(ext), "test_days": 28,
                     "metrics_base": ext_base.round(3).reset_index().to_dict("records"),
                     "metrics_weather": ext_wx.round(3).reset_index().to_dict("records")},
        "forecast": {"date": next_date.strftime("%Y-%m-%d"), "best_model": best,
                     "predictions": {k: round(v, 2) for k, v in preds.items()},
                     "rmse": met_base.loc[best, "RMSE (Wh)"], "rf_p10_p90": rf_spread,
                     "best_weather_model": best_wx,
                     "predictions_weather": {k: round(v, 2) for k, v in preds_wx.items()},
                     "weather_cloud_assumed_pct": round(daily["cloud_pct"].iloc[-3:].mean(), 1),
                     "hourly_W": curve.round(3).tolist(), "peak_hour": int(curve.idxmax())},
        "feature_importance_weather": imp.round(3).to_dict(),
        "impact": {"kwp_scale": kwp_scale, "rooftop_kWh_month": e.sum() * kwp_scale / 1000,
                   "rooftop_kWh_day": e.mean() * kwp_scale / 1000,
                   "rooftop_co2_kg_month": e.sum() * kwp_scale / 1000 * GRID_KG_PER_KWH,
                   "grid_kg_per_kwh": GRID_KG_PER_KWH},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # --- figures -------------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.bar(daily.index, e, color=BLUE, width=0.75)
    a1.axhline(e.mean(), color=MUTED, ls="--", lw=1)
    a1.text(daily.index[0], e.mean() + 0.4, f"mean {e.mean():.1f} Wh", color=MUTED, fontsize=9)
    a1.set_ylabel("Energy (Wh)")
    a1.set_title(f"Daily energy from a {config.PANEL_RATED_W:g} W panel falls on cloudy days")
    a2.bar(daily.index, daily["cloud_pct"], color=GREY, width=0.75)
    a2.set_ylabel("Cloud cover (%)")
    a2.set_ylim(0, 100)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, "fig1_daily_energy_cloud.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(hourly.values, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(hourly)), hourly.index.strftime("%d %b"), fontsize=7)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.grid(False)
    ax.set_xlabel("Hour of day (local solar time)")
    ax.set_title("Hourly panel output: generation runs 06:00 to 18:00, peaking near noon")
    fig.colorbar(im, ax=ax, label="Mean power (W)", shrink=0.8)
    save(fig, "fig2_hourly_heatmap.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x, y = daily["cloud_pct"], e
    k, c = np.polyfit(x, y, 1)
    ax.scatter(x, y, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, k * xs + c, color=MUTED, ls="--", lw=1.5)
    ax.set_xlabel("Daylight cloud cover (%)")
    ax.set_ylabel("Daily energy (Wh)")
    ax.set_title(f"Cloud cover explains {corr['cloud_pct'] ** 2:.0%} of the daily variation")
    ax.text(0.02, 0.05, f"r = {corr['cloud_pct']:.2f}; slope {k * 10:.2f} Wh per 10 % cloud",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, "fig3_cloud_vs_energy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, bt, title in [(axes[0], bt_base, "Past output only"), (axes[1], bt_wx, "With cloud-cover forecast")]:
        ax.plot(bt.index, bt["Actual"], color=INK, lw=2.5, marker="o", label="Actual", zorder=4)
        for name in m.MODEL_NAMES:
            ax.plot(bt.index, bt[name], color=MODEL_COLORS[name], lw=1.6, ls="--", marker="o", ms=4, label=name)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Energy (Wh)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Walk-forward back-test, last {td} days: weather input tracks the dips", x=0.06, y=1.04, ha="left",
                 fontweight="bold")
    save(fig, "fig4_backtest.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    order = m.MODEL_NAMES
    yb = np.arange(len(order))
    vb = [met_base.loc[n, "MAE (Wh)"] for n in order]
    vw = [met_wx.loc[n, "MAE (Wh)"] for n in order]
    ax.barh(yb + 0.2, vb, 0.38, color=GREY, label="Past output only")
    ax.barh(yb - 0.2, vw, 0.38, color=BLUE, label="With cloud-cover forecast")
    for i in range(len(order)):
        ax.text(vb[i] + 0.04, yb[i] + 0.2, f"{vb[i]:.2f}", va="center", fontsize=9, color=INK)
        ax.text(vw[i] + 0.04, yb[i] - 0.2, f"{vw[i]:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(yb, order)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (Wh per day)")
    ax.set_xlim(0, max(vb + vw) * 1.2)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("A cloud-cover forecast halves the error of the ML models")
    save(fig, "fig5_model_errors.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    s = imp["Random Forest"].sort_values()
    ax.barh(s.index, s.values, color=[BLUE if v == s.max() else GREY for v in s.values])
    ax.set_xlabel("Relative importance (Random Forest, weather inputs)")
    ax.set_title("Tomorrow's cloud cover is the input that matters")
    save(fig, "fig6_feature_importance.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hrs = np.arange(24)
    lo, hi = max(preds[best] - met_base.loc[best, "RMSE (Wh)"], 0) / preds[best], \
        (preds[best] + met_base.loc[best, "RMSE (Wh)"]) / preds[best]
    ax.fill_between(hrs, curve * lo, curve * hi, color=BLUE, alpha=0.15, lw=0, label="likely range")
    ax.plot(hrs, curve, color=BLUE, lw=2.5, label=f"forecast {next_date:%d %b} ({preds[best]:.1f} Wh)")
    ax.plot(hrs, hourly.iloc[-1].values, color=MUTED, ls=":", lw=2,
            label=f"actual {daily.index[-1]:%d %b} ({e.iloc[-1]:.1f} Wh)")
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("Power (W)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Tomorrow's forecast: {preds[best]:.1f} Wh, peak about {curve.max():.1f} W near noon")
    save(fig, "fig7_tomorrow_forecast.png")

    print(json.dumps({k: results[k] for k in ("period", "days", "energy", "corr_with_energy", "autocorr",
                                              "ma_window_sweep", "forecast", "impact")}, indent=1, default=float))
    print(met_base.round(2).to_string(), "\n", met_wx.round(2).to_string())
    print("EXT base\n", ext_base.round(2).to_string(), "\nEXT weather\n", ext_wx.round(2).to_string())


if __name__ == "__main__":
    main()
