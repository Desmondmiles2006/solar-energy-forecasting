"""Solar Energy Forecasting dashboard.

Run with:   streamlit run dashboard.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data_processing as dp
import models as m

# ---------------------------------------------------------------- palette ---
# Fixed categorical order: each model always keeps the same colour.
MODEL_COLORS = {
    "Moving Average": "#2a78d6",
    "Linear Regression": "#eb6834",
    "Decision Tree": "#1baf7a",
    "Random Forest": "#eda100",
}
PRIMARY = "#2a78d6"        # single-series marks (energy, power)
NEUTRAL = "#52514e"        # actuals, reference lines
MUTED = "#8a8984"
GRID = "#e6e5e1"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#c43d3c"], [0.25, "#ee9f98"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1, "#1c5cab"]]

LABELS = {"energy_Wh": "Energy (Wh)", "peak_W": "Peak power (W)", "sun_hours": "Sun hours (h)",
          "mean_daylight_W": "Mean daylight power (W)", "capacity_factor_%": "Capacity factor (%)",
          "avg_temp_C": "Avg temperature (°C)", "max_temp_C": "Max temperature (°C)", "peak_lux": "Peak light (lux)",
          "irradiation_kWh_m2": "Irradiation (kWh/m²)", "clearsky_kWh_m2": "Clear-sky irradiation (kWh/m²)",
          "clearness": "Clearness (actual / clear sky)", "cloud_pct": "Cloud cover (%)",
          "precip_mm": "Rainfall (mm)", "humidity_pct": "Humidity (%)"}

GRID_EMISSION_KG_PER_KWH = 0.71   # India grid average (CEA), for CO2-avoided estimate
PHONE_CHARGE_WH = 12              # one smartphone full charge

st.set_page_config(page_title="Solar Energy Forecasting", page_icon="☀️", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.hero {border-radius: 14px; padding: 18px 22px; background: #eef4fc; border: 1px solid #cde2fb;}
.hero .label {color: #52514e; font-size: .95rem;}
.hero .value {font-size: 3rem; font-weight: 700; color: #0b0b0b; line-height: 1.1;}
.hero .sub {color: #52514e; font-size: .9rem; margin-top: 6px;}
</style>""", unsafe_allow_html=True)


def style(fig, height=360, ytitle=None, xtitle=None, hover="x unified", legend=True):
    fig.update_layout(
        height=height, template="plotly_white", hovermode=hover,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0b0b0b"), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, linecolor=MUTED, zeroline=False)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, linecolor=MUTED, zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# ------------------------------------------------------------------ data ---
@st.cache_data(show_spinner=False)
def prepare(raw_bytes: bytes, rated_w: float):
    raw = dp.load_raw(io.BytesIO(raw_bytes))
    return raw, dp.to_daily(raw, rated_w), dp.hourly_matrix(raw)


@st.cache_data(show_spinner="Training and back-testing models…")
def run_models(daily, test_days, ma_window, n_trees, features, cloud_forecast):
    bt = m.backtest(daily, test_days, ma_window, n_trees, features)
    met = m.metrics(bt)
    preds, next_date, fitted, x_next, rf_spread = m.forecast_tomorrow(daily, ma_window, n_trees, features,
                                                                      cloud_forecast)
    return bt, met, preds, next_date, x_next, rf_spread, m.feature_importance(fitted, features)


with st.sidebar:
    st.header("☀️ Settings")
    source = st.radio("Data source", [f"Open data: NASA POWER ({config.SITE_NAME})", "Logged data (PV panel)",
                                      "Upload CSV"],
                      help="Open data = cleaned NASA POWER hourly data converted to panel output "
                           "(python fetch_open_data.py, then python clean_data.py). "
                           "Logged data = the CSV written by serial_logger.py.")
    raw_bytes = None
    if source.startswith("Open"):
        if config.CLEAN_HOURLY_FILE.exists():
            raw_bytes = config.CLEAN_HOURLY_FILE.read_bytes()
        else:
            st.warning("Cleaned dataset not found. Run `python fetch_open_data.py` and then "
                       "`python clean_data.py`.")
    elif source.startswith("Logged"):
        if config.LOGGED_DATA_FILE.exists():
            raw_bytes = config.LOGGED_DATA_FILE.read_bytes()
            if st.button("🔄 Reload latest readings"):
                st.rerun()
        else:
            st.warning(f"`{config.LOGGED_DATA_FILE.name}` not found yet. Start logging with "
                       "`python serial_logger.py --port COMx`.")
    else:
        up = st.file_uploader("CSV with timestamp + power_mW (or voltage_V & current_mA)", type="csv")
        if up is not None:
            raw_bytes = up.getvalue()

    st.divider()
    rated_w = st.number_input("Panel rated power (W)", 0.1, 1000.0, config.PANEL_RATED_W, 0.5)
    ma_window = st.slider("Moving-average window (days)", 2, 7, config.MA_WINDOW)
    n_trees = st.slider("Random-forest trees", 50, 500, 200, 50)

if raw_bytes is None:
    st.title("☀️ Solar Energy Forecasting")
    st.info("Choose a data source in the sidebar to begin.")
    st.stop()

try:
    raw, daily, hourly = prepare(raw_bytes, rated_w)
except Exception as exc:
    st.error(f"Could not read the data: {exc}")
    st.stop()

if len(daily) < 4:
    st.title("☀️ Solar Energy Forecasting")
    st.warning(f"Only {len(daily)} complete day(s) logged. Collect at least 4 days for the moving "
               f"average and ~{config.MIN_TRAIN_DAYS + 4} days for the ML models.")
    st.dataframe(daily, width="stretch")
    st.stop()

with st.sidebar:
    max_test = max(2, min(14, len(daily) - 4))
    test_days = st.slider("Back-test days", 2, max_test, min(config.BACKTEST_DAYS, max_test),
                          help="Each of the last N days is predicted using only the days before it.")
    features, cloud_forecast = list(dp.FEATURES), None
    if "cloud_pct" in daily.columns:
        st.divider()
        if st.checkbox("Use weather forecast (cloud cover)",
                       help="Adds tomorrow's clear-sky irradiation (from the sun's position) and forecast "
                            "cloud cover as inputs. Back-tests use the observed cloud cover as a stand-in "
                            "for a perfect forecast."):
            features += dp.WEATHER_FEATURES
            cloud_forecast = st.number_input("Tomorrow's forecast cloud cover (%)", 0.0, 100.0,
                                             float(round(daily["cloud_pct"].iloc[-3:].mean())), 5.0,
                                             help="Take it from IMD or any weather app. Default: mean of the last 3 days.")
    st.caption("Models: Moving Average · Linear Regression · Decision Tree · Random Forest")

bt, met, preds, next_date, x_next, rf_spread, importance = run_models(daily, test_days, ma_window, n_trees,
                                                                      features, cloud_forecast)

available = met.index[met.index.map(lambda n: not np.isnan(preds.get(n, np.nan)))]
best = available[0] if len(available) else "Moving Average"
best_pred = preds[best]
best_rmse = met.loc[best, "RMSE (Wh)"] if best in met.index else np.nan
low, high = max(best_pred - best_rmse, 0), best_pred + best_rmse
e = daily["energy_Wh"]
avg7 = e.iloc[-7:].mean()
interval_min = raw[config.COL_TIME].diff().median().total_seconds() / 60

# ---------------------------------------------------------------- header ---
st.title("☀️ Solar Energy Forecasting Dashboard")
st.caption(f"{daily.index[0]:%d %b %Y} → {daily.index[-1]:%d %b %Y} · {len(daily)} complete days · "
           f"{len(raw):,} readings every ~{interval_min:.0f} min · panel {rated_w:g} W · source: {source}")

hero, cards = st.columns([1.15, 2], gap="large")
with hero:
    st.markdown(f"""
<div class="hero">
  <div class="label">Tomorrow's predicted generation · <b>{next_date:%A, %d %b %Y}</b></div>
  <div class="value">{best_pred:.1f} Wh</div>
  <div class="sub">Likely range <b>{low:.1f} – {high:.1f} Wh</b> (± back-test RMSE)<br>
  Best model: <b>{best}</b> · {best_pred - avg7:+.1f} Wh vs 7-day average ({avg7:.1f} Wh)<br>
  ≈ {best_pred / PHONE_CHARGE_WH:.1f} phone charges · {best_pred / (rated_w * 24) * 100:.1f}% capacity factor</div>
</div>""", unsafe_allow_html=True)
with cards:
    cols = st.columns(2)
    for i, name in enumerate(m.MODEL_NAMES):
        p = preds.get(name, np.nan)
        mae = met.loc[name, "MAE (Wh)"] if name in met.index else np.nan
        acc = met.loc[name, "Accuracy (%)"] if name in met.index else np.nan
        with cols[i % 2]:
            st.metric(("★ " if name == best else "") + name,
                      "n/a" if np.isnan(p) else f"{p:.1f} Wh",
                      None if np.isnan(mae) else f"MAE {mae:.2f} Wh · accuracy {acc:.0f}%",
                      delta_color="off", delta_arrow="off", border=True)

st.write("")
peak_row = raw.loc[raw["power_W"].idxmax()]
k = st.columns(4)
k[0].metric("Days logged", len(daily), f"{len(raw):,} readings", delta_color="off", delta_arrow="off", border=True)
k[1].metric("Total energy", f"{e.sum():.1f} Wh", f"{e.sum() / 1000 * GRID_EMISSION_KG_PER_KWH * 1000:.0f} g CO₂ avoided",
            delta_color="off", delta_arrow="off", border=True)
k[2].metric("Average per day", f"{e.mean():.1f} Wh", f"σ = {e.std():.1f} Wh", delta_color="off", delta_arrow="off",
            border=True)
k[3].metric("Peak power", f"{peak_row['power_W']:.2f} W", f"{peak_row[config.COL_TIME]:%d %b, %H:%M}",
            delta_color="off", delta_arrow="off", border=True)
k = st.columns(4)
k[0].metric("Best day", f"{e.max():.1f} Wh", f"{e.idxmax():%a %d %b}", delta_color="off", delta_arrow="off", border=True)
k[1].metric("Worst day", f"{e.min():.1f} Wh", f"{e.idxmin():%a %d %b}", delta_color="off", delta_arrow="off", border=True)
k[2].metric("Avg capacity factor", f"{daily['capacity_factor_%'].mean():.1f} %",
            f"of {rated_w:g} W × 24 h", delta_color="off", delta_arrow="off", border=True)
k[3].metric("Avg sun hours", f"{daily['sun_hours'].mean():.1f} h",
            f"> {config.SUN_THRESHOLD:.0%} of rated power", delta_color="off", delta_arrow="off", border=True)

TAB_NAMES = ["📈 Daily generation", "🕒 Intraday patterns", "📊 Statistics",
             "🤖 Model performance", "🔮 Tomorrow's forecast", "🗂 Data"]
tab_q = st.query_params.get("tab", "")        # e.g. ?tab=4 opens the forecast tab
tabs = st.tabs(TAB_NAMES, default=TAB_NAMES[int(tab_q)] if tab_q.isdigit() and int(tab_q) < len(TAB_NAMES) else None)

# ------------------------------------------------------ tab: daily energy ---
with tabs[0]:
    fig = go.Figure()
    fig.add_bar(x=daily.index, y=e, name="Daily energy", marker_color=PRIMARY,
                hovertemplate="%{y:.1f} Wh<extra>Actual</extra>")
    fig.add_bar(x=[next_date], y=[best_pred], name=f"Forecast ({best})",
                marker=dict(color="rgba(42,120,214,0.35)", line=dict(color=PRIMARY, width=1.5),
                            pattern=dict(shape="/", fgcolor=PRIMARY)),
                error_y=dict(type="data", array=[high - best_pred], arrayminus=[best_pred - low],
                             color=NEUTRAL, thickness=1.5, width=6),
                hovertemplate="%{y:.1f} Wh<extra>Forecast</extra>")
    fig.add_scatter(x=daily.index, y=e.rolling(ma_window).mean(), name=f"{ma_window}-day moving average",
                    line=dict(color=NEUTRAL, width=2, dash="dot"), hovertemplate="%{y:.1f} Wh<extra>MA</extra>")
    fig.add_hline(y=e.mean(), line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text=f"mean {e.mean():.1f} Wh", annotation_position="top left")
    fig.update_layout(bargap=0.15)
    show(style(fig, 420, "Energy (Wh)"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peak power per day**")
        f2 = go.Figure(go.Bar(x=daily.index, y=daily["peak_W"], marker_color=PRIMARY,
                              customdata=daily["peak_time"],
                              hovertemplate="%{y:.2f} W at %{customdata}<extra></extra>"))
        show(style(f2, 280, "Power (W)", legend=False))
    with c2:
        st.markdown("**Sun hours per day**")
        f3 = go.Figure(go.Bar(x=daily.index, y=daily["sun_hours"], marker_color=PRIMARY,
                              hovertemplate="%{y:.2f} h<extra></extra>"))
        show(style(f3, 280, "Hours", legend=False))

# ------------------------------------------------------ tab: intraday -------
with tabs[1]:
    st.markdown("**Average power by hour of day (W)**: each row is one day")
    hm = go.Figure(go.Heatmap(
        z=hourly.values, x=[f"{h:02d}:00" for h in hourly.columns], y=hourly.index.strftime("%a %d %b"),
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        xgap=1, ygap=1, colorbar=dict(title="W", thickness=12),
        hovertemplate="%{y} %{x}<br>%{z:.2f} W<extra></extra>"))
    hm.update_yaxes(autorange="reversed")
    show(style(hm, max(360, 18 * len(hourly) + 80), hover="closest", legend=False))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Typical daily profile**: mean with min–max band")
        hours = hourly.columns
        prof = go.Figure()
        prof.add_scatter(x=hours, y=hourly.max(), line=dict(width=0), showlegend=False, hoverinfo="skip")
        prof.add_scatter(x=hours, y=hourly.min(), fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                         line=dict(width=0), name="min–max range", hoverinfo="skip")
        prof.add_scatter(x=hours, y=hourly.mean(), name="mean", line=dict(color=PRIMARY, width=2.5),
                         hovertemplate="%{y:.2f} W<extra>mean</extra>")
        show(style(prof, 340, "Power (W)", "Hour of day"))
    with c2:
        day_str = st.selectbox("Inspect a single day", daily.index[::-1].strftime("%Y-%m-%d").tolist(),
                               format_func=lambda d: f"{pd.Timestamp(d):%A %d %b %Y} — {e[pd.Timestamp(d)]:.1f} Wh")
        day_label = pd.Timestamp(day_str)
        day = raw[raw["date"] == day_label]
        tr = go.Figure()
        tr.add_scatter(x=day[config.COL_TIME], y=day["power_W"], name="power", fill="tozeroy",
                       fillcolor="rgba(42,120,214,0.12)", line=dict(color=PRIMARY, width=2),
                       hovertemplate="%{y:.2f} W<extra></extra>")
        show(style(tr, 300, "Power (W)", legend=False))

    if config.COL_TEMP in raw.columns and raw[config.COL_TEMP].notna().any():
        st.markdown(f"**Temperature on {day_label:%d %b}**")
        tt = go.Figure(go.Scatter(x=day[config.COL_TIME], y=day[config.COL_TEMP], line=dict(color=NEUTRAL, width=2),
                                  hovertemplate="%{y:.1f} °C<extra></extra>"))
        show(style(tt, 220, "°C", legend=False))

# ------------------------------------------------------ tab: statistics ----
with tabs[2]:
    stat_cols = [c for c in ["energy_Wh", "peak_W", "sun_hours", "mean_daylight_W", "capacity_factor_%",
                             "irradiation_kWh_m2", "clearness", "cloud_pct", "precip_mm", "humidity_pct",
                             "avg_temp_C", "max_temp_C", "peak_lux"] if c in daily.columns]
    desc = daily[stat_cols].describe().T
    desc["CV (%)"] = desc["std"] / desc["mean"] * 100
    desc["skew"] = daily[stat_cols].skew()
    desc = desc.rename(columns={"50%": "median", "25%": "Q1", "75%": "Q3"}, index=LABELS)
    st.markdown("**Descriptive statistics (per day)**")
    st.dataframe(desc.round(2), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribution of daily energy**")
        h = go.Figure(go.Histogram(x=e, nbinsx=min(12, len(e)), marker_color=PRIMARY,
                                   hovertemplate="%{x} Wh<br>%{y} days<extra></extra>"))
        h.add_vline(x=e.mean(), line=dict(color="#0b0b0b", dash="dash", width=2),
                    annotation_text=f"mean {e.mean():.1f}", annotation_position="top right")
        h.add_vline(x=e.median(), line=dict(color="#0b0b0b", dash="dot", width=2),
                    annotation_text=f"median {e.median():.1f}", annotation_position="top left")
        h.update_layout(bargap=0.08)
        show(style(h, 320, "Days", "Energy (Wh)", hover="closest", legend=False))
    with c2:
        st.markdown("**Day types** (relative to the best day)")
        ratio = e / e.max()
        kinds = pd.cut(ratio, [0, 0.4, 0.75, 1.01], labels=["Overcast / rainy (< 40%)",
                                                             "Partly cloudy (40–75%)", "Sunny (≥ 75%)"])
        counts = kinds.value_counts().reindex(kinds.cat.categories)
        dt = go.Figure(go.Bar(y=counts.index.astype(str), x=counts.values, orientation="h", marker_color=PRIMARY,
                              text=[f"{v} days · {v / len(e):.0%}" for v in counts.values], textposition="auto",
                              hovertemplate="%{x} days<extra></extra>"))
        show(style(dt, 320, None, "Days", hover="closest", legend=False))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Correlation between daily variables**")
        corr_cols = [c for c in stat_cols if c not in ("capacity_factor_%", "mean_daylight_W", "irradiation_kWh_m2")]
        corr = daily[corr_cols].corr().rename(index=LABELS, columns=LABELS)
        cm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                  colorscale=DIVERGING, text=corr.round(2).values, texttemplate="%{text}",
                                  xgap=2, ygap=2, colorbar=dict(thickness=12),
                                  hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>"))
        cm.update_yaxes(autorange="reversed")
        show(style(cm, 380, hover="closest", legend=False))
    with c2:
        st.markdown("**Autocorrelation of daily energy**")
        lags = range(1, min(8, len(e) - 2))
        acf = [e.autocorr(l) for l in lags]
        ac = go.Figure(go.Bar(x=[f"lag {l}" for l in lags], y=acf, marker_color=PRIMARY,
                              text=[f"{v:.2f}" for v in acf], textposition="outside",
                              hovertemplate="r = %{y:.2f}<extra></extra>"))
        bound = 1.96 / np.sqrt(len(e))
        for b in (bound, -bound):
            ac.add_hline(y=b, line=dict(color=MUTED, dash="dash", width=1))
        ac.update_yaxes(range=[-1, 1])
        show(style(ac, 380, "Correlation", hover="closest", legend=False))
        st.caption("Does yesterday predict today? Bars outside the dashed ±1.96/√n band are significant. Strong lag-1 "
                   "correlation means weather persists, which the lag features in the ML models use.")

    x_opts = [c for c in stat_cols if c != "energy_Wh"]
    xvar = st.selectbox("Scatter: daily energy vs", x_opts, index=0, format_func=lambda c: LABELS.get(c, c))
    xs, ys = daily[xvar], e
    slope, intercept = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    sc = go.Figure()
    sc.add_scatter(x=xs, y=ys, mode="markers", name="day",
                   marker=dict(color=PRIMARY, size=10, line=dict(color="white", width=2)),
                   text=daily.index.strftime("%d %b"), hovertemplate="%{text}<br>%{x:.2f} → %{y:.1f} Wh<extra></extra>")
    xline = np.linspace(xs.min(), xs.max(), 20)
    sc.add_scatter(x=xline, y=slope * xline + intercept, name=f"trend (r = {r:.2f}, R² = {r * r:.2f})",
                   line=dict(color=NEUTRAL, dash="dash", width=2), hoverinfo="skip")
    show(style(sc, 340, "Energy (Wh)", LABELS.get(xvar, xvar), hover="closest"))

    st.markdown("**Weekly summary**")
    weekly = daily.resample("W-SUN")["energy_Wh"].agg(["count", "sum", "mean", "min", "max", "std"])
    weekly.index = [f"Week of {max(d - pd.Timedelta(days=6), daily.index[0]):%d %b}" for d in weekly.index]
    weekly.columns = ["Days", "Total (Wh)", "Mean (Wh)", "Min (Wh)", "Max (Wh)", "Std (Wh)"]
    st.dataframe(weekly.round(2), width="stretch")

    st.markdown("**Sustainability impact of the energy logged so far**")
    kwh = e.sum() / 1000
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Energy harvested", f"{e.sum():.1f} Wh", f"since {daily.index[0]:%d %b}", delta_color="off",
              delta_arrow="off", border=True)
    i2.metric("CO₂ avoided", f"{kwh * GRID_EMISSION_KG_PER_KWH * 1000:.1f} g",
              f"@ {GRID_EMISSION_KG_PER_KWH} kg/kWh grid", delta_color="off", delta_arrow="off", border=True)
    i3.metric("Phone charges", f"{e.sum() / PHONE_CHARGE_WH:.1f}", f"@ {PHONE_CHARGE_WH} Wh each", delta_color="off", delta_arrow="off", border=True)
    i4.metric("9 W LED bulb", f"{e.sum() / 9:.1f} h", "of lighting", delta_color="off", delta_arrow="off", border=True)

# ------------------------------------------------------ tab: models --------
with tabs[3]:
    st.markdown(f"**Walk-forward back-test over the last {test_days} days**: each day predicted "
                "using only earlier data, exactly like a real next-day forecast.")
    tbl = met.copy()
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))
    st.dataframe(tbl.round(2), width="stretch",
                 column_config={"Accuracy (%)": st.column_config.ProgressColumn(
                     "Accuracy (%)", min_value=0, max_value=100, format="%.1f%%")})

    chosen = st.multiselect("Models to plot", m.MODEL_NAMES, default=m.MODEL_NAMES)
    fig = go.Figure()
    fig.add_scatter(x=bt.index, y=bt["Actual"], name="Actual", mode="lines+markers",
                    line=dict(color=NEUTRAL, width=3), marker=dict(size=9),
                    hovertemplate="%{y:.1f} Wh<extra>Actual</extra>")
    for name in chosen:
        fig.add_scatter(x=bt.index, y=bt[name], name=name, mode="lines+markers",
                        line=dict(color=MODEL_COLORS[name], width=2, dash="dash"),
                        marker=dict(size=8, line=dict(color="white", width=2)),
                        hovertemplate="%{y:.1f} Wh<extra>" + name + "</extra>")
    show(style(fig, 400, "Energy (Wh)"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mean absolute error** (lower is better)")
        order = met.index.tolist()
        mb = go.Figure(go.Bar(x=met["MAE (Wh)"], y=order, orientation="h",
                              marker_color=[MODEL_COLORS[n] for n in order],
                              text=[f"{v:.2f} Wh" for v in met["MAE (Wh)"]], textposition="outside", cliponaxis=False,
                              hovertemplate="%{x:.2f} Wh<extra></extra>"))
        mb.update_yaxes(autorange="reversed")
        mb.update_xaxes(range=[0, met["MAE (Wh)"].max() * 1.25])
        show(style(mb, 360, None, "MAE (Wh)", hover="closest", legend=False))
    with c2:
        st.markdown("**Forecast error per day** (predicted − actual)")
        eb = go.Figure()
        for name in chosen:
            eb.add_bar(x=bt.index, y=bt[name] - bt["Actual"], name=name, marker_color=MODEL_COLORS[name],
                       hovertemplate="%{y:+.1f} Wh<extra>" + name + "</extra>")
        eb.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        eb.update_layout(bargap=0.2, bargroupgap=0.08)
        show(style(eb, 360, "Error (Wh)"))

    st.markdown("**What drives the predictions?** Feature importance of the models fitted on all data")
    if not importance.empty:
        imp_model = st.radio("Model", importance.columns.tolist(), horizontal=True)
        s = importance[imp_model].sort_values()
        fi = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=MODEL_COLORS[imp_model],
                              hovertemplate="%{x:.3f}<extra></extra>"))
        show(style(fi, 320, None, "Relative importance", hover="closest", legend=False))
        st.caption("Tree models: impurity-based importance. Linear regression: |standardised coefficient| "
                   "normalised to sum to 1. lag_n = energy n days ago; roll_mean_k = mean of the last k days.")

    with st.expander("How each model works"):
        st.markdown(f"""
- **Moving Average**: tomorrow = mean of the last **{ma_window}** days. No training; a strong baseline when weather persists.
- **Linear Regression**: next-day energy = weighted sum of the features below. Ridge-regularised (small penalty on large weights, chosen by cross-validation) so it can't extrapolate wildly from a few weeks of data.
- **Decision Tree**: learns if/else splits (e.g. *"if yesterday < 18 Wh then…"*); depth limited to 3 so it can't memorise the few rows.
- **Random Forest**: averages **{n_trees}** randomised trees; usually more stable than one tree. The spread of its trees gives an uncertainty range.

**Features** (all known at the end of the day before): energy 1, 2 and 3 days ago; 3- and 7-day rolling mean;
3-day standard deviation; yesterday's peak power and sun hours; day index (trend).

**Metrics**: MAE = average absolute miss in Wh · RMSE penalises big misses · MAPE = average % miss ·
Accuracy = 100 − MAPE · Bias > 0 means the model over-predicts · R² < 0 means worse than always guessing the
test-period mean (common with only a few test days and changeable weather).
Every prediction is clamped to 0 – 1.3 × the best day logged, so no model can output a physically impossible value.
""")

# ------------------------------------------------------ tab: forecast ------
with tabs[4]:
    curve = m.hourly_forecast(hourly, best_pred)
    scale_lo = low / best_pred if best_pred else 0
    scale_hi = high / best_pred if best_pred else 0
    peak_hour = int(curve.idxmax()) if curve.max() > 0 else 12
    useful = (curve >= config.SUN_THRESHOLD * rated_w).sum()

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Predicted energy", f"{best_pred:.1f} Wh", f"{low:.1f} – {high:.1f} Wh", delta_color="off", delta_arrow="off", border=True)
    f2.metric("Expected peak", f"{curve.max():.2f} W", f"around {peak_hour:02d}:00", delta_color="off", delta_arrow="off", border=True)
    f3.metric("Useful generation", f"{useful} h", f"> {config.SUN_THRESHOLD:.0%} of rated power",
              delta_color="off", delta_arrow="off", border=True)
    f4.metric("vs yesterday", f"{best_pred - e.iloc[-1]:+.1f} Wh", f"yesterday {e.iloc[-1]:.1f} Wh",
              delta_color="off", delta_arrow="off", border=True)

    st.markdown(f"**Expected hourly power for {next_date:%A %d %b}** ({best}, shaped by the last 7 days' profile)")
    hours = [f"{h:02d}:00" for h in curve.index]
    fc = go.Figure()
    fc.add_scatter(x=hours, y=curve * scale_hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fc.add_scatter(x=hours, y=curve * scale_lo, fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                   line=dict(width=0), name="likely range", hoverinfo="skip")
    fc.add_scatter(x=hours, y=curve, name="forecast", line=dict(color=PRIMARY, width=3),
                   hovertemplate="%{y:.2f} W<extra>forecast</extra>")
    fc.add_scatter(x=hours, y=hourly.iloc[-1].values, name=f"yesterday ({daily.index[-1]:%d %b}) actual",
                   line=dict(color=NEUTRAL, width=2, dash="dot"),
                   hovertemplate="%{y:.2f} W<extra>yesterday</extra>")
    fc.update_yaxes(range=[0, max(curve.max() * scale_hi, hourly.iloc[-1].max()) * 1.1])
    show(style(fc, 380, "Power (W)", "Hour of day"))

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**All models' forecasts** (error bars = ± back-test RMSE)")
        names = [n for n in m.MODEL_NAMES if not np.isnan(preds.get(n, np.nan))]
        rmse = [met.loc[n, "RMSE (Wh)"] if n in met.index else 0 for n in names]
        pb = go.Figure(go.Bar(y=names, x=[preds[n] for n in names], orientation="h",
                              marker_color=[MODEL_COLORS[n] for n in names],
                              error_x=dict(type="data", array=rmse, color=NEUTRAL, thickness=1.5),
                              text=[f"{preds[n]:.1f} Wh" for n in names], textposition="inside", insidetextanchor="start",
                              textfont=dict(color="#0b0b0b"),
                              hovertemplate="%{x:.1f} Wh<extra></extra>"))
        pb.add_vline(x=avg7, line=dict(color=MUTED, dash="dash"), annotation_text="7-day avg")
        pb.update_yaxes(autorange="reversed")
        show(style(pb, 300, None, "Energy (Wh)", hover="closest", legend=False))
        if rf_spread:
            st.caption(f"Random-forest trees range from {rf_spread[0]:.1f} to {rf_spread[1]:.1f} Wh "
                       "(10th–90th percentile).")
    with c2:
        st.markdown("**Inputs used for this forecast**")
        st.dataframe(x_next.T.rename(columns=lambda _: "value").round(2), width="stretch")

    out = pd.DataFrame({"hour": hours, "forecast_W": curve.values.round(3),
                        "low_W": (curve * scale_lo).values.round(3), "high_W": (curve * scale_hi).values.round(3)})
    st.download_button("⬇️ Download tomorrow's hourly forecast (CSV)", out.to_csv(index=False),
                       f"forecast_{next_date:%Y-%m-%d}.csv", "text/csv")

# ------------------------------------------------------ tab: data ----------
with tabs[5]:
    st.markdown("**Daily summary**")
    st.dataframe(daily.round(2), width="stretch")
    st.download_button("⬇️ Daily summary (CSV)", daily.round(3).to_csv(), "daily_summary.csv", "text/csv")
    st.markdown("**Back-test predictions**")
    st.dataframe(bt.round(2), width="stretch")
    st.markdown(f"**Raw readings** ({len(raw):,} rows)")
    st.dataframe(raw.drop(columns=["date", "hour"]), width="stretch", height=320)
    with st.expander("Expected CSV format"):
        st.code("timestamp,voltage_V,current_mA,power_mW,temperature_C,light_lux\n"
                "2026-09-01 06:05:00,2.114,61.73,130.5,24.8,2950\n"
                "2026-09-01 06:10:00,3.020,88.10,266.1,25.0,6100", language="text")
        st.caption("Only `timestamp` and `power_mW` (or `voltage_V` + `current_mA`) are required.")
