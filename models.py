"""Forecasting models: Moving Average, Linear Regression, Decision Tree, Random Forest.

Evaluation uses a walk-forward (expanding window) backtest: for each of the last
N days, every model is trained only on the days before it and asked to predict
that day. That mimics how the forecast is used for real ("predict tomorrow from
everything logged so far") and gives honest error numbers even with 2-4 weeks
of data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import config
from data_processing import FEATURES, WEATHER_FEATURES, build_features, next_day_features

MODEL_NAMES = ["Moving Average", "Linear Regression", "Decision Tree", "Random Forest"]


def make_ml_models(n_trees=200):
    # Shallow trees / few leaves because we only have a few dozen training rows
    return {
        # Ridge = linear regression with a penalty on large coefficients; alpha is picked by
        # leave-one-out CV. Plain least squares on ~10 rows x 9 features extrapolates wildly.
        "Linear Regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=2,
                                               random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, max_depth=4, min_samples_leaf=2,
                                               random_state=config.RANDOM_STATE, n_jobs=-1),
    }


def clamp(pred, history):
    """Keep a prediction physically plausible: 0 .. 1.3 x the best day seen so far."""
    return float(np.clip(pred, 0.0, 1.3 * history.max()))


def moving_average(history, window):
    """Forecast = mean of the last `window` days of energy."""
    return float(history.iloc[-window:].mean())


def backtest(daily, test_days=config.BACKTEST_DAYS, ma_window=config.MA_WINDOW, n_trees=200, features=FEATURES):
    """Walk-forward one-day-ahead evaluation over the last `test_days` days."""
    feats = build_features(daily)
    energy = daily["energy_Wh"]
    test_days = min(test_days, len(daily) - 1)
    rows = []
    for i in range(len(daily) - test_days, len(daily)):
        day = daily.index[i]
        row = {"date": day, "Actual": energy.iloc[i]}
        row["Moving Average"] = moving_average(energy.iloc[:i], ma_window)

        train = feats.iloc[:i][features + ["target"]].dropna()
        x_test = feats.iloc[[i]][features]
        for name, model in make_ml_models(n_trees).items():
            if len(train) >= config.MIN_TRAIN_DAYS and not x_test.isna().any(axis=None):
                model.fit(train[features], train["target"])
                row[name] = clamp(model.predict(x_test)[0], energy.iloc[:i])
            else:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def metrics(results):
    """MAE, RMSE, MAPE, R^2 and accuracy for each model in a backtest frame."""
    out = []
    actual = results["Actual"]
    for name in MODEL_NAMES:
        pred = results[name]
        mask = pred.notna()
        if mask.sum() == 0:
            continue
        a, p = actual[mask], pred[mask]
        err = p - a
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        mape = (err.abs() / a.replace(0, np.nan)).mean() * 100
        ss_res = (err ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out.append({"Model": name, "MAE (Wh)": mae, "RMSE (Wh)": rmse, "MAPE (%)": mape,
                    "R²": r2, "Accuracy (%)": max(0.0, 100 - mape), "Bias (Wh)": err.mean(),
                    "Days tested": int(mask.sum())})
    return pd.DataFrame(out).set_index("Model").sort_values("MAE (Wh)")


def forecast_tomorrow(daily, ma_window=config.MA_WINDOW, n_trees=200, features=FEATURES, cloud_forecast=None):
    """Fit every model on all data and predict the day after the last logged day.
    Returns (predictions dict, next_date, fitted ML models, feature row)."""
    feats = build_features(daily)[features + ["target"]].dropna()
    x_next, next_date = next_day_features(daily, features, cloud_forecast)
    preds = {"Moving Average": moving_average(daily["energy_Wh"], ma_window)}
    fitted = {}
    for name, model in make_ml_models(n_trees).items():
        if len(feats) >= config.MIN_TRAIN_DAYS:
            model.fit(feats[features], feats["target"])
            preds[name] = clamp(model.predict(x_next)[0], daily["energy_Wh"])
            fitted[name] = model
        else:
            preds[name] = np.nan

    # Spread of individual trees gives the Random Forest its own uncertainty band
    rf = fitted.get("Random Forest")
    rf_spread = None
    if rf is not None:
        tree_preds = np.array([t.predict(x_next.to_numpy())[0] for t in rf.estimators_])
        rf_spread = (float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90)))
    return preds, next_date, fitted, x_next, rf_spread


def feature_importance(fitted, features=FEATURES):
    """Importance per feature for the tree models and |standardised coef| for LR."""
    imp = pd.DataFrame(index=features)
    for name in ("Random Forest", "Decision Tree"):
        if name in fitted:
            imp[name] = fitted[name].feature_importances_
    if "Linear Regression" in fitted:
        coef = np.abs(fitted["Linear Regression"][-1].coef_)
        imp["Linear Regression"] = coef / coef.sum() if coef.sum() else coef
    return imp


def hourly_forecast(hourly, predicted_wh, recent_days=7):
    """Shape tomorrow's hourly power curve: average normalised daily profile of
    the recent days, scaled so its area equals the predicted daily energy."""
    recent = hourly.iloc[-recent_days:]
    totals = recent.sum(axis=1).replace(0, np.nan)
    profile = recent.div(totals, axis=0).mean().fillna(0)   # fraction of energy per hour
    return profile * predicted_wh                             # mean W in each hour == Wh in that hour
