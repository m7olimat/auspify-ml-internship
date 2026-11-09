"""
Task 5 - Netflix trend forecasting model

What gets forecast: the number of titles Netflix added to its catalogue each
month (based on `date_added`). Steps follow the task list:
  1. build the monthly series + time features
  2. look at historical patterns (trend, seasonality, movies vs TV)
  3. build several forecasting models
  4. forecast the next 12 months with the best one
  5. evaluate with a rolling backtest (never on data the model has seen)

Run:
    python task5_forecasting.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "netflix_clean.csv"
OUT_DIR = BASE_DIR / "outputs" / "task5"

START = "2016-01-01"   # before 2016 only a handful of titles per month were added
LAGS = [1, 2, 3, 6, 12]
HORIZON = 6            # months predicted in each backtest fold
N_FOLDS = 4            # 4 x 6 = the last 24 months are used for testing
FUTURE_MONTHS = 12
SEED = 42

# colours
INK, BLUE, ORANGE, GREY, GRID, SURFACE = "#0b0b0b", "#2a78d6", "#eb6834", "#8a8984", "#e4e3de", "#fcfcfb"


# ------------------------------------------------- step 1: data + features

def load_data():
    df = pd.read_csv(DATA_PATH, usecols=["type", "date_added"], parse_dates=["date_added"])

    monthly = df.set_index("date_added").resample("MS").size()

    # the dataset stops on 25 Sep 2021, so September is only a partial month.
    # leave it out, otherwise it looks like a fake drop in the trend.
    last_day = df["date_added"].max()
    if last_day != last_day + pd.offsets.MonthEnd(0):
        monthly = monthly.iloc[:-1]

    monthly = monthly[START:].asfreq("MS")
    monthly.name = "titles_added"

    df = df[(df["date_added"] >= START) & (df["date_added"] <= monthly.index[-1] + pd.offsets.MonthEnd(0))]
    return df, monthly


def lag_features(history, date):
    """Features used to predict the month `date`. `history` ends right before that month."""
    row = {f"lag_{lag}": float(history.iloc[-lag]) for lag in LAGS}
    row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * date.month / 12)
    return row


def future_dates(train, horizon):
    return pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


# ------------------------------------------- step 2: historical patterns

def analyse_patterns(df, monthly):
    by_year_type = (df.assign(year=df["date_added"].dt.year)
                      .groupby(["year", "type"]).size().unstack(fill_value=0))

    # seasonality: how a month compares to its own year's average (full years only)
    full = monthly["2017":"2020"]
    seasonal_index = (full / full.groupby(full.index.year).transform("mean")).groupby(full.index.month).mean()

    print("Titles added per year (2021 covers Jan-Aug only):")
    print(by_year_type.assign(total=by_year_type.sum(axis=1)).to_string(), "\n")

    top = seasonal_index.sort_values(ascending=False)
    names = lambda months: ", ".join(pd.Timestamp(2000, m, 1).strftime("%b") for m in months)
    print(f"Busiest months (vs. yearly average): {names(top.index[:3])}")
    print(f"Quietest months: {names(top.index[-3:])}")

    peak_year = by_year_type.sum(axis=1).idxmax()
    print(f"Peak year so far: {peak_year}\n")
    return by_year_type, seasonal_index


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK)


def plot_patterns(monthly, by_year_type, seasonal_index, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), facecolor=SURFACE)

    ax = axes[0]
    ax.plot(monthly.index, monthly.values, color=BLUE, lw=1.6, label="Monthly additions")
    ax.plot(monthly.index, monthly.rolling(12).mean(), color=INK, lw=2, label="12-month average")
    ax.set_title("Titles added per month", loc="left", color=INK)
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1]
    ax.bar(seasonal_index.index, seasonal_index.values, color=BLUE, width=0.7)
    ax.axhline(1, color=INK, lw=1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title("Seasonality (1.0 = average month, 2017-2020)", loc="left", color=INK)
    style(ax)

    ax = axes[2]
    years = [str(y) if y != by_year_type.index.max() else f"{y}*" for y in by_year_type.index]
    bottom = np.zeros(len(by_year_type))
    for col, color in zip(["Movie", "TV Show"], [BLUE, ORANGE]):
        ax.bar(years, by_year_type[col], bottom=bottom, color=color, label=col, edgecolor=SURFACE, linewidth=1.5)
        bottom += by_year_type[col].to_numpy()
    ax.set_title("Titles added per year  (* Jan-Aug only)", loc="left", color=INK)
    ax.legend(frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------- step 3: the models
# every forecaster has the same signature: (train_series, horizon) -> array

def seasonal_naive(train, horizon):
    """Same month last year."""
    return np.array([train.iloc[-12 + (h % 12)] for h in range(horizon)], dtype=float)


def last_year_average(train, horizon):
    return np.full(horizon, train.iloc[-12:].mean())


def holt_winters(train, horizon):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels is chatty about optimiser warnings
        fit = ExponentialSmoothing(
            train.astype(float), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=12, initialization_method="estimated",
        ).fit()
    return np.clip(fit.forecast(horizon).to_numpy(), 0, None)


def lag_model(make_model):
    """Wrap an sklearn regressor so it forecasts from lags, one month at a time."""
    def forecast(train, horizon):
        rows, target = [], []
        for i in range(max(LAGS), len(train)):
            rows.append(lag_features(train.iloc[:i], train.index[i]))
            target.append(train.iloc[i])

        model = make_model()
        model.fit(pd.DataFrame(rows), target)

        # predict recursively: each prediction becomes a lag for the next month
        history = train.astype(float).copy()
        preds = []
        for date in future_dates(train, horizon):
            x = pd.DataFrame([lag_features(history, date)])
            pred = max(float(model.predict(x)[0]), 0.0)
            preds.append(pred)
            history.loc[date] = pred
        return np.array(preds)
    return forecast


def get_forecasters():
    return {
        "Seasonal naive": seasonal_naive,
        "Last-12-month average": last_year_average,
        "Holt-Winters": holt_winters,
        "Ridge (lags)": lag_model(lambda: make_pipeline(StandardScaler(), Ridge(alpha=5.0))),
        "Random Forest (lags)": lag_model(
            lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED)),
        "Gradient Boosting (lags)": lag_model(
            lambda: GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2,
                                              subsample=0.8, random_state=SEED)),
    }


# ------------------------------------------------------ step 5: evaluation

def rolling_backtest(forecasters, y):
    """Train on everything before a cut-off, predict the next HORIZON months, move the cut-off forward."""
    actual, preds = [], {name: [] for name in forecasters}
    for fold in range(N_FOLDS, 0, -1):
        cut = len(y) - fold * HORIZON
        train, test = y.iloc[:cut], y.iloc[cut:cut + HORIZON]
        actual.append(test)
        for name, fn in forecasters.items():
            preds[name].append(pd.Series(fn(train, HORIZON), index=test.index))
    return pd.concat(actual), {name: pd.concat(p) for name, p in preds.items()}


def error_metrics(actual, pred):
    err = pred - actual
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE_%": (np.abs(err) / actual).mean() * 100,
    }


def plot_backtest(actual, preds, best, path):
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    ax.plot(actual.index, actual.values, color=INK, lw=2, marker="o", ms=4, label="Actual")
    ax.plot(preds["Seasonal naive"].index, preds["Seasonal naive"].values, color=GREY, lw=1.6, ls="--", label="Seasonal naive (baseline)")
    ax.plot(preds[best].index, preds[best].values, color=BLUE, lw=2, marker="o", ms=4, label=f"{best} (best)")
    ax.set_title("Backtest: out-of-sample predictions, last 24 months", loc="left", color=INK)
    ax.set_ylabel("Titles added")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast(y, forecast, rmse, name, path):
    band = 1.28 * rmse  # roughly an 80% band if errors are about normal
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    recent = y["2018":]
    ax.plot(recent.index, recent.values, color=INK, lw=1.8, label="History")
    ax.plot(forecast.index, forecast.values, color=BLUE, lw=2, marker="o", ms=4, label=f"Forecast ({name})")
    ax.fill_between(forecast.index, (forecast - band).clip(lower=0), forecast + band, color=BLUE, alpha=0.15,
                    linewidth=0, label="approx. 80% band (from backtest RMSE)")
    ax.set_title("Forecast of titles added per month", loc="left", color=INK)
    ax.set_ylabel("Titles added")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, y = load_data()
    print(f"Monthly series: {y.index[0]:%b %Y} to {y.index[-1]:%b %Y} ({len(y)} months)\n")

    by_year_type, seasonal_index = analyse_patterns(df, y)
    plot_patterns(y, by_year_type, seasonal_index, OUT_DIR / "historical_patterns.png")

    forecasters = get_forecasters()
    actual, preds = rolling_backtest(forecasters, y)
    table = pd.DataFrame({name: error_metrics(actual, p) for name, p in preds.items()}).T.sort_values("RMSE")  # RMSE punishes big misses, which matter most for planning
    print(f"Rolling backtest ({N_FOLDS} folds x {HORIZON} months, {len(actual)} test months in total):")
    print(table.round(2).to_string(), "\n")
    table.round(3).to_csv(OUT_DIR / "model_metrics.csv")

    best = table.index[0]
    plot_backtest(actual, preds, best, OUT_DIR / "backtest.png")

    # refit the winner on the full history and look ahead
    values = forecasters[best](y, FUTURE_MONTHS)
    forecast = pd.Series(values, index=future_dates(y, FUTURE_MONTHS), name="forecast_titles_added")
    band = 1.28 * table.loc[best, "RMSE"]
    out = forecast.round(0).to_frame()
    out["lower_80"] = (forecast - band).clip(lower=0).round(0)
    out["upper_80"] = (forecast + band).round(0)
    out.index.name = "month"
    out.to_csv(OUT_DIR / "forecast_next_12_months.csv")
    plot_forecast(y, forecast, table.loc[best, "RMSE"], best, OUT_DIR / "forecast.png")

    last_12 = y.iloc[-12:].sum()
    gain = 100 * (1 - table.loc[best, "RMSE"] / table.loc["Seasonal naive", "RMSE"])
    print(f"Best model (lowest RMSE): {best}, {gain:.1f}% lower RMSE than the seasonal naive baseline")
    print(out.astype(int).to_string())
    print(f"\nExpected additions over the next 12 months: {forecast.sum():.0f} "
          f"(last 12 months: {last_12}, {100 * (forecast.sum() / last_12 - 1):+.1f}%)")
    print(f"Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
