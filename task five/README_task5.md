# Task 5 - Netflix Trend Forecasting Model

Goal: forecast how many titles Netflix will add to its catalogue in the coming
months, based on the historical data.

## What I used
- Python, pandas, numpy
- scikit-learn (Ridge, Random Forest, Gradient Boosting)
- statsmodels (Holt-Winters)
- matplotlib

## What is being forecast
The number of titles added per month, taken from the `date_added` column.
Two choices I made about the data:
- I start at **January 2016**. Before that only a handful of titles were added
  each month, which would just confuse the models.
- I dropped **September 2021**. The data stops on 25 September, so that month is
  incomplete and would look like a fake drop. The final series is Jan 2016 to
  Aug 2021, 68 months.

## How it works
1. **Time features.** Lags (1, 2, 3, 6 and 12 months back) plus the month of the
   year as sine/cosine, so December and January are treated as neighbours.
2. **Historical patterns.** `historical_patterns.png` shows the monthly trend,
   seasonality and yearly totals by movies vs TV shows. Main takeaways: additions
   grew fast until 2019 (about 2,000 that year) and then levelled off, and
   October to December are the busiest months while February and May are quiet.
3. **Models.** Six in total: two simple baselines (same month last year, and the
   average of the last 12 months), Holt-Winters, and Ridge / Random Forest /
   Gradient Boosting using the lag features. The ML models predict one month at a
   time and feed their own predictions back in as lags.
4. **Future predictions.** The best model is retrained on all the data and
   predicts Sep 2021 to Aug 2022.
5. **Evaluation.** A rolling backtest. I train on everything up to a cut-off,
   predict the next 6 months, move the cut-off forward and repeat 4 times. So the
   last 24 months are always predicted by models that haven't seen them.

## How to run
    pip install -r requirements.txt
    python task5_forecasting.py

Takes a few seconds. Outputs go to `outputs/task5/`: `historical_patterns.png`,
`backtest.png`, `forecast.png`, `model_metrics.csv`, `forecast_next_12_months.csv`.

## Results
| Model | MAE | RMSE |
|---|---|---|
| Random Forest (lags) | 36.0 | 43.3 |
| Last-12-month average | 35.1 | 44.1 |
| Ridge (lags) | 37.5 | 44.9 |
| Gradient Boosting (lags) | 41.2 | 48.1 |
| Seasonal naive | 36.7 | 48.3 |
| Holt-Winters | 40.8 | 50.1 |

I picked the best model by RMSE, which is Random Forest. It predicts about
1,850 titles over the next 12 months, around 6% below the last 12 months (1,973).

## Limitations (please read)
- The Random Forest is only about 10% better than the seasonal naive baseline on
  RMSE, and on MAE the plain 12-month average is actually slightly better. In
  other words the ML models don't clearly beat the simple ones here.
- The reasons are a short history (68 months) and a noisy series that stopped
  growing after 2019, so there isn't much trend to learn from.
- The shaded band on the forecast chart is rough: it is the forecast plus or
  minus 1.28 x the backtest RMSE, roughly an 80% range.
- This forecasts when titles were added to Netflix, not when they were released.
- The data ends in September 2021 and says nothing about anything after that.
