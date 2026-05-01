# Sweet Lift Taxi — Hourly Demand Forecasting

## Description
Time-series project for **Sweet Lift Taxi**, an airport taxi company. The goal is to predict the number of taxi orders for the next hour so the operator can allocate more drivers during peak hours.

**Goal:** RMSE on the test set must be **<= 48**.

## Dataset
- **Source file:** `taxi.csv`
- **Period:** 2018-03-01 to 2018-08-31
- **Granularity:** sub-hourly orders, **resampled to 1-hour intervals** (sum)
- **Records after resample:** 4,416 hourly observations
- **Target column:** `num_orders`
- **Statistics:** mean 84.4, std 45.0, min 0, max 462

## Results
| # | Model | RMSE Train | RMSE Test | Train Time (s) | Predict Time (s) | Meets ≤ 48 |
|---|-------|-----------:|----------:|---------------:|-----------------:|:----------:|
| 1 | LinearRegression | 25.70 | 45.81 | 0.09 | 0.001 | ✅ |
| 2 | RandomForest (n=100) | 8.41 | 42.80 | 0.44 | 0.035 | ✅ |
| 3 | **LightGBM (early stopping)** | **3.87** | **39.67** | 1.04 | 0.004 | ✅ |

**Best model:** LightGBM with early stopping (best iteration 816 / 10,000) — **RMSE 39.67**, well below the 48 threshold.

## Pipeline
1. **Load & resample** — read `taxi.csv`, sort by datetime, resample to 1-hour buckets summing orders.
2. **EDA** — full-series plot, descriptive stats, seasonal decomposition (`statsmodels.seasonal_decompose`).
3. **Feature engineering** (`make_features`):
   - Calendar features: `year`, `month`, `day`, `dayofweek`, `hour`
   - **24 hourly lags** (`lag_1` … `lag_24`) — captures the full daily cycle
   - **Rolling mean window 10** (with `shift()` to prevent leakage)
4. **Train/test split** — 90 / 10 chronological split (`shuffle=False`); `dropna()` after lag generation.
5. **Scaling** — `StandardScaler` fitted on train only.
6. **Training** — three models with timing tracked:
   - LinearRegression (baseline)
   - RandomForestRegressor (`n_estimators=100`, `n_jobs=-1`)
   - LGBMRegressor with `early_stopping(200)` and `log_evaluation(200)`
7. **Evaluation** — RMSE on train and test, results table with timing and threshold compliance.

## EDA findings
- **Trend:** clear upward trend from March to August (peaks grow from ~175 to >400).
- **Daily seasonality:** strong 24-hour cycle (peak/off-peak hours).
- **Weekly seasonality:** "U" shape within each week.
- **Heteroscedasticity:** oscillation amplitude grows with the mean.
- **Residuals:** no obvious patterns left — decomposition captures the structure.

## Tech Stack
| Category | Tools |
|----------|-------|
| ML | scikit-learn, LightGBM |
| Time series | statsmodels (`seasonal_decompose`) |
| Data | pandas, NumPy |
| Visualization | matplotlib |
| Preprocessing | StandardScaler, train_test_split (`shuffle=False`) |

## Structure
```
Sprint 16/
├── README.md
├── Series temporales.ipynb    # Main notebook
└── taxi.csv                   # Dataset
```

## How to Run
1. Activate a Python environment with the required dependencies:
   ```
   pip install pandas numpy matplotlib scikit-learn lightgbm statsmodels jupyter
   ```
2. Open and run the notebook:
   ```
   jupyter notebook "Series temporales.ipynb"
   ```
   (Kernel → Restart & Run All. Total runtime: ~5 seconds.)

## Conclusions
- All three models meet the RMSE ≤ 48 target.
- **LightGBM** wins with RMSE 39.67 on test thanks to early stopping; great precision/speed trade-off.
- **RandomForest** reaches 42.80 but shows a wider train/test gap (overfitting).
- **LinearRegression** is the fastest and simplest but the least accurate (45.81).
- **Recommended model for production:** LightGBM — lowest test error, sub-second prediction, robust against overfitting via early stopping.

## Author
Fernando Muerza — TripleTen Data Science, Sprint 16.
