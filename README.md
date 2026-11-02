# Retail Demand Forecasting for Inventory Optimization

An end-to-end demand forecasting project that predicts **daily unit sales for 500 store–item series** using lag/rolling features and XGBoost, validated with a strict time-based split and benchmarked against naive baselines.

**Live demo:** _add your Streamlit Community Cloud link here_

---

## Results at a glance

Validation period: **all of 2017** (182,500 store–item–day observations), never seen during training.

| Metric | XGBoost |
|---|---:|
| MAE | **6.13** |
| RMSE | **7.96** |
| WAPE | **10.42%** |
| MAPE (excluding zero-sales days) | 12.64% |
| Mean error (bias) | +0.05 units |

| Model (same validation set) | MAE | WAPE |
|---|---:|---:|
| Naive (yesterday's sales, `lag_1`) | 11.24 | 19.12% |
| Seasonal naive (same weekday last week, `lag_7`) | 8.84 | 15.03% |
| 7-day rolling mean | 8.98 | 15.27% |
| Random Forest | 6.53 | – |
| **XGBoost** | **6.13** | **10.42%** |

XGBoost reduces error by about **31% versus the seasonal-naive baseline** and shows almost no systematic over- or under-forecasting (bias ≈ 0).

> **Forecast setting:** features such as `lag_1` and the rolling statistics use the actual previous days' sales, so these results measure **one-day-ahead forecasts**. Forecasting several days ahead would need recursive prediction or direct multi-horizon models (see [Future work](#future-work)).

---

## Business problem

Retailers must decide how much inventory to stock each day. Poor forecasts cause:

- **Stockouts** → lost revenue
- **Overstocking** → storage cost and wastage

This project builds a store–item level forecasting workflow and an error analysis that shows *where* the forecasts can be trusted and where they need human review.

---

## Dataset

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle)

- 913,000 daily records, **2013-01-01 to 2017-12-31**
- 10 stores × 50 items = 500 time series
- No missing values or duplicate rows; sales range from 0 to 231 units (mean ≈ 52)

The raw data is not stored in this repo. Download `train.csv` from Kaggle and place it at `data/raw/train.csv`.

---

## Approach

| Step | Notebook | What it does |
|---|---|---|
| 1. Validation & cleaning | `notebooks/1.ipynb` | Type checks, missing/duplicate checks, date range, sorting by store–item–date |
| 2. EDA | `notebooks/02_eda.ipynb` | Daily trend, monthly and day-of-week seasonality, store and item demand, rolling statistics |
| 3. Feature engineering | `notebooks/03_feature_engineering.ipynb` | Calendar, lag and rolling features; time-based split |
| 4. Baselines & classical models | `notebooks/04_baseline_and_classical_models.ipynb` | Naive, moving average, ARIMA, SARIMA |
| 5. ML models | `notebooks/05_ml_models.ipynb` | Random Forest and XGBoost, feature importance |
| 6. Error analysis | `notebooks/06_error_analysis.ipynb` | Errors by demand level and store–item pair; saves predictions for the dashboard |
| 7. Dashboard | `app/app.py` | Streamlit app for forecast review |

### Features (18 total)

- **Series identifiers:** `store`, `item`
- **Calendar:** `year`, `month`, `week`, `day`, `day_of_week`, `is_weekend`
- **Lags:** sales 1, 7, 14 and 28 days ago
- **Rolling statistics:** mean and standard deviation over 7, 14 and 28 days

Rolling windows are computed on `shift(1)` within each store–item group, so a row never sees its own day's sales (no target leakage). The first 28 days of each series are dropped because their lags are undefined.

### Validation strategy

A **time-based split** is used instead of a random split, because random splitting would leak future information into training:

| Set | Period | Rows |
|---|---|---:|
| Train | before 2017-01-01 | 716,500 |
| Validation | 2017-01-01 to 2017-12-31 | 182,500 |

### Final model

```python
XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42,
)
```

A 500-tree version scored MAE 6.10 / RMSE 7.93 in `05_ml_models.ipynb`. The 300-tree model is used for the error analysis and the dashboard.

Hyperparameters were set by hand and not tuned with cross-validation.

---

## Error analysis

**Error grows with demand level.** Validation MAE by sales quartile:

| Demand level | MAE |
|---|---:|
| Low | 4.27 |
| Medium | 5.42 |
| High | 6.47 |
| Very high | 8.42 |

High-volume items are inherently more volatile in absolute terms, so they deserve wider safety stock or separate modelling.

**Hardest series.** The ten store–item pairs with the highest average error are dominated by **Store 2 (seven of ten)** and **Store 8 (three of ten)**, with MAE around 9.3 to 10.0. These are the first candidates for manual review.

---

## Classical models (single-series experiment)

Classical models were fitted on **one series only (Store 1, Item 1)**, tested on 2017 (365 days):

| Model | MAE | RMSE |
|---|---:|---:|
| Naive | 8.76 | 10.73 |
| Moving average (7) | 6.71 | 8.50 |
| ARIMA(1,1,1) | 6.78 | 8.60 |
| SARIMA(1,1,1)(1,1,1,7) | 6.02 | 7.59 |

These numbers are **not directly comparable** to the XGBoost results above, which are averaged over all 500 series. SARIMA is competitive on this single series, but fitting a separate model per series does not scale well to 500 series, which is the practical argument for one global ML model.

---

## Dashboard

The Streamlit app lets a user select a **store** and **item** and view:

- Historical sales
- Forecast vs actual for the 2017 validation period
- MAE and number of records evaluated
- Error distribution histogram

```bash
streamlit run app/app.py
```

_Add a screenshot here:_ `![Dashboard](assets/dashboard.png)`

---

## How to run

```bash
git clone https://github.com/cyrildude77/demand-forecasting.git
cd demand-forecasting
pip install -r requirements.txt
```

1. Download `train.csv` from Kaggle into `data/raw/`.
2. Run the notebooks in order (`1.ipynb` → `06_error_analysis.ipynb`). Each one writes its output to `data/processed/`.
3. Launch the dashboard: `streamlit run app/app.py`

The dashboard reads `data/processed/clean_sales.csv` and `data/processed/valid_with_predictions.csv`, both generated by the notebooks.

---

## Project structure

```
demand-forecasting/
├── app/
│   └── app.py                      # Streamlit dashboard
├── notebooks/
│   ├── 1.ipynb                     # data validation & cleaning
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_and_classical_models.ipynb
│   ├── 05_ml_models.ipynb
│   └── 06_error_analysis.ipynb
├── data/                           # not tracked (see .gitignore)
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

---

## Limitations

- **One-day-ahead evaluation**, not a multi-day horizon (see above).
- **Single train/validation split** with no rolling-origin cross-validation.
- **No hyperparameter tuning** and no holiday, promotion or price features (the dataset does not contain them).
- **Classical models were tested on one series only**, so there is no like-for-like comparison with the global XGBoost model.
- MAPE excludes zero-sales days, and WAPE is the more reliable percentage metric here.

## Future work

- Multi-step forecasting (recursive and direct) with a horizon-wise error report
- Rolling-origin cross-validation and hyperparameter tuning (Optuna)
- LightGBM comparison and holiday/event features
- Quantile or prediction-interval forecasts to support safety-stock decisions
- Per-series SARIMA/ETS benchmark on a sample of series for a fair classical comparison

---

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · Streamlit · Matplotlib · Seaborn

## Author

**Cyril Anand** — [GitHub](https://github.com/cyrildude77)
