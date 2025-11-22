# demand-forecasting
# Inventory Demand Forecasting using Time-Series Models

This project is a complete end-to-end **Weekly Inventory Demand Forecasting System** built using:

* **Python**
* **Pandas**
* **Prophet** (Facebook)
* **ARIMA** (Statsmodels)
* **Matplotlib**

It is designed to simulate how real-world retail/warehouse teams forecast inventory, detect demand spikes, and plan safety stock levels.

---

## 🚀 Project Overview

This system predicts weekly demand using **Prophet** and compares its performance with a traditional **ARIMA** model. It also includes:

* Time-series feature engineering (lag, rolling averages)
* Outlier handling using winsorization
* Demand spike alerts
* Trend and seasonality visualizations
* Automated rolling forecast pipeline
* Safety-stock recommendation engine
* Side-by-side Prophet vs ARIMA comparison

This project is structured to be both **interview-friendly** and **industry realistic**, making it ideal for showcasing data science skills.

---

## 📂 Files Generated

| File                              | Description                                             |
| --------------------------------- | ------------------------------------------------------- |
| `weekly_inventory_forecast.csv`   | Final Prophet forecast with safety-stock recommendation |
| `forecast_comparison.csv`         | Comparison of raw vs cleaned model predictions          |
| `rolling_forecast.csv`            | Automated rolling 4-week forecast output                |
| `prophet_vs_arima_comparison.csv` | Combined Prophet + ARIMA forecast table                 |
| `forecast_plot.png`               | Prophet forecast visualization                          |
| `forecast_components.png`         | Trend + seasonality visualization                       |
| `prophet_vs_arima_plot.png`       | Prophet vs ARIMA comparison graph                       |

---

## 🧠 Key Features

### 1️⃣ Synthetic Realistic Demand Data

* Weekly demand values generated using Poisson distribution
* Perfect for demonstrating forecasting models without exposing real company data

### 2️⃣ Feature Engineering

* **Lag feature (lag_1)** → Helps capture last week's effect
* **Rolling average (rolling_4)** → Smooths sudden fluctuations

### 3️⃣ Outlier Handling

* Uses **winsorization** to cap extreme values at 5th and 95th percentiles
* Stabilizes learning and reduces MAPE

### 4️⃣ Forecasting with Prophet

* Weekly seasonality enabled
* 12-week future predictions
* Trend + seasonality components visualized

### 5️⃣ ARIMA Comparison

* ARIMA(2,1,2) model trained on the dataset
* Future 12-step forecast generated
* Combined visualization comparing both models

### 6️⃣ Demand Spike Alerts

Alerts triggered when:

```
predicted_demand > mean + 2 * std
```

Useful for preventing stockouts or managing sudden retail surges.

### 7️⃣ Automated Rolling Forecast Pipeline

* Designed to simulate real-world deployment
* Generates fresh 4-week forecasts anytime new data arrives

### 8️⃣ Safety-Stock Recommendation Engine

Formula used:

```
safety_stock = mean(demand) + 1.5 * std(demand)
```

This ensures inventories remain stable even during uncertain periods.

---

## 🏗️ Project Architecture

```
Data Generation → Preprocessing → Feature Engineering → Outlier Handling →
Prophet Training → Forecasting → ARIMA Training → Forecasting →
Forecast Comparison → Spike Detection → Visualizations → Exported Outputs
```

---

## 📊 Visualizations Included

* Prophet forecast curve
* Trend + Seasonal decomposition
* Prophet vs ARIMA comparison curve

These plots make your project highly presentable for technical interviews.

---

## 📌 How to Run the Project

### 1. Install dependencies

```
pip install pandas numpy prophet statsmodels matplotlib
```

_
(Prophet may require pystan setup depending on OS.)_

### 2. Run the Python script

```
python inventory_forecasting.py
```

### 3. Check the output files

you will get CSV files & PNG plots in the same directory.

---
