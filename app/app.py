import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
)

# -----------------------------
# Paths
# -----------------------------
# Resolve the project root from this file instead of using a machine-specific
# absolute Windows path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DATA_PATH = DATA_DIR / "clean_sales.csv"
PRED_DATA_PATH = DATA_DIR / "valid_with_predictions.csv"


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical sales file not found: {CLEAN_DATA_PATH}"
        )

    if not PRED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_DATA_PATH}"
        )

    sales_df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    pred_df = pd.read_csv(PRED_DATA_PATH, parse_dates=["date"])

    required_sales_columns = {"date", "store", "item", "sales"}
    required_pred_columns = {
        "date",
        "store",
        "item",
        "sales",
        "predicted_sales",
    }

    missing_sales = required_sales_columns - set(sales_df.columns)
    missing_pred = required_pred_columns - set(pred_df.columns)

    if missing_sales:
        raise ValueError(
            f"Missing columns in clean_sales.csv: {sorted(missing_sales)}"
        )

    if missing_pred:
        raise ValueError(
            "Missing columns in valid_with_predictions.csv: "
            f"{sorted(missing_pred)}"
        )

    return sales_df, pred_df


try:
    sales_df, pred_df = load_data()
except Exception as exc:
    st.error("The dashboard could not load its data.")
    st.exception(exc)
    st.stop()


# -----------------------------
# App UI
# -----------------------------
st.title("📦 Demand Forecasting Dashboard")
st.markdown(
    """
    This dashboard shows **historical demand**, **model predictions**,
    and **forecast errors** for store–item combinations.
    """
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Filters")

stores = sorted(sales_df["store"].dropna().unique())
items = sorted(sales_df["item"].dropna().unique())

store = st.sidebar.selectbox("Select Store", stores)
item = st.sidebar.selectbox("Select Item", items)

# -----------------------------
# Filter Data
# -----------------------------
hist_data = sales_df[
    (sales_df["store"] == store) &
    (sales_df["item"] == item)
].sort_values("date")

pred_data = pred_df[
    (pred_df["store"] == store) &
    (pred_df["item"] == item)
].sort_values("date")

if pred_data.empty:
    st.warning(
        f"No validation predictions are available for Store {store}, Item {item}."
    )
    st.stop()

# -----------------------------
# Historical Sales Plot
# -----------------------------
st.subheader(f"📈 Historical Sales — Store {store}, Item {item}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hist_data["date"], hist_data["sales"], label="Actual Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Forecast vs Actual Plot
# -----------------------------
st.subheader("🔮 Forecast vs Actual (Validation Period)")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    pred_data["date"],
    pred_data["sales"],
    label="Actual",
    linewidth=2,
)
ax.plot(
    pred_data["date"],
    pred_data["predicted_sales"],
    label="Predicted",
    linestyle="--",
)
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Error Metrics
# -----------------------------
st.subheader("📊 Error Metrics")

errors = pred_data["sales"] - pred_data["predicted_sales"]
mae = np.mean(np.abs(errors))

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
col2.metric("Records Evaluated", f"{len(pred_data):,}")

# -----------------------------
# Error Distribution
# -----------------------------
st.subheader("⚠️ Error Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors, bins=40)
ax.set_xlabel("Error (Actual - Predicted)")
ax.set_ylabel("Frequency")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# Business Insight Section
# -----------------------------
st.subheader("💡 Business Insights")

st.markdown(
    """
    - Forecast accuracy varies by **item volatility**.
    - High-demand items tend to have **larger absolute errors**.
    - Slight over-forecasting can reduce the risk of stockouts, but may
      increase inventory holding costs.
    - Store–item pairs with persistent error should be **manually reviewed**.
    """
)

st.success("✅ Dashboard loaded successfully")
