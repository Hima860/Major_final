import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Demand Forecasting")
st.caption("AI-powered steel demand predictions with real-time accuracy metrics")

# Load data with fallback
def load_data():
    try:
        from src.backend.services import get_forecast_data
        df = get_forecast_data(window_days=365)
        if not df.empty:
            return df
    except:
        pass
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "forecast_results.csv"))
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["Date"])
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ No forecast data available. Please run the model first.")
    st.stop()

# Standardize columns
df = df.rename(columns={
    'date': 'Date', 'actual': 'Actual', 'ensemble_forecast': 'Predicted'
})
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

# Calculate metrics
if "Actual" in df.columns and "Predicted" in df.columns:
    rmse = ((df["Actual"] - df["Predicted"]) ** 2).mean() ** 0.5
    mape = ((df["Actual"] - df["Predicted"]).abs() / df["Actual"].replace(0, 1)).mean() * 100
    accuracy = max(0, 100 - mape)
else:
    rmse, mape, accuracy = 0, 0, 0

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediction Accuracy", f"{accuracy:.1f}%")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("Data Points", f"{len(df)}")
col4.metric("Avg Demand", f"{df['Actual'].mean():.1f}" if "Actual" in df.columns else "N/A")

st.divider()

# Chart
st.subheader("Forecast vs Actual")
if "Actual" in df.columns and "Predicted" in df.columns and "Date" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"], name="Actual", line=dict(color="#1e40af", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted"], name="Predicted", line=dict(color="#059669", width=2, dash="dash")))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Steel Demand Index", hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Table
st.subheader("Recent Predictions")
if not df.empty:
    display_df = df.tail(10).copy()
    if "Actual" in display_df.columns and "Predicted" in display_df.columns:
        display_df["Error %"] = ((display_df["Actual"] - display_df["Predicted"]).abs() / display_df["Actual"].replace(0, 1) * 100).round(1)
        st.dataframe(display_df[["Date", "Actual", "Predicted", "Error %"]], use_container_width=True, hide_index=True)

st.divider()
st.caption("Model: Hybrid SARIMA+LSTM | Data: SQLite/CSV | Updates daily")