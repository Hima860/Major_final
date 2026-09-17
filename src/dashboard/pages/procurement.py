import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import warnings
warnings.filterwarnings('ignore')

st.title("Procurement Optimization")
st.caption("AI-driven demand forecasting with dynamic pricing and cost optimization powered by decoupled FastAPI Backend models")

API_URL = "http://127.0.0.1:8000/api"

# Verify backend health
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    backend_active = health_resp.status_code == 200
except Exception:
    backend_active = False

if not backend_active:
    st.markdown("""
    <div class="custom-alert alert-danger">
        <strong>Backend Offline Alert</strong>: The FastAPI server is currently offline or unreachable at port 8000. 
        Please start the backend server using uvicorn to run predictions: <br/>
        <code>uvicorn src.backend.main:app --reload</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load historical dates baseline
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "forecast_results.csv"))
if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path, parse_dates=["Date"])
    last_date = pd.to_datetime(df_hist["Date"]).max()
else:
    last_date = pd.to_datetime("2026-03-01")

# Sidebar Inputs
st.sidebar.header("Procurement Inputs")
current_stock = st.sidebar.number_input("Current Stock (MT)", min_value=0, max_value=2000, value=130, step=10)
base_price = st.sidebar.number_input("Base Steel Price (₹/MT)", min_value=30000, max_value=60000, value=45000, step=500)
days_horizon = st.sidebar.slider("Planning Horizon (days)", 7, 30, 14)
selected_date = st.sidebar.date_input("Target Date", (last_date + pd.Timedelta(days=7)).date())

# Automatic Real-Time Calculations on Widget Change
payload = {
    "current_stock": float(current_stock),
    "base_price": float(base_price),
    "days_horizon": int(days_horizon),
    "date": selected_date.strftime("%Y-%m-%d")
}

try:
    response = requests.post(f"{API_URL}/procurement", json=payload)
    if response.status_code == 200:
        result = response.json()
        
        if result["status"] == "NO_ORDER":
            st.subheader(f"Procurement Plan for {selected_date.strftime('%Y-%m-%d')}")
            st.markdown(f'<div class="custom-alert alert-success"><strong>Inventory Optimization Alert</strong>: {result["message"]}</div>', unsafe_allow_html=True)
        else:
            st.subheader(f"Procurement Plan for {selected_date.strftime('%Y-%m-%d')}")

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Net Order Required", f"{result['net_order']} MT")
            col2.metric("Adjusted Price", f"₹{result['dynamic_price']:,}/MT")
            col3.metric("Bulk Discount", result['discount'])
            col4.metric("Urgency Surcharge", result['urgency_fee'])

            st.markdown(f"""
            <div class="highlight-metric" style="margin: 1.5rem 0;">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: rgba(255, 255, 255, 0.85);">Total Procurement Cost</div>
                <div style="font-size: 2.25rem; font-weight: 800; margin-top: 6px; color: #ffffff;">₹{result['total_cost']:,}</div>
                <div style="font-size: 0.9rem; margin-top: 8px; color: rgba(255, 255, 255, 0.85);">Effective Rate: <strong>₹{result['effective_rate']:,}/MT</strong> (includes market volatility, tiered discounts, and urgency surcharges)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="custom-alert alert-success">Calculation successful using AI demand forecasts, dynamic market price shifts, and tiered bulk procurement rules.</div>', unsafe_allow_html=True)
    else:
        st.error(f"Error from FastAPI backend: {response.text}")
except Exception as e:
    st.error(f"HTTP request failed: {e}")