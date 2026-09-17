import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import warnings
warnings.filterwarnings('ignore')

st.title("Inventory Optimization")
st.caption("AI-driven safety stock, reorder points and EOQ powered by decoupled FastAPI Backend models")

API_URL = "http://127.0.0.1:8000/api"

# Verify backend health before rendering
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

# Load historical dates for baseline
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "forecast_results.csv"))
if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path, parse_dates=["Date"])
    last_date = pd.to_datetime(df_hist["Date"]).max()
else:
    last_date = pd.to_datetime("2026-03-01")

# Sidebar controls
st.sidebar.header("Inventory Settings")

current_stock_input = st.sidebar.number_input(
    "Current Stock (units)", 
    min_value=0, 
    max_value=1000, 
    value=130,
    step=5,
    help="Enter your actual physical inventory count from warehouse records"
)

lead_time = st.sidebar.slider("Lead Time (days)", 3, 21, 7, 1)
service_level = st.sidebar.slider("Target Service Level", 90, 99, 95, 1) / 100
selected_date = st.sidebar.date_input("Target Date", (last_date + pd.Timedelta(days=7)).date())

# Automatic Real-Time Calculations on Widget Change
payload = {
    "current_stock": float(current_stock_input),
    "lead_time": int(lead_time),
    "service_level": float(service_level),
    "date": selected_date.strftime("%Y-%m-%d")
}

try:
    response = requests.post(f"{API_URL}/inventory", json=payload)
    if response.status_code == 200:
        metrics = response.json()
        
        # Display metrics in elegant card grid
        st.markdown('<h3 style="font-family: \'Plus Jakarta Sans\', sans-serif; font-size: 1.25rem; font-weight: 700; margin-top: 1rem; color: #1e293b;">Key Performance Metrics</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AI Forecasted Demand", f"{metrics['forecast_demand']} units/day")
        col2.metric("Safety Stock", f"{metrics['safety_stock']} units")
        col3.metric("Reorder Point", f"{metrics['reorder_point']} units")
        col4.metric("EOQ (Economic Order Qty)", f"{metrics['eoq']} units")
        
        col5, col6, col7 = st.columns(3)
        col5.metric("Your Current Stock", f"{metrics['current_stock']} units")
        col6.metric("Days of Coverage", f"{metrics['days_of_inventory']} days")
        col7.metric("Stockout Risk", f"{metrics['stockout_risk']}%", delta_color="inverse")
        
        st.divider()
        
        # Visualization: Inventory trajectory over lead time
        st.subheader("Inventory Projection During Lead Time")
        
        # Retrieve dates & trajectory simulated values from backend
        dates = pd.to_datetime(metrics["trajectory_dates"])
        inventory_levels = metrics["trajectory_levels"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=inventory_levels,
            name="Projected Inventory",
            line=dict(color="#4f46e5", width=3, shape="spline"),
            fill='tozeroy',
            fillcolor='rgba(79,70,229,0.06)'
        ))
        
        # Reorder point line
        fig.add_hline(y=metrics['reorder_point'], line_dash="dash", line_color="#f59e0b", annotation_text="Reorder Point", annotation_position="top left")
        
        # Safety stock zone
        fig.add_hrect(y0=max(0, metrics['reorder_point']-metrics['safety_stock']), y1=metrics['reorder_point'],
                      fillcolor="rgba(245,158,11,0.06)", line_width=0, annotation_text="Safety Buffer", annotation_position="top left")
        
        # Stockout warning line
        fig.add_hline(y=0, line_dash="dot", line_color="#ef4444", annotation_text="Stockout Threshold", annotation_position="bottom left")
        
        fig.update_layout(
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title=dict(
                    text="Date",
                    font=dict(family="Plus Jakarta Sans", size=12, color="#0f172a", weight="bold")
                ),
                gridcolor="rgba(148, 163, 184, 0.15)",
                linecolor="rgba(148, 163, 184, 0.2)",
                tickfont=dict(family="Plus Jakarta Sans", size=11, color="#475569")
            ),
            yaxis=dict(
                title=dict(
                    text="Inventory Level (units)",
                    font=dict(family="Plus Jakarta Sans", size=12, color="#0f172a", weight="bold")
                ),
                gridcolor="rgba(148, 163, 184, 0.15)",
                linecolor="rgba(148, 163, 184, 0.2)",
                tickfont=dict(family="Plus Jakarta Sans", size=11, color="#475569")
            ),
            template="plotly_white",
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on API metrics
        st.subheader("AI Recommendations")
        if metrics['days_of_inventory'] < lead_time:
            st.markdown(f'<div class="custom-alert alert-warning">**Low Stock Alert**: Your current inventory ({current_stock_input} units) covers only {metrics["days_of_inventory"]} days, but lead time is {lead_time} days. Consider ordering {metrics["eoq"]} units soon.</div>', unsafe_allow_html=True)
        elif metrics['stockout_risk'] > 10:
            st.markdown(f'<div class="custom-alert alert-danger">**Elevated Stockout Risk**: {metrics["stockout_risk"]}% chance of stockout during lead time. Consider increasing safety stock or expediting orders.</div>', unsafe_allow_html=True)
        elif metrics['days_of_inventory'] > lead_time * 2:
            st.markdown(f'<div class="custom-alert alert-info">**Excess Inventory**: You have {metrics["days_of_inventory"]} days of coverage. Consider reducing order size to free up working capital.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="custom-alert alert-success">**Healthy Inventory**: {metrics["days_of_inventory"]} days of coverage with {metrics["stockout_risk"]}% stockout risk. Maintain current strategy.</div>', unsafe_allow_html=True)
        
        st.caption(f"Calculations use your hybrid model's forecast + {metrics['service_level']}% service level + {lead_time}-day lead time + your entered stock level.")
    else:
        st.error(f"Error from FastAPI backend: {response.text}")
except Exception as e:
    st.error(f"HTTP request failed: {e}")