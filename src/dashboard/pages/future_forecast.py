import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import warnings
warnings.filterwarnings('ignore')

st.title("Future Demand Predictor")
st.caption("Forward-looking steel demand intelligence powered by decoupled FastAPI Backend models")

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

# Load historical dates for date widget baseline
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "forecast_results.csv"))
if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path, parse_dates=["Date"])
    last_date = pd.to_datetime(df_hist["Date"]).max()
else:
    last_date = pd.to_datetime("2026-03-01")

st.sidebar.header("Prediction Settings")
selected_date = st.sidebar.date_input("Select Target Date", (last_date + pd.Timedelta(days=7)).date())

if st.sidebar.button("Generate Forecast", type="primary"):
    payload = {"date": selected_date.strftime("%Y-%m-%d")}
    
    with st.spinner("Quering FastAPI Forecasting service..."):
        try:
            response = requests.post(f"{API_URL}/forecast", json=payload)
            if response.status_code == 200:
                res = response.json()
                pred = res["prediction"]
                ci_low = res["ci_lower"]
                ci_high = res["ci_upper"]
                model_used = res["method"]
                days_ahead = res["days_ahead"]
                last_val = res["last_val"]
                last_date_str = res["last_date"]
                last_date = pd.to_datetime(last_date_str)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Target Date", selected_date.strftime("%Y-%m-%d"))
                col2.metric("Predicted Demand", f"{pred:.2f} units")
                col3.metric("95% Confidence Interval", f"{ci_low:.2f} to {ci_high:.2f}")
                st.caption(f"Backend Service Method: {model_used}")
                st.divider()

                if days_ahead > 0:
                    st.subheader("Future Demand Projection")
                    fig = go.Figure()
                    
                    future_x = [last_date, pd.Timestamp(selected_date)]
                    future_y = [last_val, pred]
                    
                    # Projection line
                    fig.add_trace(go.Scatter(
                        x=future_x, y=future_y, 
                        name="Projected Demand", 
                        line=dict(color="#4f46e5", width=3, shape="spline")
                    ))
                    
                    # Confidence interval shading
                    fig.add_trace(go.Scatter(
                        x=future_x + future_x[::-1], 
                        y=[ci_high, ci_low][::-1], 
                        fill='toself', 
                        fillcolor='rgba(79,70,229,0.06)', 
                        line=dict(color='rgba(0,0,0,0)'), 
                        name="95% Confidence Range"
                    ))
                    
                    # Add starting point marker
                    fig.add_trace(go.Scatter(
                        x=[last_date], y=[last_val], 
                        mode='markers', 
                        marker=dict(size=8, color='#4f46e5', symbol='circle'), 
                        name="Last Known Forecast"
                    ))

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
                                text="Demand (units)",
                                font=dict(family="Plus Jakarta Sans", size=12, color="#0f172a", weight="bold")
                            ),
                            gridcolor="rgba(148, 163, 184, 0.15)",
                            linecolor="rgba(148, 163, 184, 0.2)",
                            tickfont=dict(family="Plus Jakarta Sans", size=11, color="#475569")
                        ),
                        template="plotly_white",
                        legend=dict(orientation="h", y=1.08, xanchor="center", x=0.5, font=dict(family="Plus Jakarta Sans", size=11, color="#64748b")),
                        margin=dict(l=40, r=40, t=30, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select a future date in the sidebar to view the AI projection.")
            else:
                st.error(f"Error from FastAPI backend: {response.text}")
        except Exception as e:
            st.error(f"HTTP request failed: {e}")
else:
    st.info("Select a date and click 'Generate Forecast'")