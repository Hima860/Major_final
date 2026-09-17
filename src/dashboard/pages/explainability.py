import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import warnings
warnings.filterwarnings('ignore')

st.title("AI Prediction Explainer")
st.caption("One-sentence explanation for each forecast powered by decoupled FastAPI Backend models")

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

st.sidebar.header("Select Date")
selected_date = st.sidebar.date_input("Target Date", (last_date + pd.Timedelta(days=7)).date())

if st.sidebar.button("Explain Forecast", type="primary"):
    payload = {"date": selected_date.strftime("%Y-%m-%d")}
    
    with st.spinner("Extracting model SHAP values from FastAPI backend..."):
        try:
            response = requests.post(f"{API_URL}/explain", json=payload)
            if response.status_code == 200:
                res = response.json()
                forecast_val = res["forecast_val"]
                days_ahead = res["days_ahead"]
                explanation = res["explanation"]
                sorted_features = res["feature_impacts"]
                
                # --- UI LAYOUT ---
                st.info(f"Prediction: {forecast_val:.1f} MT/day (+{days_ahead} days)")

                # The ONE statement (big, bold, centered in a custom card)
                st.markdown(f"""
                <div class="glass-panel" style="text-align: center; font-size: 1.2rem; border-left: 5px solid #4f46e5 !important; background: #ffffff !important; color: #1e293b !important; margin: 1rem 0;">
                    {explanation}
                </div>
                """, unsafe_allow_html=True)

                # Simple horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(sorted_features.values()), 
                    y=list(sorted_features.keys()),
                    orientation='h',
                    marker=dict(
                        color=["#4f46e5", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"],
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    hovertemplate="%{y}: %{x}%<extra></extra>",
                    showlegend=False
                ))
                fig.update_layout(
                    height=280, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title=dict(
                            text="Influence (%)",
                            font=dict(family="Plus Jakarta Sans", size=12, color="#0f172a", weight="bold")
                        ),
                        gridcolor="rgba(148, 163, 184, 0.15)",
                        linecolor="rgba(148, 163, 184, 0.2)",
                        tickfont=dict(family="Plus Jakarta Sans", size=11, color="#475569")
                    ),
                    yaxis=dict(
                        tickfont=dict(family="Plus Jakarta Sans", size=11, color="#0f172a", weight="bold")
                    ),
                    template="plotly_white",
                    bargap=0.3,
                    margin=dict(l=140, r=40, t=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tiny footer
                st.caption("SARIMA (40%) + LSTM (60%) ensemble | MAPE: 7.37% | Confidence scales with horizon")
            else:
                st.error(f"Error from FastAPI backend: {response.text}")
        except Exception as e:
            st.error(f"HTTP request failed: {e}")
else:
    st.info("Select a target date and click 'Explain Forecast'")