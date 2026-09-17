import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backend.database import init_db, seed_from_csv

# DYNAMIC STYLESHEET LOADING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
style_path = os.path.join(BASE_DIR, "style.css")
if os.path.exists(style_path):
    try:
        with open(style_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        pass


st.set_page_config(page_title="Steel Procurement Intelligence", layout="wide")

# Initialize DB
if "db_initialized" not in st.session_state:
    try:
        init_db()
        seed_from_csv("data/cleaned_steel.csv", "market_data")
        seed_from_csv("outputs/forecast_results.csv", "predictions")
        st.session_state.db_initialized = True
    except Exception as e:
        st.warning(f"Database initialization note: {e}")

st.markdown("""
<div class="custom-header">
    <h1>Steel Procurement Intelligence</h1>
    <div class="header-subtitle">Hybrid AI Forecasting • Inventory Control • Procurement Analytics</div>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

page = st.sidebar.selectbox("Select Module", [
    "Future Predictor",
    "Inventory Optimization",
    "Procurement Analytics",
    "Explainability AI"
])

if page == "Future Predictor":
    exec(open(os.path.join(BASE_DIR, "pages", "future_forecast.py"), encoding="utf-8").read())
elif page == "Inventory Optimization":
    exec(open(os.path.join(BASE_DIR, "pages", "inventory.py"), encoding="utf-8").read())
elif page == "Procurement Analytics":
    exec(open(os.path.join(BASE_DIR, "pages", "procurement.py"), encoding="utf-8").read())
elif page == "Explainability AI":
    exec(open(os.path.join(BASE_DIR, "pages", "explainability.py"), encoding="utf-8").read())