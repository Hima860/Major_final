import pandas as pd
import os
from .database import get_connection

def get_forecast_data(window_days=365):
    conn = get_connection()
    query = """
        SELECT date, actual, sarima_forecast, lstm_forecast, ensemble_forecast 
        FROM predictions 
        ORDER BY date DESC 
        LIMIT ?
    """
    try:
        df = pd.read_sql_query(query, conn, params=(window_days,))
    except Exception as e:
        print(f"[DB] Query failed: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()
    
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        'date': 'Date', 'actual': 'Actual',
        'sarima_forecast': 'SARIMA_Forecast',
        'lstm_forecast': 'LSTM_Forecast',
        'ensemble_forecast': 'Ensemble_Forecast'
    })
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")