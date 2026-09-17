import sqlite3
import os
import pandas as pd

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "steel_ai.db"))

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            steel_long REAL,
            iron_ore REAL,
            coal REAL,
            bdi REAL,
            usd_inr REAL
        );
        
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            actual REAL,
            sarima_forecast REAL,
            lstm_forecast REAL,
            ensemble_forecast REAL,
            model_version TEXT DEFAULT 'v1.0',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS procurement_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_date TEXT,
            quantity_mt REAL,
            price_per_mt REAL,
            total_cost REAL,
            decision_type TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def seed_from_csv(csv_path, table_name):
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df["Date"] = df["Date"].astype(str)
        conn = get_connection()
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        print(f"Seeding error: {e}")