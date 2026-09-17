import pandas as pd
import os

# Expected 18 columns matching your dataset
EXPECTED_COLS = [
    "Date", "Steel_Long", "Iron_Ore", "Coal_Price", "Steel_Scrap",
    "Crude_Oil", "BDI_Index", "Global_Iron_Ore", "Basic_Metals",
    "Scrap_Lag1", "Scrap_Lag2", "BDI_Lag1", "BDI_Lag2", "BDI_Lag3",
    "USD_INR", "USD_INR_Lag1", "Price_Lag_24h", "Price_Lag_7D"
]

# 1. Load CSV & auto-detect header
df = pd.read_csv("data/steel_market_big_data_23k.csv", header="infer")

# 2. If header doesn't match, force-rename columns
if df.columns[0].strip() != "Date":
    df.columns = EXPECTED_COLS

# 3. Parse dates safely with explicit format
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# 4. Remove rows where date parsing failed (catches leftover header rows)
df = df.dropna(subset=["Date"])

# 5. Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

# 6. Save cleaned version
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_steel.csv", index=False)

print(f"✅ Cleaned data saved: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")