import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("outputs", exist_ok=True)


df = pd.read_csv("data/cleaned_steel.csv", parse_dates=["Date"])
df = df.set_index("Date")
df_daily = df.resample("D").mean().dropna()


split_idx = int(len(df_daily) * 0.8)
train, test = df_daily.iloc[:split_idx], df_daily.iloc[split_idx:]

FEATURES = [c for c in df_daily.columns if c != "Steel_Long"]
TARGET = "Steel_Long"


scalers = {}
for col in FEATURES + [TARGET]:
    scaler = StandardScaler()
    train[f"{col}_s"] = scaler.fit_transform(train[[col]])
    test[f"{col}_s"] = scaler.transform(test[[col]])
    scalers[col] = scaler

X_train = train[[f"{c}_s" for c in FEATURES]].values
y_train = train[f"{TARGET}_s"].values
X_test = test[[f"{c}_s" for c in FEATURES]].values
y_test = test[f"{TARGET}_s"].values

# 4. Train Model (Gradient Boosting: fast, accurate, SHAP-compatible)
print("🔹 Training model...")
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Predict & Inverse Scale
y_pred_s = model.predict(X_test)
y_pred = scalers[TARGET].inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
y_actual = scalers[TARGET].inverse_transform(y_test.reshape(-1, 1)).flatten()

# 6. Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / np.clip(y_actual, 1e-8, None))) * 100

print(f"✅ MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

# 7. Save Forecast Results (Dashboard expects these exact columns)
forecast_df = pd.DataFrame({
    "Date": test.index,
    "Actual": y_actual,
    "Ensemble_Forecast": y_pred
})
forecast_df["RMSE"] = rmse
forecast_df["MAPE"] = mape
forecast_df.to_csv("outputs/forecast_results.csv", index=False)
print("📁 Saved: outputs/forecast_results.csv")

# 8. Compute & Save SHAP Values
print(" Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
np.save("outputs/shap_values.npy", shap_values)
np.save("outputs/test_X.npy", X_test)
print("📁 Saved: outputs/shap_values.npy, outputs/test_X.npy")

# 9. Save Model & Scalers
joblib.dump(model, "outputs/aggregator.pkl")
joblib.dump(scalers, "outputs/scalers.pkl")
print("📁 Saved: outputs/aggregator.pkl, outputs/scalers.pkl")
print(" Model training complete! Next: streamlit run dashboard/app.py")