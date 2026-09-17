import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

print("="*60)
print("STEEL FORECASTING: SARIMA + PYTORCH LSTM")
print("="*60)

# 1. LOAD DATA
print("\n[1/6] Loading data...")
df = pd.read_csv("data/cleaned_steel.csv", parse_dates=["Date"])
df = df.set_index("Date")
df_daily = df.resample("D").mean().dropna()

TARGET = "Steel_Long"
FEATURES = [c for c in df_daily.columns if c != TARGET]

split_idx = int(len(df_daily) * 0.8)
train_df = df_daily.iloc[:split_idx].copy()
test_df = df_daily.iloc[split_idx:].copy()
print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")

# 2. SCALE
print("\n[2/6] Scaling...")
scalers = {}
for col in FEATURES + [TARGET]:
    scaler = StandardScaler()
    train_df[f"{col}_s"] = scaler.fit_transform(train_df[[col]])
    test_df[f"{col}_s"] = scaler.transform(test_df[[col]])
    scalers[col] = scaler

# 3. SARIMA
print("\n[3/6] Training SARIMA...")
sarima = sm.tsa.statespace.SARIMAX(
    train_df[TARGET], order=(2,1,2), seasonal_order=(1,1,1,7),
    enforce_stationarity=False, enforce_invertibility=False
)
sarima_res = sarima.fit(disp=False)
sarima_pred = sarima_res.forecast(steps=len(test_df)).values

# 4. LSTM SEQUENCES
print("\n[4/6] Building sequences...")
SEQ = 30
def make_seq(feat, tgt, seq_len):
    X, y = [], []
    for i in range(len(feat)-seq_len):
        X.append(feat[i:i+seq_len])
        y.append(tgt[i+seq_len])
    return np.array(X), np.array(y)

X_tr_f = train_df[[f"{c}_s" for c in FEATURES]].values
y_tr_t = train_df[f"{TARGET}_s"].values
X_tr_seq, y_tr_seq = make_seq(X_tr_f, y_tr_t, SEQ)

X_te_f = test_df[[f"{c}_s" for c in FEATURES]].values
y_te_t = test_df[f"{TARGET}_s"].values
X_comb = np.vstack([X_tr_f[-SEQ:], X_te_f])
y_comb = np.concatenate([y_tr_t[-SEQ:], y_te_t])
X_te_seq, y_te_seq = make_seq(X_comb, y_comb, SEQ)

X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_te_seq, dtype=torch.float32)

# 5. PYTORCH LSTM
print("\n[5/6] Training PyTorch LSTM...")
class Net(nn.Module):
    def __init__(self, inp, hid, layers):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(len(FEATURES), 64, 2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

for ep in range(50):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        l = loss_fn(model(xb), yb)
        l.backward()
        opt.step()
    if (ep+1)%10==0: print(f"  Epoch {ep+1}/50 | Loss: {l.item():.6f}")

# 6. PREDICT & SAVE
print("\n[6/6] Predicting & saving...")
model.eval()
with torch.no_grad():
    lstm_pred_s = model(X_te_t.to(device)).cpu().numpy().flatten()
lstm_pred = scalers[TARGET].inverse_transform(lstm_pred_s.reshape(-1,1)).flatten()

# Align lengths
n = len(lstm_pred)
sarima_a = sarima_pred[-n:]
actual_a = test_df[TARGET].values[-n:]
dates_a = test_df.index[-n:]

ens = 0.4*sarima_a + 0.6*lstm_pred
mae = mean_absolute_error(actual_a, ens)
rmse = np.sqrt(mean_squared_error(actual_a, ens))
mape = np.mean(np.abs((actual_a-ens)/np.clip(actual_a,1e-8,None)))*100
print(f"✅ MAE:{mae:.2f} | RMSE:{rmse:.2f} | MAPE:{mape:.2f}%")

pd.DataFrame({
    "Date": dates_a, "Actual": actual_a,
    "SARIMA_Forecast": sarima_a, "LSTM_Forecast": lstm_pred,
    "Ensemble_Forecast": ens
}).to_csv("outputs/forecast_results.csv", index=False)

torch.save(model.state_dict(), "models/lstm_model.pt")
joblib.dump(sarima_res, "models/sarima_model.pkl")
joblib.dump(scalers, "models/scalers.pkl")
print("📁 Saved to outputs/")
