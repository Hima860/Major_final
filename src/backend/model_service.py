import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Define the EXACT same class structure as in your training script
class Net(nn.Module):
    def __init__(self, inp, hid, layers):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        # Hidden state size must match training (64)
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class HybridForecaster:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.sarima_model = None
        self.lstm_model = None
        self.scalers = None
        self.is_loaded = False
        self.SEQ = 30  # Must match your training script

    def load_models(self):
        try:
            # 1. Load Scalers (Your code saves a dictionary of scalers)
            scaler_path = os.path.join(self.model_dir, "scalers.pkl")
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
                
                # Calculate input size automatically (Total scalers - 1 for target)
                inp_size = len(self.scalers) - 1
                
                # 2. Load LSTM Model
                lstm_path = os.path.join(self.model_dir, "lstm_model.pt")
                if os.path.exists(lstm_path):
                    # Initialize the network architecture
                    self.lstm_model = Net(inp_size, 64, 2)
                    # Load the saved weights
                    self.lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                    self.lstm_model.eval() # Set to evaluation mode

            # 3. Load SARIMA Model
            sarima_path = os.path.join(self.model_dir, "sarima_model.pkl")
            if os.path.exists(sarima_path):
                self.sarima_model = joblib.load(sarima_path)

            self.is_loaded = True
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f" Model loading error: {e}")
            return False

    def forecast_future(self, historical_data, steps_ahead):
        if not self.is_loaded:
            self.load_models()

        if not self.is_loaded:
            return self._fallback_forecast(historical_data, steps_ahead)

        try:
            # --- SARIMA PREDICTION ---
            sarima_forecast = self.sarima_model.get_forecast(steps=steps_ahead)
            sarima_pred = sarima_forecast.predicted_mean.values[-1]
            sarima_ci = sarima_forecast.conf_int()
            sarima_lower = sarima_ci.iloc[-1, 0]
            sarima_upper = sarima_ci.iloc[-1, 1]

            # --- LSTM PREDICTION ---
            lstm_pred = None
            if self.lstm_model is not None and self.scalers is not None:
                lstm_pred = self._predict_lstm(historical_data, steps_ahead)

            # --- ENSEMBLE (50/50) ---
            if lstm_pred is not None:
                ensemble_pred = (sarima_pred + lstm_pred) / 2
                # Combine confidence intervals
                ci_width = sarima_upper - sarima_lower
                lstm_uncertainty = abs(lstm_pred - sarima_pred) * 0.5
                final_lower = ensemble_pred - (ci_width / 2) - lstm_uncertainty
                final_upper = ensemble_pred + (ci_width / 2) + lstm_uncertainty
            else:
                ensemble_pred = sarima_pred
                final_lower = sarima_lower
                final_upper = sarima_upper

            return ensemble_pred, final_lower, final_upper, "Hybrid SARIMA+LSTM"

        except Exception as e:
            print(f"Forecast error: {e}")
            return self._fallback_forecast(historical_data, steps_ahead)

    def _predict_lstm(self, historical_data, steps_ahead):
        """Recursive prediction using the saved LSTM model"""
        try:
            # Get the last SEQ days of data
            last_seq = historical_data.tail(self.SEQ)
            
            # Prepare features (assuming historical_data has the same columns as training)
            # If historical_data is just a Series of predictions, we need to be careful.
            # Ideally, pass the full dataframe with features. 
            # For now, simple fallback if input is just a Series
            if isinstance(last_seq, pd.Series):
                # If we only have the target series, we can't run LSTM properly without features
                # But we can try to scale just the target
                target_scaler = self.scalers.get('Steel_Long') # Adjust name if different
                if target_scaler:
                    seq_scaled = target_scaler.transform(last_seq.values.reshape(-1, 1))
                else:
                    seq_scaled = last_seq.values.reshape(-1, 1)
            else:
                # If it's a DataFrame with features
                cols = list(self.scalers.keys())
                # Remove target from feature list if present
                feature_cols = [c for c in cols if c != 'Steel_Long']
                
                # Scale features
                scaled_features = []
                for col in feature_cols:
                    if col in last_seq.columns and col in self.scalers:
                        scaled_features.append(self.scalers[col].transform(last_seq[[col]].values))
                
                if scaled_features:
                    seq_scaled = np.hstack(scaled_features)
                else:
                    seq_scaled = last_seq.values # Fallback

            # Reshape for LSTM: [1, SEQ, Features]
            # If seq_scaled is [SEQ, 1], we might need to transpose or handle dimensions
            # Based on your training code: X_tr_seq shape is [samples, SEQ, features]
            
            x_input = torch.FloatTensor(seq_scaled).unsqueeze(0) # [1, SEQ, features]
            
            with torch.no_grad():
                pred_scaled = self.lstm_model(x_input).item()
            
            # Inverse transform the result
            target_scaler = self.scalers.get('Steel_Long')
            if target_scaler:
                final_pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
            else:
                final_pred = pred_scaled
                
            return final_pred
        except Exception as e:
            print(f"LSTM Prediction Error: {e}")
            return None

    def _fallback_forecast(self, historical_data, steps_ahead):
        last_val = historical_data.iloc[-1]
        trend = historical_data.iloc[-1] - historical_data.iloc[-5] if len(historical_data) >= 5 else 0
        daily_trend = trend / 5
        pred = last_val + (daily_trend * steps_ahead)
        margin = abs(pred) * (0.01 + steps_ahead * 0.0005)
        return pred, pred - margin, pred + margin, "Fallback (Trend)"