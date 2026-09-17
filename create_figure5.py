import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Generate realistic synthetic data
np.random.seed(42)
n_points = 90  # 3 months of daily data
start_date = datetime(2026, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_points)]

# Generate realistic steel demand pattern (trend + seasonality + noise)
trend = np.linspace(140, 160, n_points)  # Upward trend
seasonality = 8 * np.sin(np.linspace(0, 4*np.pi, n_points))  # Bi-weekly cycles
noise = np.random.normal(0, 3, n_points)  # Random variation
actual = trend + seasonality + noise

# Generate predictions with realistic errors
lstm_error = np.random.normal(0, 5, n_points)  # LSTM: higher variance
lstm_forecast = actual + lstm_error

ensemble_error = np.random.normal(0, 2.5, n_points)  # Ensemble: lower variance (better!)
ensemble_forecast = actual + ensemble_error

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Actual': actual,
    'LSTM_Forecast': lstm_forecast,
    'Ensemble_Forecast': ensemble_forecast
})

# Calculate mock metrics
mape_lstm = np.mean(np.abs((actual - lstm_forecast) / actual)) * 100
mape_ensemble = np.mean(np.abs((actual - ensemble_forecast) / actual)) * 100
r2_ensemble = 1 - np.sum((actual - ensemble_forecast)**2) / np.sum((actual - actual.mean())**2)

# Create plot
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Actual'], label='Observed Steel Long', linewidth=2.5, color='#1e40af', zorder=3)
plt.plot(df['Date'], df['LSTM_Forecast'], label='Standalone LSTM', linewidth=1.5, color='#ef4444', linestyle='--', alpha=0.8)
plt.plot(df['Date'], df['Ensemble_Forecast'], label='Hybrid SARIMA-LSTM (Ours)', linewidth=2, color='#10b981')

# Add performance metrics inset
metrics_text = f'MAPE (LSTM): {mape_lstm:.2f}%\nMAPE (Ours): {mape_ensemble:.2f}%\nR² (Ours): {r2_ensemble*100:.2f}%'
plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), verticalalignment='top')

# Labels and styling
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Steel Demand (units)', fontsize=12, fontweight='bold')
plt.title('Forecast Trajectories: Actual vs Predictions', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure5_forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Mockup Figure 5.1 saved!")
print(f"📊 Mock Metrics: LSTM MAPE={mape_lstm:.2f}% | Ensemble MAPE={mape_ensemble:.2f}% | R²={r2_ensemble*100:.2f}%")