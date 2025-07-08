import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load real Nikkei 225 data
nikkei = yf.download("^N225", start="2000-01-01", end="2025-05-31", auto_adjust=False)
nikkei.columns = nikkei.columns.droplevel(1)
print("Final columns:", nikkei.columns)
if 'Close' not in nikkei.columns:
    raise ValueError("Expected 'Close' column not found in downloaded data.")
nikkei = nikkei[['Close']]
nikkei['Return'] = nikkei['Close'].pct_change()
nikkei['month'] = nikkei.index.month
month_dummies = pd.get_dummies(nikkei['month'], prefix='month', drop_first=True)
nikkei = pd.concat([nikkei, month_dummies], axis=1)
nikkei.dropna(inplace=True)

# Step 2: Split data
train = nikkei['Close'][:'2022']
test = nikkei['Close']['2023':]
train_exog = nikkei.loc[:'2022', month_dummies.columns]
test_exog = nikkei.loc['2023':, month_dummies.columns]
actual_prices = test.copy()

# Step 3: ARIMA
arima_model = ARIMA(train, order=(1, 1, 1))
arima_result = arima_model.fit()
arima_forecast_price = arima_result.forecast(steps=len(test))

# Step 4: ARIMAX
arimax_model = ARIMA(train, order=(1, 1, 1), exog=train_exog)
arimax_result = arimax_model.fit()
arimax_forecast_price = arimax_result.forecast(steps=len(test), exog=test_exog)

# Step 5: SARIMAX
sarimax_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), exog=train_exog)
sarimax_result = sarimax_model.fit()
sarimax_forecast_price = sarimax_result.forecast(steps=len(test), exog=test_exog)

# Step 6: Metrics
arima_rmse = np.sqrt(mean_squared_error(actual_prices.to_numpy(), arima_forecast_price.to_numpy()))
arimax_rmse = np.sqrt(mean_squared_error(actual_prices.to_numpy(), arimax_forecast_price.to_numpy()))
sarimax_rmse = np.sqrt(mean_squared_error(actual_prices.to_numpy(), sarimax_forecast_price.to_numpy()))

arima_mae = mean_absolute_error(actual_prices.to_numpy(), arima_forecast_price.to_numpy())
arimax_mae = mean_absolute_error(actual_prices.to_numpy(), arimax_forecast_price.to_numpy())
sarimax_mae = mean_absolute_error(actual_prices.to_numpy(), sarimax_forecast_price.to_numpy())

print(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
print(f"ARIMAX RMSE: {arimax_rmse:.2f}, MAE: {arimax_mae:.2f}")
print(f"SARIMAX RMSE: {sarimax_rmse:.2f}, MAE: {sarimax_mae:.2f}")

# Step 7: Plot
plt.figure(figsize=(14, 6))
plt.plot(actual_prices.index, actual_prices, label='Actual Prices', color='black')
plt.plot(actual_prices.index, arima_forecast_price, label='ARIMA Forecast', linestyle='--')
plt.plot(actual_prices.index, arimax_forecast_price, label='ARIMAX Forecast', linestyle='-.')
plt.plot(actual_prices.index, sarimax_forecast_price, label='SARIMAX Forecast', linestyle=':')
plt.title('Nikkei 225 Forecast Comparison (2023–2025)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
