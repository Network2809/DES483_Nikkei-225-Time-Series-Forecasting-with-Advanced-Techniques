# Nikkei 225 Time Series Forecasting using ARIMA and SARIMA

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load and Prepare Data
nikkei = yf.download("^N225", start="2000-01-01", end="2025-06-01")
nikkei = nikkei[['Close']].dropna()
nikkei['Return'] = nikkei['Close'].pct_change()
nikkei = nikkei.dropna()

# 2. ADF Test and Differencing
def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return result[1]

def difference_until_stationary(series):
    d = 0
    while adf_test(series) > 0.05:
        series = series.diff().dropna()
        d += 1
    return d

# Determine d for ARIMA
d_price = difference_until_stationary(nikkei['Close'])
d_return = difference_until_stationary(nikkei['Return'])

# 3. Train/Test Split
split_date = "2023-01-01"
train_price = nikkei.loc[:split_date, "Close"]
test_price = nikkei.loc[split_date:, "Close"]
train_return = nikkei.loc[:split_date, "Return"]
test_return = nikkei.loc[split_date:, "Return"]

# 4. Fit Models
arima_price_model = ARIMA(train_price, order=(5, d_price, 1)).fit()
arima_price_forecast = arima_price_model.forecast(steps=len(test_price))

arima_return_model = ARIMA(train_return, order=(5, d_return, 1)).fit()
arima_return_forecast = arima_return_model.forecast(steps=len(test_return))

sarima_model = SARIMAX(train_price, order=(5, d_price, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_forecast = sarima_model.forecast(steps=len(test_price))

# 5. Evaluation
def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return round(rmse, 2), round(mae, 2), round(mape, 2)

results = pd.DataFrame([
    ["ARIMA (Price)", *evaluate(test_price, arima_price_forecast)],
    ["ARIMA (Return)", *evaluate(test_return, arima_return_forecast)],
    ["SARIMA (Price)", *evaluate(test_price, sarima_forecast)],
], columns=["Model", "RMSE", "MAE", "MAPE (%)"])

print("\nForecast Evaluation Metrics:")
print(results)

# 6. Plotting
plt.figure(figsize=(14, 6))
plt.plot(test_price.index, test_price, label="Actual Price", color='black')
plt.plot(test_price.index, arima_price_forecast, label="ARIMA Forecast", color='orange')
plt.plot(test_price.index, sarima_forecast, label="SARIMA Forecast", color='green')
plt.title("Nikkei 225 Forecast Comparison (2023–2025)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()