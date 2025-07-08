import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load data
nikkei = yf.download("^N225", start="2000-01-01", end="2025-06-01")
nikkei = nikkei[['Close']].dropna()
nikkei['Return'] = nikkei['Close'].pct_change().dropna()

# 2. ADF test for stationarity
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    return result[1]

def make_stationary(series):
    d = 0
    pval = adf_test(series)
    while pval > 0.05:
        series = series.diff().dropna()
        d += 1
        pval = adf_test(series)
    return d, series

d_price, stationary_price = make_stationary(nikkei['Close'])
d_return, stationary_return = make_stationary(nikkei['Return'])

# 3. ACF/PACF
sm.graphics.tsa.plot_acf(stationary_price, lags=30)
sm.graphics.tsa.plot_pacf(stationary_price, lags=30)
plt.show()

# 4. Split train/test
train_price = nikkei.loc[:"2022-12-31", "Close"]
test_price = nikkei.loc["2023-01-01":, "Close"]
train_return = nikkei.loc[:"2022-12-31", "Return"]
test_return = nikkei.loc["2023-01-01":, "Return"]

# 5. ARIMA Models
model_price = ARIMA(train_price, order=(5, d_price, 1)).fit()
forecast_price = model_price.forecast(steps=len(test_price))

model_return = ARIMA(train_return, order=(5, d_return, 1)).fit()
forecast_return = model_return.forecast(steps=len(test_return))

# 6. SARIMAX (seasonal order as example)
sarima_price = SARIMAX(train_price, order=(5, d_price, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_forecast_price = sarima_price.forecast(steps=len(test_price))

# 7. Evaluation
def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape

price_rmse, price_mae, price_mape = evaluate(test_price, forecast_price)
ret_rmse, ret_mae, ret_mape = evaluate(test_return, forecast_return)
sarima_rmse, sarima_mae, sarima_mape = evaluate(test_price, sarima_forecast_price)

# 8. Summary Table
summary = pd.DataFrame({
    "Model": ["ARIMA (Price)", "ARIMA (Return)", "SARIMA (Price)"],
    "RMSE": [price_rmse, ret_rmse, sarima_rmse],
    "MAE": [price_mae, ret_mae, sarima_mae],
    "MAPE": [price_mape, ret_mape, sarima_mape]
})
print(summary)

# 9. Plot
plt.figure(figsize=(12, 6))
plt.plot(test_price.index, test_price, label="Actual")
plt.plot(test_price.index, forecast_price, label="ARIMA Forecast")
plt.plot(test_price.index, sarima_forecast_price, label="SARIMA Forecast")
plt.legend()
plt.title("Nikkei 225 Forecasts")
plt.show()