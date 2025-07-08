
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Simulate Nikkei 225-like data
np.random.seed(42)
dates = pd.date_range(start="2000-01-01", end="2025-05-31", freq='B')
returns = np.random.normal(0, 0.01, size=len(dates))
close_prices = 100 * (1 + returns).cumprod()
df = pd.DataFrame({'Close': close_prices}, index=dates)

# 2. Create Return and Log(Close)
df['Return'] = df['Close'].pct_change()
df['LogClose'] = np.log(df['Close'])
df['month'] = df.index.month
month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
df = pd.concat([df, month_dummies], axis=1)
df.dropna(inplace=True)

# ACF/PACF (can be uncommented to view)
# plot_acf(df['Return'][:'2022'])
# plt.title('ACF of Returns')
# plt.show()

# plot_pacf(df['Return'][:'2022'])
# plt.title('PACF of Returns')
# plt.show()

# 3. ARIMA on log(Close)
train_log = df['LogClose'][:'2022']
test_log = df['LogClose']['2023':]
arima_price_model = ARIMA(train_log, order=(1, 1, 1))
arima_price_result = arima_price_model.fit()
forecast_log_prices = arima_price_result.forecast(steps=len(test_log))
forecast_prices = np.exp(forecast_log_prices)
actual_prices = df['Close']['2023':]
arima_rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices))
arima_mae = mean_absolute_error(actual_prices, forecast_prices)

plt.figure(figsize=(12, 5))
plt.plot(actual_prices.index, actual_prices, label='Actual Prices')
plt.plot(actual_prices.index, forecast_prices, label='ARIMA Forecast', linestyle='--')
plt.title(f'ARIMA Forecast | RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}')
plt.legend()
plt.show()

# 4. ARIMAX
train_exog = df.loc[:'2022', month_dummies.columns]
test_exog = df.loc['2023':, month_dummies.columns]
arimax_model = ARIMA(train_log, order=(1, 1, 1), exog=train_exog)
arimax_result = arimax_model.fit()
arimax_forecast_log = arimax_result.forecast(steps=len(test_log), exog=test_exog)
arimax_forecast_price = np.exp(arimax_forecast_log)

# 5. SARIMAX
sarimax_model = SARIMAX(train_log, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), exog=train_exog)
sarimax_result = sarimax_model.fit()
sarimax_forecast_log = sarimax_result.forecast(steps=len(test_log), exog=test_exog)
sarimax_forecast_price = np.exp(sarimax_forecast_log)

# 6. Metrics and Plot
arimax_rmse = np.sqrt(mean_squared_error(actual_prices, arimax_forecast_price))
arimax_mae = mean_absolute_error(actual_prices, arimax_forecast_price)
sarimax_rmse = np.sqrt(mean_squared_error(actual_prices, sarimax_forecast_price))
sarimax_mae = mean_absolute_error(actual_prices, sarimax_forecast_price)

plt.figure(figsize=(12, 5))
plt.plot(actual_prices.index, actual_prices, label='Actual Price')
plt.plot(actual_prices.index, arimax_forecast_price, label='ARIMAX Forecast', linestyle='--')
plt.plot(actual_prices.index, sarimax_forecast_price, label='SARIMAX Forecast', linestyle=':')
plt.title(f'ARIMAX vs SARIMAX Forecast | ARIMAX RMSE: {arimax_rmse:.2f}, SARIMAX RMSE: {sarimax_rmse:.2f}')
plt.legend()
plt.show()
