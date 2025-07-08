
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Simulate data
np.random.seed(42)
dates = pd.date_range(start="2000-01-01", end="2025-05-31", freq='B')
returns = np.random.normal(0, 0.01, size=len(dates))
close_prices = 100 * (1 + returns).cumprod()
df = pd.DataFrame({'Close': close_prices}, index=dates)
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

# Step 2: ACF/PACF
plot_acf(df['Return'][:'2022'])
plt.title('ACF of Returns')
plt.show()

plot_pacf(df['Return'][:'2022'])
plt.title('PACF of Returns')
plt.show()

# Step 3: Train/Test Split
train = df[:'2022']
test = df['2023':]

# Step 4: ARIMA Model
arima_model = ARIMA(train['Return'], order=(1, 0, 1))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test))
arima_rmse = np.sqrt(mean_squared_error(test['Return'], arima_forecast))
arima_mae = mean_absolute_error(test['Return'], arima_forecast)

plt.plot(test.index, test['Return'], label='Actual')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', linestyle='--')
plt.title(f'ARIMA: RMSE={arima_rmse:.5f}, MAE={arima_mae:.5f}')
plt.legend()
plt.show()

# Step 5: SARIMAX with Monthly Dummies
df['month'] = df.index.month
month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
df = pd.concat([df, month_dummies], axis=1)
train_exog = df.loc[:'2022', month_dummies.columns]
test_exog = df.loc['2023':, month_dummies.columns]

sarimax_model = SARIMAX(train['Return'], exog=train_exog, order=(1, 0, 1))
sarimax_result = sarimax_model.fit()
sarimax_forecast = sarimax_result.forecast(steps=len(test), exog=test_exog)
sarimax_rmse = np.sqrt(mean_squared_error(test['Return'], sarimax_forecast))
sarimax_mae = mean_absolute_error(test['Return'], sarimax_forecast)

plt.plot(test.index, test['Return'], label='Actual')
plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast', linestyle='--')
plt.title(f'SARIMAX: RMSE={sarimax_rmse:.5f}, MAE={sarimax_mae:.5f}')
plt.legend()
plt.show()
