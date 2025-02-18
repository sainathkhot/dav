#this is for vs code. csv and code should be in same directory


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the path to your CSV file
csv_path = "Electric_Production.csv"  # Adjust this path if needed

# Load the dataset, adjusting the column name for the date and value
data = pd.read_csv(
    csv_path,
    parse_dates=["DATE"],  # Adjust the column name to 'DATE'
    dayfirst=False,
    index_col="DATE"  # Adjust the column name to 'DATE'
)

# Sort the DataFrame by the Date index in ascending order
data.sort_index(inplace=True)

# Clean the "VALUE" column (assuming it represents the data you want to forecast)
data["VALUE"] = data["VALUE"].replace(',', '', regex=True)
data["VALUE"] = pd.to_numeric(data["VALUE"], errors='coerce')
data["VALUE"].replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["VALUE"], inplace=True)

# Plotting the original VALUE time series
plt.figure(figsize=(14, 7))
plt.plot(data.index, data["VALUE"], label='Original Series')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Perform the Augmented Dickey-Fuller test on the original series
result_original = adfuller(data["VALUE"])

print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")

if result_original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")

# Apply first-order differencing
data['VALUE_Diff'] = data['VALUE'].diff()

# Perform the Augmented Dickey-Fuller test on the differenced series
result_diff = adfuller(data["VALUE_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Interpretation: The differenced series is Stationary.")
else:
    print("Interpretation: The differenced series is Non-Stationary.")

# Plotting the differenced series
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['VALUE_Diff'], label='Differenced Series', color='orange')
plt.title('Differenced Time Series Data')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.legend()
plt.show()

# Plot ACF and PACF for the differenced series
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# ACF plot
plot_acf(data['VALUE_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(data['VALUE_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Split the data into training and testing (80-20 split)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Fit ARIMA model (adjust p, d, q as needed)
model = ARIMA(train_data["VALUE"], order=(1, 1, 1))  # Example order (1, 1, 1), you can adjust this based on ACF/PACF
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast on the test set
forecast = model_fit.forecast(steps=len(test_data))
forecast_series = pd.Series(forecast, index=test_data.index)

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data["VALUE"], label='Train', color='#203147')
plt.plot(test_data.index, test_data["VALUE"], label='Test', color='#01ef63')
plt.plot(forecast_series, label='Forecast', color='orange')
plt.title('Forecast vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Evaluate the model (calculate RMSE)
rmse = np.sqrt(mean_squared_error(test_data["VALUE"], forecast_series))
print(f"RMSE: {rmse:.4f}")
