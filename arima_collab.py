# Step 1: Upload CSV file to Google Colab
# fisrt run this 2 lines and select file or upload file from local device then run the rest of the code
from google.colab import files
uploaded = files.upload()

# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 3: Load the uploaded dataset
csv_path = 'Electric_Production.csv'  # Use the name of the uploaded file
data = pd.read_csv(
    csv_path,
    parse_dates=["DATE"],  # Adjust column name if necessary
    dayfirst=False,
    index_col="DATE"  # Ensure the date column is set as the index
)

# Step 4: Sort the DataFrame by the Date index
data.sort_index(inplace=True)

# Step 5: Clean the "VALUE" column
data["VALUE"] = data["VALUE"].replace(',', '', regex=True)
data["VALUE"] = pd.to_numeric(data["VALUE"], errors='coerce')
data["VALUE"].replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["VALUE"], inplace=True)

# Step 6: Plot the original series
plt.figure(figsize=(14, 7))
plt.plot(data.index, data["VALUE"], label='Original Series')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 7: Perform Augmented Dickey-Fuller test
result_original = adfuller(data["VALUE"])
print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")
if result_original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")

# Step 8: Apply first-order differencing
data['VALUE_Diff'] = data['VALUE'].diff()

# Step 9: Perform Augmented Dickey-Fuller test on differenced series
result_diff = adfuller(data["VALUE_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Interpretation: The differenced series is Stationary.")
else:
    print("Interpretation: The differenced series is Non-Stationary.")

# Step 10: Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(data['VALUE_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(data['VALUE_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Step 11: Train-Test Split (80-20)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Step 12: Fit ARIMA model (adjust p, d, q as needed)
model = ARIMA(train_data["VALUE"], order=(1, 1, 1))  # Example order (1, 1, 1)
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Step 13: Forecast on the test set
forecast = model_fit.forecast(steps=len(test_data))
forecast_series = pd.Series(forecast, index=test_data.index)

# Step 14: Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_data["VALUE"], label='Train', color='#203147')
plt.plot(test_data.index, test_data["VALUE"], label='Test', color='#01ef63')
plt.plot(forecast_series, label='Forecast', color='orange')
plt.title('Forecast vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 15: Evaluate the model (RMSE)
rmse = np.sqrt(mean_squared_error(test_data["VALUE"], forecast_series))
print(f"RMSE: {rmse:.4f}")
