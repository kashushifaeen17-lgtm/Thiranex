import warnings
warnings.filterwarnings("ignore")

from data_generator import generate_data
from data_preprocessing import preprocess
from regression_model import train_regression
from arima_model import train_arima   # ✅ FIXED

from evaluation import evaluate
from visualization import plot_results, plot_forecast

import pandas as pd

df = generate_data()
df = preprocess(df)

# Regression
df, lr_forecast = train_regression(df)

# ARIMA
df, arima_forecast = train_arima(df)

# Evaluation
metrics = evaluate(df)

print("\nMODEL PERFORMANCE")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 5. Visualization
plot_results(df)
plot_forecast(df, lr_forecast, arima_forecast)

# 6. Save files
df.to_csv("processed_data.csv", index=False)

pd.DataFrame({
    "LR_Forecast": lr_forecast,
    "ARIMA_Forecast": arima_forecast
}).to_csv("future_forecast.csv", index=False)

print("\nFiles saved successfully!")