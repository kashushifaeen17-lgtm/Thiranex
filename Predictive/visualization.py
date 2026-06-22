import matplotlib.pyplot as plt
import pandas as pd

def plot_results(df):

    plt.figure(figsize=(10,5))
    plt.plot(df["Sales"], label="Actual")
    plt.plot(df["LR_Pred"], label="LR Prediction")
    plt.plot(df["ARIMA_Fitted"], label="ARIMA Fitted")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    plt.show()


def plot_forecast(df, lr_forecast, arima_forecast):

    future_dates = pd.date_range(df.index[-1], periods=13, freq="ME")[1:]

    plt.figure(figsize=(10,5))
    plt.plot(df["Sales"], label="Historical")
    plt.plot(future_dates, lr_forecast, marker="o", label="LR Forecast")
    plt.plot(future_dates, arima_forecast, marker="o", label="ARIMA Forecast")
    plt.legend()
    plt.title("Future Forecast (12 Months)")
    plt.show()