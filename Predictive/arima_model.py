from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def train_arima(df):

    df = df.copy()

    model = ARIMA(df["Sales"], order=(1, 1, 1))
    result = model.fit()

    df["ARIMA_Fitted"] = result.fittedvalues

    forecast = result.forecast(steps=12)

    return df, forecast