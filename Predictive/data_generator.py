import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)

    dates = pd.date_range(start="2019-01-01", periods=60, freq="ME")

    trend = np.arange(60) * 5
    seasonality = 50 * np.sin(np.arange(60) * 2 * np.pi / 12)
    noise = np.random.normal(0, 20, 60)

    sales = 200 + trend + seasonality + noise

    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales
    })

    return df