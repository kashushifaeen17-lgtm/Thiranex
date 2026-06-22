from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(df):

    mae_lr = mean_absolute_error(df["Sales"], df["LR_Pred"])
    rmse_lr = np.sqrt(mean_squared_error(df["Sales"], df["LR_Pred"]))

    arima_pred = df["ARIMA_Fitted"].dropna()
    actual = df.loc[arima_pred.index, "Sales"]

    mae_arima = mean_absolute_error(actual, arima_pred)
    rmse_arima = np.sqrt(mean_squared_error(actual, arima_pred))

    return {
        "LR_MAE": mae_lr,
        "LR_RMSE": rmse_lr,
        "ARIMA_MAE": mae_arima,
        "ARIMA_RMSE": rmse_arima
    }