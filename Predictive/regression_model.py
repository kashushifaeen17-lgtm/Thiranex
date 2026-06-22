def train_regression(df):
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    df = df.copy()

    X = df[["TimeIndex"]]
    y = df["Sales"]

    lr = LinearRegression()
    lr.fit(X, y)

    df["LR_Pred"] = lr.predict(X)

    future = pd.DataFrame({
        "TimeIndex": range(df["TimeIndex"].max() + 1,
                          df["TimeIndex"].max() + 13)
    })

    lr_forecast = lr.predict(future)

    return df, lr_forecast