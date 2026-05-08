from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# -----------------------------
# TRAIN ARIMA / SARIMA
# -----------------------------
def train_arima(train_series):

    # ensure clean series
    train_series = train_series.dropna()

    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),  # weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    return results


# -----------------------------
# FORECAST
# -----------------------------
def forecast_arima(model, steps):

    forecast = model.forecast(steps=steps)

    # convert to clean output (VERY IMPORTANT for API)
    return pd.Series(forecast).reset_index(drop=True)