import pandas as pd
import numpy as np
import joblib

from src.models.arima import forecast_arima
from src.models.prophet_model import forecast_prophet
from src.models.lstm import forecast_lstm
from src.feature_engineering import create_features


def forecast_state(state_df, model_info):

    model_name = model_info["model"]
    model_path = model_info["path"]

    # load trained model
    model = joblib.load(model_path)

    state_df = state_df.sort_values("date")

    horizon = 56

    last_date = state_df["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    # =========================================
    # ARIMA
    # =========================================
    if model_name == "arima":

        preds = forecast_arima(model, horizon)

    # =========================================
    # PROPHET
    # =========================================
    elif model_name == "prophet":

        preds = forecast_prophet(model, horizon).values

    # =========================================
    # LSTM
    # =========================================
    elif model_name == "lstm":

        preds = forecast_lstm(model, state_df, steps=horizon)

    # =========================================
    # XGBOOST
    # =========================================
    elif model_name == "xgboost":

        future_df = state_df.copy()

        predictions = []

        for future_date in future_dates:

            temp = create_features(future_df)

            latest = temp.iloc[-1:]

            X = latest.drop(
                columns=['total', 'date', 'state', 'category'],
                errors='ignore'
            )

            pred = model.predict(X)[0]

            predictions.append(pred)

            new_row = {
                "date": future_date,
                "state": state_df["state"].iloc[0],
                "total": pred
            }

            future_df = pd.concat(
                [future_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

        preds = predictions

    else:
        raise ValueError("Unsupported model")

    # =========================================
    # DAILY FORECAST
    # =========================================
    daily_forecast = pd.DataFrame({
        "date": future_dates,
        "forecast": preds
    })

    # =========================================
    # WEEKLY FORECAST
    # =========================================
    weekly_forecast = (
        daily_forecast
        .set_index("date")
        .resample("W")["forecast"]
        .mean()
        .reset_index()
    )

    return {
        "model": model_name,
        "daily_forecast": daily_forecast,
        "weekly_forecast": weekly_forecast
    }