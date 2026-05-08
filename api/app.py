from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Sales Forecasting API",
    description="End-to-End Time Series Forecasting System",
    version="1.0"
)

# Load model metadata
model_map = joblib.load("artifacts/model_map.pkl")


@app.get("/")
def home():

    return {
        "message": "Forecast API running",
        "available_states": list(model_map.keys())
    }


@app.get("/forecast")
def forecast(state: str):

    state = state.lower()

    # Validate state
    if state not in model_map:

        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found"
        )

    # Get model info
    info = model_map[state]

    model_name = info["best_model"]

    model_path = info["model_path"]

    metrics = info["metrics"]

    # Load trained model
    model = joblib.load(model_path)

    # Generate next 56 days
    future_dates = pd.date_range(
        start=pd.Timestamp.today().normalize(),
        periods=56,
        freq="D"
    )

    # ==========================================
    # ARIMA
    # ==========================================
    if model_name == "arima":

        preds = model.forecast(steps=56).tolist()

    # ==========================================
    # PROPHET
    # ==========================================
    elif model_name == "prophet":

        future = model.make_future_dataframe(
            periods=56
        )

        forecast_df = model.predict(future)

        preds = forecast_df["yhat"].tail(56).tolist()

    # ==========================================
    # XGBOOST
    # ==========================================
    elif model_name == "xgboost":

        preds = []

        # Initial rolling history
        history = [100000.0] * 30

        for i in range(56):

            current_date = future_dates[i]

            # Lag features
            lag_1 = history[-1]
            lag_7 = history[-7]
            lag_30 = history[-30]

            # Rolling statistics
            rolling_mean_7 = np.mean(history[-7:])
            rolling_std_7 = np.std(history[-7:])
            rolling_mean_30 = np.mean(history[-30:])

            # Feature row
            row = pd.DataFrame([{
                "total_scaled": lag_1,
                "day_of_week": current_date.dayofweek,
                "month": current_date.month,
                "is_holiday": 0,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_30": lag_30,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "rolling_mean_30": rolling_mean_30
            }])

            # Predict
            pred = float(model.predict(row)[0])

            # Prevent negative predictions
            pred = max(0, pred)

            preds.append(pred)

            # Update history recursively
            history.append(pred)

    # ==========================================
    # LSTM
    # ==========================================
    elif model_name == "lstm":

        # Replace with real LSTM inference later
        preds = [100000.0] * 56

    # ==========================================
    # FALLBACK
    # ==========================================
    else:

        preds = [0.0] * 56

    # Format response
    output = []

    for d, p in zip(future_dates, preds):

        output.append({
            "date": str(d.date()),
            "prediction": f"{round(float(p), 2):,.2f}"
        })

    return {
        "state": state,
        "model_used": model_name,
        "metrics": metrics,
        "forecast_horizon_days": 56,
        "forecast_8_weeks": output
    }