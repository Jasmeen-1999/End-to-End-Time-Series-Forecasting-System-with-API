from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from src.forecasting import forecast_state
from src.preprocessing import preprocess_data

app = FastAPI()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel("data/Forecasting Case- Study.xlsx")
df, scaler = preprocess_data(df)

df["state"] = df["state"].str.lower().str.strip()

# load model map
model_map = joblib.load("artifacts/model_map.pkl")


@app.get("/")
def home():
    return {"message": "Forecasting API is running"}


@app.get("/forecast/{state}")
def get_forecast(state: str):

    state = state.lower().strip()

    if state not in df["state"].unique():
        raise HTTPException(status_code=404, detail="State not found in dataset")

    if state not in model_map:
        raise HTTPException(status_code=404, detail="Model not trained for this state")

    state_df = df[df["state"] == state].copy()
    model_info = model_map[state]

    try:
        output = forecast_state(state_df, model_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "state": state,
        "model_used": output["model"],
        "daily_forecast": output["daily_forecast"].to_dict(orient="records"),
        "weekly_forecast": output["weekly_forecast"].to_dict(orient="records")
    }