import os
import warnings
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from src.preprocessing import preprocess_data
from src.feature_engineering import create_features

from src.models.arima import train_arima, forecast_arima
from src.models.prophet_model import train_prophet, forecast_prophet
from src.models.xgboost_model import train_xgb
from src.models.lstm import train_lstm, forecast_lstm


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/Forecasting Case- Study.xlsx"
MODEL_DIR = "artifacts/trained_models"
MODEL_MAP_PATH = "artifacts/model_map.pkl"

VAL_DAYS = 56
MIN_ROWS = 100

os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def debug_predictions(name, predictions):
    logger.info(f"{name} Prediction Sample: {predictions[:5]}")
    logger.info(f"{name} Contains NaN: {np.isnan(predictions).any()}")
    logger.info(f"{name} Contains Inf: {np.isinf(predictions).any()}")


def save_model(model, path):
    joblib.dump(model, path)
    logger.info(f"Model saved -> {path}")


# =========================================================
# LOAD DATA
# =========================================================

logger.info("Loading dataset...")

df = pd.read_excel(DATA_PATH)

logger.info(f"Dataset Shape: {df.shape}")

df, scaler = preprocess_data(df)

logger.info("Preprocessing completed")


# =========================================================
# TRAINING LOOP
# =========================================================

results = {}

for state in df["state"].unique():

    logger.info("=" * 60)
    logger.info(f"TRAINING STATE: {state}")
    logger.info("=" * 60)

    try:

        # =================================================
        # FILTER STATE DATA
        # =================================================

        state_df = (
            df[df["state"] == state]
            .copy()
            .sort_values("date")
        )

        if len(state_df) < MIN_ROWS:
            logger.warning(f"Skipping {state} -> insufficient rows")
            continue

        logger.info(f"Rows Available: {len(state_df)}")

        # =================================================
        # TRAIN / VALIDATION SPLIT
        # =================================================

        train = state_df.iloc[:-VAL_DAYS]
        val = state_df.iloc[-VAL_DAYS:]

        logger.info(f"Train Size: {len(train)}")
        logger.info(f"Validation Size: {len(val)}")

        metrics = {}

        arima_model = None
        prophet_model = None
        xgb_model = None
        lstm_model = None

        # =================================================
        # ARIMA
        # =================================================

        try:
            logger.info("Training ARIMA...")

            arima_model = train_arima(train["total"])

            arima_pred = forecast_arima(
                arima_model,
                len(val)
            )

            rmse_arima = calculate_rmse(
                val["total"],
                arima_pred
            )

            metrics["arima"] = rmse_arima

            logger.info(f"ARIMA RMSE: {rmse_arima:.2f}")

        except Exception as e:
            logger.exception(f"ARIMA failed for {state}")
            metrics["arima"] = float("inf")

        # =================================================
        # PROPHET
        # =================================================

        try:
            logger.info("Training Prophet...")

            prophet_model = train_prophet(train)

            prophet_pred = forecast_prophet(
                prophet_model,
                len(val)
            )

            rmse_prophet = calculate_rmse(
                val["total"],
                prophet_pred
            )

            metrics["prophet"] = rmse_prophet

            logger.info(f"Prophet RMSE: {rmse_prophet:.2f}")

        except Exception:
            logger.exception(f"Prophet failed for {state}")
            metrics["prophet"] = float("inf")

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        logger.info("Creating XGBoost features...")

        feat_df = create_features(state_df)

        feat_df = feat_df.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        logger.info(f"Feature DF Shape: {feat_df.shape}")

        train_feat = feat_df.iloc[:-VAL_DAYS]
        val_feat = feat_df.iloc[-VAL_DAYS:]

        drop_cols = ["total", "date", "state"]

        X_train = train_feat.drop(
            columns=drop_cols,
            errors="ignore"
        )

        y_train = train_feat["total"]

        X_val = val_feat.drop(
            columns=drop_cols,
            errors="ignore"
        )

        y_val = val_feat["total"]

        # =================================================
        # XGBOOST
        # =================================================

        try:
            logger.info("Training XGBoost...")

            X_val = X_val[X_train.columns]

            xgb_model = train_xgb(
                X_train,
                y_train
            )

            xgb_pred = xgb_model.predict(X_val)

            debug_predictions(
                "XGBoost",
                xgb_pred
            )

            rmse_xgb = calculate_rmse(
                y_val,
                xgb_pred
            )

            metrics["xgboost"] = rmse_xgb

            logger.info(f"XGBoost RMSE: {rmse_xgb:.2f}")

        except Exception:
            logger.exception(f"XGBoost failed for {state}")
            metrics["xgboost"] = float("inf")

        # =================================================
        # LSTM
        # =================================================

        try:
            logger.info("Training LSTM...")

            lstm_model = train_lstm(train)

            lstm_pred = forecast_lstm(
                lstm_model,
                train,
                steps=len(val)
            )

            debug_predictions(
                "LSTM",
                lstm_pred
            )

            rmse_lstm = calculate_rmse(
                val["total"],
                lstm_pred
            )

            metrics["lstm"] = rmse_lstm

            logger.info(f"LSTM RMSE: {rmse_lstm:.2f}")

        except Exception:
            logger.exception(f"LSTM failed for {state}")
            metrics["lstm"] = float("inf")

        # =================================================
        # BEST MODEL
        # =================================================

        logger.info("FINAL METRICS")
        logger.info(metrics)

        best_model = min(
            metrics,
            key=metrics.get
        )

        logger.info(f"Best Model: {best_model}")

        model_path = (
            f"{MODEL_DIR}/{state}_model.pkl"
        )

        if best_model == "arima" and arima_model:
            save_model(arima_model, model_path)

        elif best_model == "prophet" and prophet_model:
            save_model(prophet_model, model_path)

        elif best_model == "xgboost" and xgb_model:
            save_model(xgb_model, model_path)

        elif best_model == "lstm" and lstm_model:
            save_model(lstm_model, model_path)

        results[state] = {
            "best_model": best_model,
            "metrics": metrics,
            "model_path": model_path
        }

    except Exception:
        logger.exception(f"Fatal error for state: {state}")


# =========================================================
# SAVE MODEL MAP
# =========================================================

joblib.dump(
    results,
    MODEL_MAP_PATH
)

logger.info("Training completed successfully")