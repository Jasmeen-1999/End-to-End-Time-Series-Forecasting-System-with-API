import joblib
import pandas as pd


def load_model(state):

    model_path = f"artifacts/trained_models/{state}_model.pkl"

    model = joblib.load(model_path)

    return model


def forecast_next_8_weeks(state):

    model = load_model(state)

    # temporary dummy predictions
    predictions = [100] * 56

    return predictions