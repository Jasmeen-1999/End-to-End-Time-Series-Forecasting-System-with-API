import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense


# -----------------------------
# CREATE SEQUENCES
# -----------------------------
def create_sequences(data, window=14):
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)


# -----------------------------
# TRAIN LSTM
# -----------------------------
def train_lstm(df):

    df = df.copy().sort_values("date")

    values = df["total"].values.reshape(-1, 1)

    # scaling
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    if len(scaled_values) <= 20:
        raise ValueError("Not enough data for LSTM")

    # sequences
    X, y = create_sequences(scaled_values, window=14)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(14, 1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    # attach metadata
    model.scaler = scaler
    model.window = 14

    return model


# -----------------------------
# FORECAST (8 WEEKS = 56 DAYS)
# -----------------------------
def forecast_lstm(model, df, steps=56):

    df = df.copy().sort_values("date")

    values = df["total"].values.reshape(-1, 1)

    scaled = model.scaler.transform(values)

    window = model.window

    if len(scaled) < window:
        raise ValueError("Not enough data for forecasting")

    input_seq = scaled[-window:].reshape(1, window, 1)

    predictions = []

    for _ in range(steps):

        pred = model.predict(input_seq, verbose=0)[0][0]

        predictions.append(pred)

        # FIX: safer update (no dtype issues)
        next_input = np.array([[pred]])
        input_seq = np.concatenate(
            (input_seq[:, 1:, :], next_input.reshape(1, 1, 1)),
            axis=1
        )

    # inverse scale
    predictions = model.scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten()