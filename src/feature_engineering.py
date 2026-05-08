import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # -----------------------------
    # DATE FEATURES
    # -----------------------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # simple weekend holiday flag
    df["is_holiday"] = df["day_of_week"].isin([5, 6]).astype(int)

    # -----------------------------
    # LAG FEATURES
    # -----------------------------
    df["lag_1"] = df["total_scaled"].shift(1)
    df["lag_7"] = df["total_scaled"].shift(7)
    df["lag_30"] = df["total_scaled"].shift(30)

    # -----------------------------
    # ROLLING FEATURES
    # -----------------------------
    df["rolling_mean_7"] = df["total_scaled"].rolling(7).mean()
    df["rolling_std_7"] = df["total_scaled"].rolling(7).std()

    df["rolling_mean_30"] = df["total_scaled"].rolling(30).mean()

    # -----------------------------
    # REMOVE NULLS
    # -----------------------------
    df = df.dropna().reset_index(drop=True)

    return df