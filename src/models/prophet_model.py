from prophet import Prophet
import pandas as pd


# -----------------------------------
# TRAIN PROPHET
# -----------------------------------
def train_prophet(train_df):

    df = train_df[['date', 'total']].rename(
        columns={
            'date': 'ds',
            'total': 'y'
        }
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(df)

    return model


# -----------------------------------
# FORECAST
# -----------------------------------
def forecast_prophet(model, periods):

    future = model.make_future_dataframe(
        periods=periods,
        freq='D'
    )

    forecast = model.predict(future)

    # RETURN ONLY PREDICTION VALUES
    return forecast['yhat'].tail(periods).values