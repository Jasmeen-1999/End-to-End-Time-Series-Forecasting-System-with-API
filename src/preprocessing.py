import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):

    # normalize columns
    df.columns = df.columns.str.strip().str.lower()

    required = ['state', 'date', 'total']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['state'] = df['state'].astype(str).str.strip().str.lower()

    df = df.dropna(subset=['date', 'total'])

    # aggregate duplicates
    df = df.groupby(['state', 'date'], as_index=False)['total'].sum()

    # sort
    df = df.sort_values(['state', 'date'])

    # fill missing dates per state
    df = df.set_index('date').groupby('state').apply(
        lambda x: x.asfreq('D')
    ).reset_index()

    # fill missing values
    df['total'] = df['total'].ffill().bfill()

    # scaler for ML models (XGBoost + LSTM)
    scaler = MinMaxScaler()
    df['total_scaled'] = scaler.fit_transform(df[['total']])

    return df, scaler