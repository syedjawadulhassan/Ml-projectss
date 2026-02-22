import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import *

os.makedirs("models", exist_ok=True)

def load_and_clean(path):
    import pandas as pd

    df = pd.read_csv(path)

    # normalize column names
    df.columns = df.columns.str.strip()

    # rename to project format
    df = df.rename(columns={
        'Date': 'date',
        'City': 'city',
        'Temperature_Max (°C)': 'max_temp',
        'Temperature_Min (°C)': 'min_temp',
        'Temperature_Avg (°C)': 'avg_temp',
        'Humidity (%)': 'humidity',
        'Rainfall (mm)': 'rainfall',
        'Wind_Speed (km/h)': 'wind_speed',
        'Pressure (hPa)': 'pressure',
        'Cloud_Cover (%)': 'cloud_cover'
    })

    # keep only needed columns
    df = df[
        ['date','city','max_temp','min_temp','avg_temp',
         'humidity','rainfall','wind_speed','pressure','cloud_cover']
    ]

    # convert date
    df['date'] = pd.to_datetime(df['date'])

    # filter city safely
    df['city'] = df['city'].astype(str)
    df = df[df['city'].str.lower() == CITY.lower()].sort_values('date')

    # fill missing
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df

def scale_features(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, "models/scaler.pkl")
    return scaled, scaler

def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN][TARGET_INDEX])
    return np.array(X), np.array(y)