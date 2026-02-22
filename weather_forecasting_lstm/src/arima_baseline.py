import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from config import CITY

def run_arima():
    # load raw data
    df = pd.read_csv("data/indian_cities_weather.csv")

    # normalize column names (IMPORTANT FIX)
    df.columns = df.columns.str.strip()

    # rename to standard format
    df = df.rename(columns={
        'Date': 'date',
        'City': 'city',
        'Temperature_Avg (°C)': 'avg_temp'
    })

    # filter city safely
    df['city'] = df['city'].astype(str)
    df = df[df['city'].str.lower() == CITY.lower()]

    if len(df) == 0:
        raise ValueError(f"No data found for city: {CITY}")

    series = df['avg_temp']

    # ARIMA model
    model = ARIMA(series, order=(5,1,0))
    fitted = model.fit()

    print(fitted.summary())

if __name__ == "__main__":
    run_arima()