import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.express as px

st.set_page_config(
    page_title="AI Weather Intelligence",
    page_icon="🌦",
    layout="wide"
)

st.title("🌦 AI Weather Intelligence Dashboard")

@st.cache_resource
def load_model():
   return tf.keras.models.load_model("models/lstm_model.h5", compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/indian_cities_weather.csv")

    # normalize column names
    df.columns = df.columns.str.strip()

    # rename to standard format
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

    df['date'] = pd.to_datetime(df['date'])
    return df

model = load_model()
scaler = load_scaler()
df = load_data()

cities = sorted(df['city'].unique())
city = st.sidebar.selectbox("Select City", cities)

city_df = df[df['city'] == city].sort_values("date")

features = [
    'max_temp','min_temp','avg_temp',
    'humidity','rainfall','wind_speed',
    'pressure','cloud_cover'
]

SEQ_LEN = 60

if len(city_df) >= SEQ_LEN:

    latest = city_df[features].tail(SEQ_LEN).values
    latest_scaled = scaler.transform(latest)
    X_input = np.expand_dims(latest_scaled, axis=0)

    if st.button("Generate Forecast"):

        pred = model.predict(X_input, verbose=0)

        st.metric("Predicted Avg Temp", f"{pred[0][0]:.3f}")
        st.metric("Predicted Rainfall", f"{pred[0][1]:.3f}")

fig = px.line(city_df.tail(120), x="date", y="avg_temp",
              title="Temperature Trend")
st.plotly_chart(fig, use_container_width=True)