import tensorflow as tf
import numpy as np
from itertools import product

from data_preprocessing import load_and_clean, scale_features, create_sequences
from config import *

def build_model(units1, units2, dropout):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units1, return_sequences=True,
                             input_shape=(SEQ_LEN, len(FEATURES))),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.LSTM(units2),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def tune():

    df = load_and_clean("data/indian_cities_weather.csv")
    scaled, _ = scale_features(df)
    X, y = create_sequences(scaled)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    param_grid = {
        "units1": [32, 64],
        "units2": [16, 32],
        "dropout": [0.2, 0.3]
    }

    best_loss = float("inf")
    best_params = None

    for u1, u2, d in product(
        param_grid["units1"],
        param_grid["units2"],
        param_grid["dropout"]
    ):
        print(f"\nTesting: {u1}, {u2}, {d}")

        model = build_model(u1, u2, d)

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )

        val_loss = min(history.history["val_loss"])

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = (u1, u2, d)

    print("\nBest Params:", best_params)
    print("Best Val Loss:", best_loss)

if __name__ == "__main__":
    tune()