import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_preprocessing import load_and_clean, scale_features, create_sequences
from utils import set_seed
from config import *

set_seed()

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32),
        Dropout(0.3),

        Dense(16, activation='relu'),
        Dense(2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def train():
    df = load_and_clean("data/indian_cities_weather.csv")
    scaled, _ = scale_features(df)
    X, y = create_sequences(scaled)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((SEQ_LEN, X.shape[2]))

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ]

    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    model.save("models/lstm_model.h5")
    print("Model saved successfully")

if __name__ == "__main__":
    train()