import tensorflow as tf
from data_preprocessing import load_and_clean, scale_features, create_sequences
from utils import print_metrics
from config import *

def evaluate():
    # load and prepare data
    df = load_and_clean("data/indian_cities_weather.csv")
    scaled, _ = scale_features(df)
    X, y = create_sequences(scaled)

    split = int(len(X) * TRAIN_SPLIT)
    X_test, y_test = X[split:], y[split:]

    # ✅ SAFE MODEL LOAD (important fix)
    model = tf.keras.models.load_model(
        "models/lstm_model.h5",
        compile=False
    )

    preds = model.predict(X_test, verbose=0)

    print_metrics(y_test, preds)

if __name__ == "__main__":
    evaluate()