import joblib

def load_models():
    model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def predict_customer(rfm_row):
    model, scaler = load_models()
    scaled = scaler.transform([rfm_row])
    cluster = model.predict(scaled)
    return cluster[0]