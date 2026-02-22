from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def perform_clustering(rfm, k=4):
    os.makedirs("models", exist_ok=True)

    # scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm)

    # save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # model
    model = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = model.fit_predict(scaled_data)

    # save model
    joblib.dump(model, "models/kmeans_model.pkl")

    return rfm