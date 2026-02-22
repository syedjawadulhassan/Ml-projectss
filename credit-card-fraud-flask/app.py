"""
Flask Web App for Credit Card Fraud Detection
Production-ready version
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-in-prod")

# -------------------------------------------------
# Load model and scaler once at startup
# -------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fraud_model.pkl")

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    print("✅ Model and scaler loaded successfully")
except Exception as exc:
    model = None
    scaler = None
    print(f"[ERROR] could not load model from {MODEL_PATH}: {exc}")

# -------------------------------------------------
# Feature order (MUST match training)
# -------------------------------------------------
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html", features=FEATURES)


@app.route("/result", methods=["POST"])
def result():
    if model is None or scaler is None:
        flash("Prediction model is not available.", "danger")
        return redirect(url_for("index"))

    try:
        # -----------------------------------------
        # Collect and validate input
        # -----------------------------------------
        data = []
        for feat in FEATURES:
            raw = request.form.get(feat)
            if raw is None or raw.strip() == "":
                raise ValueError(f"Missing value for {feat}")
            data.append(float(raw))

        # -----------------------------------------
        # Convert to DataFrame (professional fix)
        # -----------------------------------------
        features_df = pd.DataFrame([data], columns=FEATURES)

        # Apply same scaling used during training
        features_scaled = scaler.transform(features_df)

        # -----------------------------------------
        # Prediction
        # -----------------------------------------
        prediction = model.predict(features_scaled)[0]

        probability = (
            model.predict_proba(features_scaled)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        label = "Fraud" if prediction == 1 else "Legitimate"
        color = "danger" if prediction == 1 else "success"

        return render_template(
            "result.html",
            prediction=label,
            probability=probability,
            color=color,
        )

    except Exception as exc:
        flash(f"Error during prediction: {exc}", "danger")
        return redirect(url_for("predict"))


# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)