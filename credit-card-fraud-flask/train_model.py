import os
import argparse
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "creditcard.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "fraud_model.pkl")


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load CSV dataset from `path`.

    If the file is missing, a FileNotFoundError is raised so the caller
    can decide to create a demo dataset instead.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop missing values and split into features/target.

    Assumes target column is named `Class` as in the Kaggle dataset.
    """
    df = df.dropna()
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def balance_data(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance the minority class.

    Returns resampled X and y as numpy arrays.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("Class distribution after SMOTE:", np.bincount(y_res))
    return X_res, y_res


def train_models(X_train, y_train):
    """Train candidate models inside scaling pipelines and return dict.

    The pipeline handles scaling so the saved object can accept raw features
    in the same order as training data (no separate scaler needed).
    """
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    pipelines = {}
    for name, mdl in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
    return pipelines


def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm


def create_demo_dataset(path: str = DATA_FILE, n_samples: int = 2000) -> None:
    """Create a small synthetic dataset resembling the real feature layout.

    This function is meant to help quickly generate a toy CSV for local
    development/demo when the real Kaggle file is not available.
    """
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, len(cols) - 1))
    # Create an imbalanced target
    y = rng.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    df = pd.DataFrame(X, columns=cols[:-1])
    df["Class"] = y
    df.to_csv(path, index=False)
    print(f"Demo dataset written to {path}")


def main(demo: bool = False):
    if demo:
        if not os.path.exists(DATA_FILE):
            print("Creating demo dataset...")
            create_demo_dataset(DATA_FILE, n_samples=2000)

    print("Loading data...")
    df = load_data()
    X, y = preprocess(df)

    print("Balancing classes with SMOTE...")
    X_bal, y_bal = balance_data(X, y)

    print("Performing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    print("Training candidate models...")
    pipelines = train_models(X_train, y_train)

    best_pipe = None
    best_f1 = -1
    for name, pipe in pipelines.items():
        acc, prec, rec, f1, cm = evaluate_model(pipe, X_test, y_test)
        print(f"\n{name} results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 score: {f1:.4f}")
        print("Confusion matrix:\n", cm)
        if f1 > best_f1:
            best_f1 = f1
            best_pipe = pipe

    if best_pipe is not None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_pipe, MODEL_FILE, compress=3)
        print(f"\nBest model saved to {MODEL_FILE} (F1={best_f1:.4f})")
    else:
        print("No model was trained. Something went wrong.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--demo", action="store_true", help="Create demo dataset if missing and train quickly")
    args = parser.parse_args()
    main(demo=args.demo)
