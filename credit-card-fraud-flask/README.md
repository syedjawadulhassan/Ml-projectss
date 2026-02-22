# Credit Card Fraud Detection: Machine Learning Classification

This repository contains a complete end-to-end machine learning web application built using **Flask**. The project detects fraudulent credit card transactions in real time using models trained on the Kaggle Credit Card Fraud Detection dataset.

---

## 📌 Project Goal

Develop a real time fraud detection system that classifies credit card transactions as **Fraud** or **Legitimate** using machine learning.

## 🛠 Tech Stack

- Python
- Flask
- scikit-learn
- pandas, NumPy
- Bootstrap for frontend
- joblib for model persistence

## 📁 Project Structure

```
credit-card-fraud-flask/
│
├── app.py
├── model/
│   └── fraud_model.pkl      # trained pipeline
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   └── result.html
├── train_model.py
├── requirements.txt
└── README.md
```

## 🧠 Dataset

Use the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. Download the `creditcard.csv` file and place it in the root of the project before training.

Because the dataset is highly imbalanced, the training pipeline applies **SMOTE** oversampling to balance classes.

## 🔧 Training the Model

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put `creditcard.csv` in the project root.
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. The best model (Random Forest by default) will be saved to `model/fraud_model.pkl`.

During training the console will print accuracy, precision, recall, F1 score, and confusion matrix for each candidate model.

## 🚀 Running the Flask Application

1. Ensure `model/fraud_model.pkl` exists (run training first).
2. Start the server:
   ```bash
   python app.py
   ```
3. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Pages
- **Home** – overview and link to prediction page.
- **About** – project details and methodology.
- **Predict** – form to enter transaction features and submit for prediction.
- **Result** – shows whether the transaction is Fraud or Legitimate with probability and styled alert.

## 🖼 Sample Output

After training, use the Predict page to input the following example values:

```
Time: 0
V1: -1.3598
V2: -0.0728
...
Amount: 149.62
```

The app will return a green alert for **Legitimate** or a red alert for **Fraud** along with the probability score.

## ✅ Code Quality & Notes

- Paths are not hardcoded; joins use `os.path`.
- Flask app loads model once and handles errors gracefully.
- Training script is modular and well-documented.
- UI uses Bootstrap and responsive design, with clear alert colors.

Feel free to extend the application by adding user authentication, logging, or deploying to a cloud provider.
