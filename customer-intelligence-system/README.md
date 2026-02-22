# 🚀 AI Customer Segmentation and Market Basket Intelligence System

An end-to-end Machine Learning project that performs customer segmentation using RFM analysis and KMeans clustering, along with Market Basket Analysis using Apriori algorithm to generate cross-sell product recommendations.

---

## 📌 Project Overview

This system analyzes retail transaction data to:

* Segment customers based on purchasing behavior
* Discover product purchase patterns
* Generate intelligent product recommendations
* Provide an interactive analytics dashboard

The project demonstrates practical applications of unsupervised learning and association rule mining in retail analytics.

---

## 🧠 Key Features

* RFM-based customer behavioral analysis
* KMeans customer segmentation
* Apriori-based market basket analysis
* Lift and confidence driven recommendations
* Interactive Streamlit dashboard
* Clean modular project structure
* Production-ready pipeline

---

## 🏗️ System Architecture

```
Transaction Data
      ↓
Data Preprocessing
      ↓
RFM Feature Engineering
      ↓
KMeans Clustering
      ↓
Apriori Association Mining
      ↓
Recommendation Engine
      ↓
Streamlit Dashboard
```

---

## 📁 Project Structure

```
customer-intelligence-system/
│
├── data/
│   └── sample_transactions.csv
│
├── src/
│   ├── preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   ├── association_rules.py
│   └── recommender.py
│
├── dashboard/
│   └── app.py
│
├── models/
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* MLxtend
* Streamlit
* Plotly

---

## 🚀 Installation and Setup

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd customer-intelligence-system
```

### 2️⃣ Create virtual environment

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**

```
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run dashboard/app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 How It Works

### 🔹 Customer Segmentation

* Computes Recency, Frequency, Monetary metrics
* Applies Standard Scaling
* Uses KMeans clustering
* Groups customers into behavioral segments

### 🔹 Market Basket Analysis

* Converts transactions into basket matrix
* Applies Apriori algorithm
* Generates association rules
* Ranks rules using lift and confidence

### 🔹 Recommendation Engine

* Accepts product input
* Finds strong association rules
* Suggests cross-sell products

---

## 🧪 Sample Test Inputs

Try entering:

* Milk
* Bread
* Butter
* Eggs

---

## 📈 Future Enhancements

* DBSCAN clustering comparison
* Customer Lifetime Value prediction
* Real-time FastAPI deployment
* Advanced visualization dashboard
* Large-scale retail dataset integration

---

## 👨‍💻 Author

**Jawad**
BTech CSE | Machine Learning Enthusiast

---

## ⭐ If you found this useful

Give the repository a star and share feedback.
