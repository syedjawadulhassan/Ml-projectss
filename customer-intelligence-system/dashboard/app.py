import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.rfm_analysis import create_rfm
from src.clustering import perform_clustering
from src.association_rules import market_basket
from src.recommender import recommend_products

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

st.title("AI Customer Segmentation and Market Basket System")

# ================= FILE UPLOADER =================
file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # ================= KPI =================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Unique Customers", df["CustomerID"].nunique())
    col3.metric("Unique Products", df["Description"].nunique())

    # ================= SEGMENTATION =================
    if st.button("Run Customer Segmentation"):
        rfm = create_rfm(df)
        clustered = perform_clustering(rfm)

        st.success("Segmentation Completed")
        st.dataframe(clustered)

    # ================= MBA =================
    if st.button("Run Market Basket Analysis"):
        st.session_state["rules"] = market_basket(df)
        st.success("Association Rules Generated")

    # ================= SHOW RULES =================
    if "rules" in st.session_state:
        rules = st.session_state["rules"]
        st.subheader("Association Rules")
        st.dataframe(rules.head())

        # ================= RECOMMENDER =================
        product = st.text_input("Enter product for recommendation")

        if product:
            recs = recommend_products(rules, product)

            if len(recs) > 0:
                st.subheader("Recommended Products")
                st.dataframe(recs)
            else:
                st.warning("No strong recommendations found for this product")