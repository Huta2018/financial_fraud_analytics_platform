import streamlit as st
import joblib
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------
# Fix path so Streamlit can find src modules
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.features_message import extract_manual_features


# -------------------------------------------------
# Required transformer for loading saved model
# -------------------------------------------------

class ManualFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame([extract_manual_features(text) for text in X])


# -------------------------------------------------
# Load models
# -------------------------------------------------

MESSAGE_MODEL_PATH = BASE_DIR / "models" / "message_fraud_model.pkl"
TRANSACTION_MODEL_PATH = BASE_DIR / "models" / "transaction_fraud_model.pkl"

message_model = joblib.load(MESSAGE_MODEL_PATH)
transaction_model = joblib.load(TRANSACTION_MODEL_PATH)


# -------------------------------------------------
# Risk scoring helper
# -------------------------------------------------

def risk_band(prob):

    score = prob * 100

    if score < 30:
        return score, "LOW", "green"

    elif score < 70:
        return score, "MEDIUM", "orange"

    else:
        return score, "HIGH", "red"


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(
    page_title="Credit Union Fraud Monitor",
    layout="wide"
)

st.title("Credit Union Fraud Detection Platform")

tab1, tab2 = st.tabs([
    "Communication Fraud Detection",
    "Transaction Fraud Detection"
])


# =================================================
# TAB 1 — MESSAGE FRAUD
# =================================================

with tab1:

    st.header("Communication Fraud Detection")

    message = st.text_area(
        "Enter financial message",
        height=150
    )

    if st.button("Analyze Message Fraud"):

        if len(message.strip()) == 0:
            st.warning("Please enter a message")
            st.stop()

        df = pd.DataFrame({"message": [message]})

        prob = message_model.predict_proba(df)[0][1]

        score, level, color = risk_band(prob)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Probability", f"{score:.2f}%")

        with col2:
            st.markdown(
                f"### Risk Level: <span style='color:{color}'>{level}</span>",
                unsafe_allow_html=True
            )

        features = extract_manual_features(message)

        st.subheader("Detected Fraud Signals")

        for key, val in features.items():
            if val > 0:
                st.write("-", key.replace("_", " "))


# =================================================
# TAB 2 — TRANSACTION FRAUD
# =================================================

with tab2:

    st.header("Transaction Fraud Detection")

    st.write(
        "Upload a transaction dataset to detect suspicious transactions."
    )

    file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        # Remove label column if present
        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        probs = transaction_model.predict_proba(df)[:, 1]

        df["fraud_probability"] = probs

        # -------------------------------------------------
        # Risk Level Categorization
        # -------------------------------------------------

        df["risk_level"] = df["fraud_probability"].apply(
            lambda x: "HIGH" if x > 0.8 else "MEDIUM" if x > 0.3 else "LOW"
        )

        st.subheader("Transaction Data Preview")
        st.write(df.head())

        # -------------------------------------------------
        # Fraud Probability Distribution
        # -------------------------------------------------

        st.subheader("Fraud Probability Distribution")

        fig = px.histogram(
            df,
            x="fraud_probability",
            nbins=50,
            title="Distribution of Fraud Probabilities"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------
        # Risk Level Summary Chart
        # -------------------------------------------------

        st.subheader("Fraud Risk Category Summary")

        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig2 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            title="Fraud Risk Categories"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------------------------
        # High Risk Transactions
        # -------------------------------------------------

        st.subheader("High Risk Transactions")

        high_risk = df[df["fraud_probability"] > 0.8]

        st.write(high_risk.head())

        st.metric(
            "Number of High Risk Transactions",
            len(high_risk)
        )

        # -------------------------------------------------
        # Top Suspicious Transactions
        # -------------------------------------------------

        st.subheader("Top Suspicious Transactions")

        top_fraud = df.sort_values(
            "fraud_probability",
            ascending=False
        ).head(10)

        st.write(top_fraud)
