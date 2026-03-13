import streamlit as st
import joblib
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

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

st.title("🏦 Credit Union Fraud Detection Platform")

st.markdown(
"""
AI-powered monitoring system for detecting **communication fraud and suspicious financial transactions**.
"""
)

tab1, tab2 = st.tabs([
    "📩 Communication Fraud Detection",
    "💳 Transaction Fraud Detection"
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

        # -------------------------------------------------
        # Fraud Risk Gauge
        # -------------------------------------------------

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------
        # Highlight Suspicious Words
        # -------------------------------------------------

        st.subheader("Suspicious Language Highlight")

        suspicious_words = [
            "urgent",
            "immediately",
            "transfer",
            "wire",
            "ceo",
            "confidential",
            "asap"
        ]

        highlighted_message = message

        for word in suspicious_words:
            highlighted_message = highlighted_message.replace(
                word,
                f"**:red[{word}]**"
            )

        st.markdown(highlighted_message)

        # -------------------------------------------------
        # Fraud Signals
        # -------------------------------------------------

        features = extract_manual_features(message)

        st.subheader("Detected Fraud Signals")

        signal_df = pd.DataFrame(
            list(features.items()),
            columns=["Signal", "Detected"]
        )

        signal_df["Detected"] = signal_df["Detected"].astype(int)

        detected_signals = signal_df[signal_df["Detected"] > 0]

        if len(detected_signals) > 0:

            st.dataframe(detected_signals)

            fig_signals = px.bar(
                detected_signals,
                x="Signal",
                y="Detected",
                title="Fraud Signal Activation"
            )

            st.plotly_chart(fig_signals, use_container_width=True)

        else:
            st.success("No suspicious signals detected")


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

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        probs = transaction_model.predict_proba(df)[:, 1]

        df["fraud_probability"] = probs

        df["risk_level"] = df["fraud_probability"].apply(
            lambda x: "HIGH" if x > 0.8 else "MEDIUM" if x > 0.3 else "LOW"
        )

        st.subheader("Transaction Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------------------------------------
        # Fraud Statistics
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(df))
        col2.metric("High Risk", len(df[df["risk_level"] == "HIGH"]))
        col3.metric("Medium Risk", len(df[df["risk_level"] == "MEDIUM"]))

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
        # Risk Category Chart
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

        st.dataframe(high_risk, use_container_width=True)

        st.metric(
            "Number of High Risk Transactions",
            len(high_risk)
        )

        # -------------------------------------------------
        # Top Fraud Leaderboard
        # -------------------------------------------------

        st.subheader("Top Suspicious Transactions")

        top_fraud = df.sort_values(
            "fraud_probability",
            ascending=False
        ).head(10)

        fig3 = px.bar(
            top_fraud,
            x=top_fraud.index,
            y="fraud_probability",
            title="Top 10 Suspicious Transactions"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(top_fraud, use_container_width=True)

        # -------------------------------------------------
        # Download flagged transactions
        # -------------------------------------------------

        csv = high_risk.to_csv(index=False)

        st.download_button(
            label="Download High Risk Transactions",
            data=csv,
            file_name="flagged_transactions.csv",
            mime="text/csv"
        )
