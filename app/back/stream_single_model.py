import streamlit as st
import joblib
import pandas as pd
import sys
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------
# Fix module path so Streamlit sees project folders
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.features_message import extract_manual_features


# -------------------------------------------------
# Recreate transformer class (required for joblib)
# -------------------------------------------------

class ManualFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame([extract_manual_features(text) for text in X])


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

MODEL_PATH = BASE_DIR / "models" / "message_fraud_model.pkl"

model = joblib.load(MODEL_PATH)


# -------------------------------------------------
# Risk scoring
# -------------------------------------------------

def risk_band(probability):

    score = probability * 100

    if score < 30:
        return score, "LOW", "green"

    elif score < 70:
        return score, "MEDIUM", "orange"

    else:
        return score, "HIGH", "red"


# -------------------------------------------------
# Fraud signal explanation
# -------------------------------------------------

def investigator_guidance(features):

    signals = []

    if features["urgency_score"] > 0:
        signals.append("Urgency language detected")

    if features["authority_score"] > 0:
        signals.append("Authority impersonation language")

    if features["payment_score"] > 0:
        signals.append("Payment request detected")

    if features["amount_flag"] == 1:
        signals.append("Money amount mentioned")

    if features["bank_change_flag"] == 1:
        signals.append("Bank account change language")

    if features["secrecy_score"] > 0:
        signals.append("Secrecy request detected")

    if len(signals) == 0:
        signals.append("No strong fraud signals detected")

    return signals


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(
    page_title="Credit Union Fraud Monitor",
    layout="wide"
)

st.title("Credit Union Fraud Detection System")

st.write(
"""
This dashboard analyzes financial communications and flags potential fraud risks
such as business email compromise, payment redirection, and executive impersonation.
"""
)


message = st.text_area(
    "Enter financial message to analyze",
    height=150
)


if st.button("Analyze Fraud Risk"):

    if len(message.strip()) == 0:
        st.warning("Please enter a message.")
        st.stop()

    df = pd.DataFrame({"message": [message]})

    probability = model.predict_proba(df)[0][1]

    score, level, color = risk_band(probability)

    st.subheader("Fraud Risk Assessment")

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

    signals = investigator_guidance(features)

    for s in signals:
        st.write("-", s)

    st.subheader("Recommended Investigator Action")

    if level == "HIGH":

        st.error(
"""
High fraud risk detected.

Recommended actions:
• Verify request through known phone number  
• Confirm sender identity  
• Review payment authorization process  
• Escalate to fraud investigation team
"""
        )

    elif level == "MEDIUM":

        st.warning(
"""
Moderate fraud risk.

Recommended actions:
• Validate request with sender  
• Confirm vendor bank account  
• Review communication context
"""
        )

    else:

        st.success(
"""
Low fraud risk.

No strong fraud indicators detected.
"""
        )
