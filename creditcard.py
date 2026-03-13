import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

image1 = Image.open("images/Fraud1.jpg")
image2 = Image.open("images/fraud3.jpg")

#load models
model = joblib.load("best_model_RFC.pkl")
label_encoders = joblib.load("label_encoder_RFC.pkl")
scaler= joblib.load("scaler_RFC.pkl")
#======================================================================================
#add background
st.markdown("""
<style>
.block-container {
    background: rgba(30, 30, 30, 0.85);
    padding: 2rem;
    border-radius: 16px;
}

/* Make labels visible */
label {
    color: white !important;
    font-weight: 500;
}

/* Input text color */
input, select {
    color: white !important;
    background-color: #1e1e1e !important;
}
</style>
""", unsafe_allow_html=True)

#================================================================================

#title and sub title with images
st.markdown(
    """
    <h1 style='text-align:center;font-size:48px;font-weight:800;color:#C9A227;'>
        CREDITCARD TRANSACTION FRAUD DETECTION
    </h1>
    <p style='text-align:center;font-size:24px;color:#C9A227;margin-top:-10px;'>
        Provide Transaction Details To Predict Potential Fraud
    </p>
    """,
    unsafe_allow_html=True
)

# --- Images ---
col1, col2 = st.columns([1,1])

with col1:
    st.image(image1, width=280)

with col2:
    st.image(image2, width=280)

st.markdown("<br>", unsafe_allow_html=True)

# ================= USER INPUT =================

merchant_category = st.selectbox("Merchant Category",["Grocery","Electronics","Fuel","unknown","Restaurant","Travel"])
transaction_type = st.selectbox("Transaction Type",["Online","POS","ATM"])
transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=100.0)
account_balance = st.number_input("Account Balance", min_value=-26000.0, value=50000.0, step=100.0)
transaction_hour = st.number_input("Transaction Hour", min_value=0, max_value=23,value=12, step=1)

is_international_ui = st.selectbox("Is International?", ["Yes","No"])
is_international = 1 if is_international_ui == "Yes" else 0

device_type = st.selectbox("Device Type", ["Mobile","Web","Card","unknown"])
transaction_channel = st.selectbox("Transaction Channel", ["Online","POS","ATM"])
txn_count_last_24h = st.number_input("Txn Count(last 24h)", min_value=0, value=5, step=1)

# ================= ENCODING =================

merchant_category_encoded = label_encoders["merchant_category"].transform([merchant_category])[0]
transaction_type_encoded = label_encoders["transaction_type"].transform([transaction_type])[0]
device_type_encoded = label_encoders["device_type"].transform([device_type])[0]
transaction_channel_encoded = label_encoders["transaction_channel"].transform([transaction_channel])[0]

# ================= PREDICT BUTTON =================

if st.button("Predict Fraud"):

    input_data = [[
        merchant_category_encoded,
        transaction_type_encoded,
        transaction_amount,
        account_balance,
        transaction_hour,
        is_international,
        device_type_encoded,
        transaction_channel_encoded,
        txn_count_last_24h
    ]]

    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0]

    fraud_prob = proba[1]
    THRESHOLD = 0.30

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------- RESULT ----------
    if fraud_prob >= THRESHOLD:
        st.markdown(
            "<h2 style='color:#ff4b4b; text-align:center;'>🚨 Fraudulent Transaction Chance High</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='color:#00ff9c; text-align:center;'>✅ Transaction is Legitimate</h2>",
            unsafe_allow_html=True
        )

    # ---------- CONFIDENCE ----------
    st.markdown("<h3 style='color:white; text-align:center;'>Prediction Confidence</h3>", unsafe_allow_html=True)
    st.progress(int(max(proba) * 100))

    st.markdown(
        f"""
        <p style='color:white;font-size:18px;text-align:center;'>
        Legitimate Probability: <b>{round(proba[0],2)}</b><br>
        Fraud Probability: <b>{round(proba[1],2)}</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    # ---------- CHART ----------
    proba_df = pd.DataFrame({
        "Class": ["Legitimate","Fraud"],
        "Probability": [proba[0]*100, proba[1]*100]
    })

    st.markdown("<h3 style='color:white; text-align:center;'>Prediction Probability Distribution</h3>", unsafe_allow_html=True)
    st.bar_chart(proba_df.set_index("Class"))

    # ---------- RISK ----------
    if fraud_prob > 0.40:
        risk = "High Risk"
        color = "#ff4b4b"
    elif fraud_prob > 0.30:
        risk = "Medium Risk"
        color = "#ffa500"
    else:
        risk = "Low Risk"
        color = "#00ff9c"

    st.markdown("<h3 style='color:white; text-align:center;'>Risk Assessment</h3>", unsafe_allow_html=True)

    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>⚠️ {risk}</h2>",
        unsafe_allow_html=True
    )