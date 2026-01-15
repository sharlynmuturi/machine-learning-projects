import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading model & encoders
model = joblib.load("models/rf_credit_model.pkl")
encoders = {col: joblib.load(f"models/{col}_encoder.pkl") for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', "Age_bin"]}

# User inputs (raw)
st.title("Credit Risk Prediction App")
st.write("Enter applicant information")

age = st.number_input("Age", min_value = 18, max_value = 80, value = 35)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value = 0, max_value = 3, value = 1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["unknown", "little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Account", ["unknown", "little", "moderate", "rich"])
purpose = st.selectbox("Purpose", ["car", "radio/TV", "furniture/equipment", "education", "business", "repairs", "domestic appliances", "vacation/others"])
credit_amount = st.number_input("Credit Amount", min_value = 0, value = 1500)
duration = st.number_input("Duration (months)", min_value = 1, value = 12)

# Reproducing Feature Engineering
age_bin = pd.cut([age], bins=[18, 25, 35, 45, 60, 100], labels=['18-25', '26-35', '36-45', '46-60', '60+'])[0]
credit_per_month = credit_amount / duration
has_savings = 0 if saving_accounts == "unknown" else 1
has_checking = 0 if checking_account == "unknown" else 1

# Encoding categorical variables (same encoders as training)
input_df = pd.DataFrame({
    "Age_bin": [encoders["Age_bin"].transform([age_bin])[0]],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Credit_per_month": [credit_per_month],
    "Has_savings": [has_savings],
    "Has_checking": [has_checking]
})


if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"High Credit Risk (Probability: {prob:.2%})")
    else:
        st.success(f"Low Credit Risk (Probability: {prob:.2%})")
