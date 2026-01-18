import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")


BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "churn_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Collecting inputs
input_data = {
    "gender": st.selectbox("Gender", ["Female", "Male"]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.number_input("Tenure (months)", min_value=0, max_value=100, value=12),
    "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No"]),
    "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No"]),
    "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No"]),
    "TechSupport": st.selectbox("Tech Support", ["Yes", "No"]),
    "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No"]),
    "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card(automatic)"]),
    "MonthlyCharges": st.number_input("Monthly Charges", 0.0),
    "TotalCharges": st.number_input("Total Charges", 0.0),
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    st.metric("Churn Probability", f"{prob:.2%}")
    st.success("High risk of churn" if pred else "Customer likely to stay")
