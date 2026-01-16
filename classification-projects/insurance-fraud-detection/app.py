import streamlit as st
import joblib
import pandas as pd
import os

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgb

# Getting absolute path to current file
BASE_DIR = os.path.dirname(__file__)

# Load preprocessor and model
preprocessor_path = os.path.join(BASE_DIR, "preprocessor.pkl")
preprocessor = joblib.load(preprocessor_path)

model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)


st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
st.title("Vehicle Insurance Fraud Detection")

st.markdown("Please enter the claim details below:")

st.subheader("Demographics & Behavior")

Sex = st.selectbox("Sex", ['Female', 'Male'])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Widow', 'Divorced'])
AgeOfPolicyHolder = st.selectbox("Age Of PolicyHolder", ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25', '36 to 40', '16 to 17', 'over 65', '18 to 20'])

st.subheader("Vehicle & Risk Signals")

Make = st.selectbox("Vehicle Make", ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
VehiclePrice = st.selectbox("Vehicle Price", ['more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000', '40000 to 59000', '60000 to 69000'])
AgeOfVehicle = st.selectbox("Age Of Vehicle", ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years'])
NumberOfCars = st.selectbox("Number Of Cars", ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'])

st.subheader("Claim Circumstances")

Fault = st.selectbox("Fault", ['Policy Holder', 'Third Party'])
AccidentArea = st.selectbox("Area Accident Occurred", ['Urban', 'Rural'])
PoliceReportFiled = st.selectbox("Police Report Filed", ['No', 'Yes'])
WitnessPresent = st.selectbox("Witness Present", ['No', 'Yes'])
AgentType = st.selectbox("Agent Type", ['External', 'Internal'])
Month = st.selectbox("Month Accident Occurred", ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug', 'Jul', 'May', 'Sep'])
DayOfWeek = st.selectbox("Day of Week Accident Occurred", ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday', 'Thursday'])
WeekOfMonth = st.number_input("Week of Month Accident Occurred", min_value = 1, max_value = 5, value = 1)
PolicyType = st.selectbox("Policy Type", ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])

st.subheader("Policy & History")

Deductible = st.selectbox("Deductible", [300, 400, 500, 700])
DriverRating = st.selectbox("Driver Rating", [1, 4, 3, 2])
PastNumberOfClaims = st.selectbox("Past Number Of Claims", ['more than 30', '15 to 30', '8 to 15'])
Days_Policy_Accident = st.selectbox("Days Policy Accident", ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'])
Days_Policy_Claim = st.selectbox("Days Policy Claim", ['more than 30', '15 to 30', '8 to 15'])
AddressChange_Claim = st.selectbox("Address Change Claim ", ['1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months'])

if st.button("Predict Fraud Risk"):
    input_data = pd.DataFrame([{
        'Sex': Sex,
        'MaritalStatus': MaritalStatus,
        'AgeOfPolicyHolder': AgeOfPolicyHolder,
        'Make': Make,
        'VehiclePrice': VehiclePrice,
        'AgeOfVehicle': AgeOfVehicle,
        'NumberOfCars': NumberOfCars,
        'Fault': Fault,
        'AccidentArea': AccidentArea,
        'PoliceReportFiled': PoliceReportFiled,
        'WitnessPresent': WitnessPresent,
        'AgentType': AgentType,
        'Month': Month,
        'DayOfWeek': DayOfWeek,
        'WeekOfMonth': WeekOfMonth,
        'PolicyType': PolicyType,
        'Deductible': Deductible,
        'DriverRating': DriverRating,
        'PastNumberOfClaims': PastNumberOfClaims,
        'Days_Policy_Accident': Days_Policy_Accident,
        'Days_Policy_Claim': Days_Policy_Claim,
        'AddressChange_Claim': AddressChange_Claim
    }])

    
    # Transform input
    input_data = preprocessor.transform(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Fraud Risk (Probability: {probability:.2%})")
    else:
        st.success(f"Low Fraud Risk (Probability: {probability:.2%})")
