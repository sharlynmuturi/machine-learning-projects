import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

st.set_page_config(
    page_title="SaaS Customer Churn System",
    layout="wide",
    initial_sidebar_state="expanded"
)


BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "churn_features.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
SURVIVAL_MODEL_PATH = BASE_DIR / "survival-ml" / "survival_model.pkl"

# Cached Loaders
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_PATH)
    return clf

df = load_data()
model = load_models()

with open(SURVIVAL_MODEL_PATH, "rb") as f:
    survival_model = pickle.load(f)

# Prepping Features
FEATURES = df.drop(columns=["customer_id", "start_date", "churn_30d"])
for col in ["plan_type", "billing_cycle"]:
    FEATURES[col] = FEATURES[col].astype("category")

probs = model.predict_proba(FEATURES)[:, 1]
df["churn_risk"] = probs

def risk_band(p):
    if p >= 0.7:
        return "High"
    elif p >= 0.4:
        return "Medium"
    return "Low"

df["risk_band"] = df["churn_risk"].apply(risk_band)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Explorer"]
)

# Overview
if page == "Overview":
    st.title("SaaS Churn Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(df))
    col2.metric("Avg Churn Risk", f"{df.churn_risk.mean():.2%}")
    col3.metric("High Risk Customers", (df.risk_band == "High").sum())

    st.subheader("Risk Segmentation")
    band = st.selectbox("Select Risk Band", ["High", "Medium", "Low"])
    subset = df[df.risk_band == band]

    st.metric("Customers in Segment", len(subset))
    st.dataframe(
        subset[["customer_id", "churn_risk", "risk_band"]]
        .sort_values("churn_risk", ascending=False)
    )


    st.subheader("Churn Risk Distribution")
    st.bar_chart(df["risk_band"].value_counts())

# Customer Explorer
elif page == "Customer Explorer":
    st.title("Customer Explorer")

    customer_id = st.selectbox(
        "Select Customer",
        df["customer_id"].unique()
    )

    cust_X = FEATURES.loc[df.customer_id == customer_id]

    # Survival Analysis
    st.subheader("Survival Analysis (Churn Timing)")

    # Prepare the features for the survival model
    surv_X = pd.get_dummies(
        df.drop(columns=["customer_id", "churn_30d", "churn_risk", "risk_band"]),
        drop_first=True
    )

    # Selecting the row for the current customer
    cust_surv_X = surv_X.loc[cust_X.index]

    # Predicting survival function
    surv_func = survival_model.predict_survival_function(cust_surv_X)

    # Extracting probabilities at key days (30, 60, 90)
    days = [30, 60, 90]
    prob_churn = {}
    for d in days:
        # survival probability at day d
        if d in surv_func.index:
            surv_prob = surv_func.loc[d].values[0]
        else:
            # pick closest day
            idx = (surv_func.index - d).abs().argmin()
            surv_prob = surv_func.iloc[idx].values[0]
        prob_churn[d] = 1 - surv_prob  # convert survival to churn probability


    surv_table = pd.DataFrame({
        "Metric": ["Churn Probability at 30 days", 
                   "Churn Probability at 60 days", 
                   "Churn Probability at 90 days"],
        "Value": [f"{prob_churn[30]:.2%}", 
                  f"{prob_churn[60]:.2%}", 
                  f"{prob_churn[90]:.2%}"]
    })

    st.table(surv_table)

    # SHAP Explanation
    st.subheader("Top Features Driving Churn Risk")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(cust_X)[0]

    # Creating a DataFrame for easy display
    shap_df = pd.DataFrame({
        "Feature": cust_X.columns,       
        "Feature Value": cust_X.iloc[0].astype(str).values,
        "SHAP Value": shap_values
    })

    # Sorting by absolute impact
    shap_df["Abs Impact"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Abs Impact", ascending=False).drop(columns="Abs Impact")

    # highlighting positive vs negative contributions
    shap_df["Impact"] = shap_df["SHAP Value"].apply(lambda x: "Increases Risk" if x > 0 else "Reduces Risk")

    # Displaying top 10 features
    st.dataframe(shap_df.head(10).reset_index(drop=True))
