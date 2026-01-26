import joblib
import pandas as pd

MODEL_PATH = "ml/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def risk_band(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def predict_churn(df):
    model = load_model()

    X = df.drop(columns=["customer_id", "churn_30d"])
    for col in ["plan_type", "billing_cycle"]:
        X[col] = X[col].astype("category")

    probs = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": probs,
        "risk_band": [risk_band(p) for p in probs]
    })