import shap
import pandas as pd
import joblib

MODEL_PATH = "ml/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def prepare_features(df):
    X = df.drop(columns=["customer_id", "churn_30d"])
    for col in ["plan_type", "billing_cycle"]:
        X[col] = X[col].astype("category")
    return X

def global_shap_summary(df):
    model = load_model()
    X = prepare_features(df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1]

    return shap_values, X

def explain_customer(model, explainer, X, customer_idx, top_n=5):
    shap_values = explainer.shap_values(X)[1][customer_idx]

    top_drivers = (
        pd.DataFrame({
            "feature": X.columns,
            "impact": shap_values
        })
        .sort_values("impact", key=abs, ascending=False)
        .head(top_n)
    )

    return top_drivers