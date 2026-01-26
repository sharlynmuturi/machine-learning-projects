import pandas as pd
import joblib

MODEL_PATH = "survival-ml/survival_model.pkl"
DATA_PATH = "data/churn_features.csv"

cph = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customer_id", "churn_30d"])
X = pd.get_dummies(X, drop_first=True)

# Picking a customer
customer_idx = 10
customer_features = X.iloc[[customer_idx]]

survival_curve = cph.predict_survival_function(customer_features)

print(survival_curve.head())
