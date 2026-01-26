import pandas as pd
import pickle
from lifelines import CoxPHFitter

DATA_PATH = "data/churn_features.csv"
MODEL_PATH = "survival-ml/survival_model.pkl"

df = pd.read_csv(DATA_PATH)

# Survival targets
df["duration"] = df["tenure_days"]
df["event"] = df["churn_30d"]

X = df.drop(columns=["customer_id", "churn_30d", "start_date"])

# Encode categoricals properly
X = pd.get_dummies(
    X,
    columns=["plan_type", "billing_cycle"],
    drop_first=True
)

# Adding survival columns
X["duration"] = df["duration"]
X["event"] = df["event"]

# Training Cox model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(X, duration_col="duration", event_col="event")

cph.print_summary()

# Saving model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(cph, f)

print("Survival model saved")
