import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

DATA_PATH = "data/churn_features.csv"
MODEL_PATH = "ml/model.pkl"

# Loading data
df = pd.read_csv(DATA_PATH)

TARGET = "churn_30d"
CATEGORICALS = ["plan_type", "billing_cycle"]

X = df.drop(columns=[TARGET, "customer_id", "start_date"])
y = df[TARGET]

# Convert categoricals
for col in CATEGORICALS:
    X[col] = X[col].astype("category")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# Model
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# Evaluation
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print(f"ROC-AUC: {auc:.3f}")
print(classification_report(y_test, model.predict(X_test)))


# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved to ml/model.pkl")
