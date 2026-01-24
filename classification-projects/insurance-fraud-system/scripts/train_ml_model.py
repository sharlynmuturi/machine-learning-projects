import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
from db.mongo import claims_col

import numpy as np
import datetime
from db.mongo import claims_col, policies_col, vehicles_col, events_col

def extract_features(claim):
    """
    Create a feature vector for ML model
    """
    # Fetch policy and vehicle
    policy = policies_col.find_one({"_id": claim["policy_id"]})
    vehicle = vehicles_col.find_one({"_id": policy["vehicle_id"]})

    # Days since policy start
    days_since_start = (claim["submitted_at"] - policy["start_date"]).days

    # Claim amount / vehicle value ratio
    amount_ratio = claim["claim_amount"] / vehicle["estimated_value"]

    # Past claims in last 30 days
    start_window = claim["submitted_at"] - datetime.timedelta(days=30)
    recent_claims = list(claims_col.find({
        "policy_id": claim["policy_id"],
        "submitted_at": {"$gte": start_window, "$lt": claim["submitted_at"]}
    }))
    recent_claim_count = len(recent_claims)

    # Repair shop risk
    last_event = events_col.find_one({"claim_id": claim["_id"], "event_type": "repair_estimate_added"})
    risky_shops = ["QuickFix Garage"]
    repair_shop_risk = 1 if last_event and last_event["metadata"]["repair_shop"] in risky_shops else 0

    # Feature vector
    return np.array([amount_ratio, days_since_start, recent_claim_count, repair_shop_risk]).reshape(1, -1)


# Fetching all existing claims and building feature matrix
claims = list(claims_col.find({}))

X = np.vstack([extract_features(c) for c in claims])

# Training Isolation Forest
clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)

# Saving model
joblib.dump(clf, "models/fraud_model.pkl")
print("ML model trained and saved.")


# Updating ML scores for all existing claims
for claim in claims_col.find({}):
    features = extract_features(claim)
    ml_pred = clf.predict(features)[0]
    ml_score = 50 if ml_pred == -1 else 0
    fraud_score = min(claim.get("rule_score", 0) + ml_score, 100)
    claims_col.update_one(
        {"_id": claim["_id"]},
        {"$set": {"ml_score": ml_score, "fraud_score": fraud_score}}
    )

print("All existing claims updated with ML scores!")