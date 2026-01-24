import streamlit as st
import pandas as pd
import datetime
from db.mongo import policies_col, claims_col, events_col, vehicles_col
from utils.fraud import compute_fraud_score

import joblib

from scripts.train_ml_model import extract_features

ml_model = joblib.load("models/fraud_model.pkl")

st.set_page_config(page_title="Insurance Fraud System", layout="wide")

st.title("Vehicle Insurance Fraud System")


tabs = ["Submit Claim", "View Claims"]
selected_tab = st.sidebar.radio("Select Tab", tabs)

# Submit Claim Tab
if selected_tab == "Submit Claim":
    st.header("Submit a New Claim")

    policies = list(policies_col.find({}))
    policy_options = {p["_id"]: f"{p['_id']} - {p['coverage_type']}" for p in policies}

    selected_policy_id = st.selectbox("Select Policy", options=list(policy_options.keys()),
                                      format_func=lambda x: policy_options[x])

    claim_type = st.selectbox("Claim Type", ["Accident", "Theft", "Fire"])

    claim_amount = st.number_input("Claim Amount (KES)", min_value=0, step=1000)

    accident_date = st.date_input("Accident Date", min_value=datetime.date(2023, 1, 1),
                                  max_value=datetime.date.today())

    location = st.text_input("Accident Location", value="Nairobi")

    repair_shop = st.text_input("Repair Shop", value="QuickFix Garage")

    if st.button("Submit Claim"):
        # Generate claim ID
        existing_claims = claims_col.count_documents({})
        claim_id = f"CLM{str(existing_claims+1).zfill(4)}"

        # Insert claim document
        claim_doc = {
            "_id": claim_id,
            "policy_id": selected_policy_id,
            "claim_amount": claim_amount,
            "claim_type": claim_type,
            "submitted_at": datetime.datetime.combine(accident_date, datetime.datetime.now().time()),
            "status": "under_review"
        }
        claims_col.insert_one(claim_doc)

        # Insert claim_submitted event
        event_doc = {
            "claim_id": claim_id,
            "event_type": "claim_submitted",
            "timestamp": claim_doc["submitted_at"],
            "metadata": {
                "location": location,
                "accident_type": claim_type,
                "repair_shop": repair_shop
            }
        }
        events_col.insert_one(event_doc)

        # Rule-based score
        rule_score = compute_fraud_score(claim_doc)

        # ML-based score (Isolation Forest outputs -1 for outlier, 1 for normal)
        features = extract_features(claim_doc)
        ml_pred = ml_model.predict(features)[0]  # -1 = fraud, 1 = normal
        ml_score = 50 if ml_pred == -1 else 0  # Scale ML score 0-50

        # Hybrid score: rule-based (0-50) + ML-based (0-50)
        fraud_score = min(rule_score + ml_score, 100)

        # Update claim document
        claims_col.update_one({"_id": claim_id}, {"$set": {
            "rule_score": rule_score,
            "ml_score": ml_score,
            "fraud_score": fraud_score
        }})

        st.success(f"Claim {claim_id} submitted successfully! Fraud Score: {fraud_score}")


# View Claims Tab
elif selected_tab == "View Claims":
    st.header("All Claims")

    claims = list(claims_col.find({}))
    if claims:
        # Converting to pandas DataFrame for easy display
        df = pd.DataFrame(claims)
        df["submitted_at"] = df["submitted_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["fraud_score"] = df.get("fraud_score", 0)
        df["rule_score"] = df.get("rule_score", 0)
        df["ml_score"] = df.get("ml_score", 0)

        # Highlight high-risk claims
        def highlight_risk(row):
            if row["fraud_score"] >= 60:
                return ["background-color: #FFCCCC"]*len(row)
            return [""]*len(row)

        st.dataframe(df[["_id", "policy_id", "claim_type", "claim_amount",
                         "submitted_at", "status", "rule_score", "ml_score", "fraud_score"]]
                    .style.apply(highlight_risk, axis=1))

        st.subheader("View Claim Events")
        selected_claim_id = st.selectbox("Select Claim ID", options=df["_id"])
        events = list(events_col.find({"claim_id": selected_claim_id}))
        if events:
            df_events = pd.DataFrame(events)
            df_events["timestamp"] = df_events["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.table(df_events[["event_type", "timestamp", "metadata"]])
        else:
            st.info("No events for this claim yet.")
    else:
        st.info("No claims found.")
