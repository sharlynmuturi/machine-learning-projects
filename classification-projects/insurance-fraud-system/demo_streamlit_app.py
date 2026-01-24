import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os

st.set_page_config(page_title="Vehicle Insurance Fraud System", layout="wide")
st.title("Vehicle Insurance Fraud Detection System")

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
ml_model = joblib.load(model_path)


def compute_fraud_score_demo(claim):
    score = 0

    # High claim amount
    if claim["claim_amount"] > 300_000:
        score += 30

    # Risky claim types
    if claim["claim_type"] in ["Theft", "Fire"]:
        score += 20

    # Simple policy risk proxy
    if str(claim["policy_id"]).endswith("3"):
        score += 10

    return min(score, 50)

import numpy as np

def extract_features_demo(claim):
    # fixed proxy for vehicle value
    estimated_vehicle_value = 1_000_000
    amount_ratio = claim["claim_amount"] / estimated_vehicle_value

    # fixed proxy for policy start date
    days_since_start = 180

    # No history in demo
    recent_claim_count = 0

    # repair_shop_risk
    risky_shops = ["QuickFix Garage"]
    repair_shop_risk = int(claim.get("repair_shop") in risky_shops)

    return np.array([
        amount_ratio,
        days_since_start,
        recent_claim_count,
        repair_shop_risk
    ]).reshape(1, -1)

# In-Memory Storage to act as DB
if "claims" not in st.session_state:
    st.session_state.claims = []

if "events" not in st.session_state:
    st.session_state.events = []


# Sample Policies
policies = [
    {"policy_id": "POL001", "coverage_type": "Comprehensive"},
    {"policy_id": "POL002", "coverage_type": "Third Party"},
    {"policy_id": "POL003", "coverage_type": "Commercial"},
]

policy_options = {
    p["policy_id"]: f"{p['policy_id']} - {p['coverage_type']}"
    for p in policies
}

tabs = ["Submit Claim", "View Claims"]
selected_tab = st.sidebar.radio("Navigation", tabs)

# Submit Claim Tab
if selected_tab == "Submit Claim":
    st.header("Submit a New Claim")

    selected_policy_id = st.selectbox("Select Policy", options=list(policy_options.keys()), format_func=lambda x: policy_options[x])

    claim_type = st.selectbox("Claim Type", ["Accident", "Theft", "Fire"])
    claim_amount = st.number_input("Claim Amount (KES)", min_value=10000, step=1000)

    accident_date = st.date_input("Accident Date", min_value=datetime.date(2023, 1, 1), max_value=datetime.date.today())

    location = st.text_input("Accident Location", value="Nairobi")
    repair_shop = st.text_input("Repair Shop", value="QuickFix Garage")

    if st.button("Submit Claim"):
        # Generate Claim ID
        claim_id = f"CLM{str(len(st.session_state.claims)+1).zfill(4)}"

        # Claim Object
        claim_doc = {
            "claim_id": claim_id,
            "policy_id": selected_policy_id,
            "claim_type": claim_type,
            "claim_amount": claim_amount,
            "submitted_at": datetime.datetime.now(),
            "status": "under_review"
        }

        rule_score = compute_fraud_score_demo(claim_doc)

        features = extract_features_demo(claim_doc)
        ml_pred = ml_model.predict(features)[0]  # -1 = anomaly
        ml_score = 50 if ml_pred == -1 else 0

        fraud_score = min(rule_score + ml_score, 100)

        claim_doc.update({
            "rule_score": rule_score,
            "ml_score": ml_score,
            "fraud_score": fraud_score
        })

        # Save Claim
        st.session_state.claims.append(claim_doc)

        # Log Event
        st.session_state.events.append({
            "claim_id": claim_id,
            "event_type": "claim_submitted",
            "timestamp": claim_doc["submitted_at"],
            "metadata": {
                "location": location,
                "repair_shop": repair_shop,
                "claim_type": claim_type
            }
        })

        st.success(f"Claim {claim_id} submitted successfully | Fraud Score: {fraud_score}")

# View Claims Tab
elif selected_tab == "View Claims":
    st.header("Submitted Claims")

    if not st.session_state.claims:
        st.info("No claims submitted yet.")
    else:
        df = pd.DataFrame(st.session_state.claims)
        df["submitted_at"] = pd.to_datetime(df["submitted_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        def highlight_risk(row):
            if row["fraud_score"] >= 60:
                return ["background-color: #FFCCCC"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df[
                [
                    "claim_id",
                    "policy_id",
                    "claim_type",
                    "claim_amount",
                    "submitted_at",
                    "status",
                    "rule_score",
                    "ml_score",
                    "fraud_score",
                ]
            ].style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        # View Events
        st.subheader("Claim Events")

        selected_claim_id = st.selectbox("Select Claim ID", options=df["claim_id"])

        events = [
            e for e in st.session_state.events
            if e["claim_id"] == selected_claim_id
        ]

        if events:
            df_events = pd.DataFrame(events)
            df_events["timestamp"] = pd.to_datetime(df_events["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            st.table(df_events[["event_type", "timestamp", "metadata"]])
        else:
            st.info("No events found for this claim.")
