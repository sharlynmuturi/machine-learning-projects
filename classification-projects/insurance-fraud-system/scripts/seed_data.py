"""
Generates synthetic insurance data and inserts into MongoDB.
Collections - policyholders, vehicles, policies, claims, claim_events
Simulates Normal vs fraudulent claims and Collusion patterns (e.g., same repair shop in suspicious claims)
"""

import sys
from pathlib import Path

# Adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import random
import datetime
from faker import Faker
from db.mongo import policyholders_col, vehicles_col, policies_col, claims_col, events_col
from utils.fraud import compute_fraud_score

fake = Faker()
random.seed(42)

NUM_POLICYHOLDERS = 50
NUM_VEHICLES = 50
NUM_POLICIES = 50
NUM_CLAIMS = 100

FRAUD_PERCENTAGE = 0.1  # 10% of claims will be fraudulent


# Helper Functions
def random_date(start, end):
    """Return a random datetime between two datetime objects"""
    return start + datetime.timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()))
    )

def insert_policyholder(ph_id):
    ph = {
        "_id": ph_id,
        "name": fake.name(),
        "phone": fake.phone_number(),
        "email": fake.email(),
        "risk_profile": "normal",
        "created_at": datetime.datetime.now()
    }
    policyholders_col.insert_one(ph)
    return ph

def insert_vehicle(vh_id):
    makes = ["Toyota", "Honda", "Nissan", "Subaru", "Ford"]
    models = ["Axio", "Fit", "Leaf", "Impreza", "Focus"]
    year = random.randint(2010, 2024)
    vh = {
        "_id": vh_id,
        "registration": fake.bothify(text="K??###A"),
        "make": random.choice(makes),
        "model": random.choice(models),
        "year": year,
        "estimated_value": random.randint(500000, 2000000)
    }
    vehicles_col.insert_one(vh)
    return vh

def insert_policy(policy_id, ph, vh):
    start_date = random_date(datetime.datetime(2023, 1, 1), datetime.datetime(2025, 1, 1))
    coverage_types = ["Comprehensive", "Third Party", "Third Party Fire & Theft"]
    pol = {
        "_id": policy_id,
        "policyholder_id": ph["_id"],
        "vehicle_id": vh["_id"],
        "coverage_type": random.choice(coverage_types),
        "start_date": start_date,
        "status": "active",
        "premium": random.randint(20000, 50000)
    }
    policies_col.insert_one(pol)
    return pol

# Insert Claim + Events
def insert_claim(claim_id, policy, fraud=False):
    claim_amount = int(policy["premium"] * random.randint(10, 50))
    if fraud:
        claim_amount *= random.uniform(1.5, 3)

    claim = {
        "_id": claim_id,
        "policy_id": policy["_id"],
        "claim_amount": int(claim_amount),
        "claim_type": random.choice(["Accident", "Theft", "Fire"]),
        "submitted_at": random_date(policy["start_date"], datetime.datetime(2025, 1, 1)),
        "status": "under_review"
    }
    claims_col.insert_one(claim)

    # Create claim events
    events = []

    events.append({
        "claim_id": claim_id,
        "event_type": "claim_submitted",
        "timestamp": claim["submitted_at"],
        "metadata": {
            "location": fake.city(),
            "accident_type": claim["claim_type"]
        }
    })

    repair_shops = ["QuickFix Garage", "SuperRepair Ltd", "FastTrack Auto", "Trusted Repairs"]
    repair_shop = random.choice(repair_shops)
    if fraud:
        repair_shop = "QuickFix Garage"  # collusion pattern

    estimate_date = claim["submitted_at"] + datetime.timedelta(days=random.randint(1, 5))
    events.append({
        "claim_id": claim_id,
        "event_type": "repair_estimate_added",
        "timestamp": estimate_date,
        "metadata": {
            "repair_shop": repair_shop,
            "estimated_cost": claim["claim_amount"] * random.uniform(0.8, 1.1)
        }
    })

    events_col.insert_many(events)

    # Compute rule-based fraud score
    rule_score = compute_fraud_score(claim)
    ml_score = 0
    fraud_score = rule_score

    claims_col.update_one(
        {"_id": claim["_id"]},
        {"$set": {
            "rule_score": rule_score,
            "ml_score": ml_score,
            "fraud_score": fraud_score
        }}
    )

    return claim


# Main Loop
def generate_data():
    # Clear collections
    policyholders_col.delete_many({})
    vehicles_col.delete_many({})
    policies_col.delete_many({})
    claims_col.delete_many({})
    events_col.delete_many({})

    # Generate policyholders
    policyholders = [insert_policyholder(f"PH{str(i+1).zfill(3)}") for i in range(NUM_POLICYHOLDERS)]

    # Generate vehicles
    vehicles = [insert_vehicle(f"VH{str(i+1).zfill(3)}") for i in range(NUM_VEHICLES)]

    # Generate policies
    policies = []
    for i in range(NUM_POLICIES):
        ph = random.choice(policyholders)
        vh = random.choice(vehicles)
        policies.append(insert_policy(f"POL{str(i+1).zfill(3)}", ph, vh))

    # Generate claims
    for i in range(NUM_CLAIMS):
        policy = random.choice(policies)
        fraud_flag = random.random() < FRAUD_PERCENTAGE
        insert_claim(f"CLM{str(i+1).zfill(4)}", policy, fraud=fraud_flag)

    print("Data generation complete!")

if __name__ == "__main__":
    generate_data()
