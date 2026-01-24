import datetime
from db.mongo import policies_col, vehicles_col, claims_col, events_col

def compute_fraud_score(claim):
    """
    Compute a simple rule-based fraud score (0-100)
    """
    score = 0

    # Amount vs estimated vehicle value
    policy = policies_col.find_one({"_id": claim["policy_id"]})
    vehicle = vehicles_col.find_one({"_id": policy["vehicle_id"]})
    if claim["claim_amount"] > 1.5 * vehicle["estimated_value"]:
        score += 40  # High claim relative to vehicle value

    # Time since policy start
    days_since_start = (claim["submitted_at"] - policy["start_date"]).days
    if days_since_start < 14:
        score += 30  # Early claim

    # Previous claims in last 30 days
    start_window = claim["submitted_at"] - datetime.timedelta(days=30)
    recent_claims = list(claims_col.find({
        "policy_id": claim["policy_id"],
        "submitted_at": {"$gte": start_window, "$lt": claim["submitted_at"]}
    }))
    if len(recent_claims) >= 2:
        score += 20

    # Repair shop risk
    last_event = events_col.find_one({"claim_id": claim["_id"], "event_type": "repair_estimate_added"})
    risky_shops = ["QuickFix Garage"]
    if last_event and last_event["metadata"]["repair_shop"] in risky_shops:
        score += 10

    return min(score, 100)
