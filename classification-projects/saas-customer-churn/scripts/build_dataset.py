import sqlite3
import pandas as pd
from datetime import datetime
from feature_engineering import get_usage_features, get_payment_features, get_support_features, get_churn_label

DB_PATH = "data/saas_churn.db"
CUTOFF_DATE = datetime(2024, 6, 30)

conn = sqlite3.connect(DB_PATH)

usage = get_usage_features(conn, CUTOFF_DATE)
payments = get_payment_features(conn, CUTOFF_DATE)
support = get_support_features(conn, CUTOFF_DATE)
labels = get_churn_label(conn, CUTOFF_DATE)

subs = pd.read_sql("""
    SELECT customer_id, plan_type, billing_cycle, start_date
    FROM subscriptions
""", conn)

subs["start_date"] = pd.to_datetime(subs["start_date"])
subs["tenure_days"] = (CUTOFF_DATE - subs["start_date"]).dt.days

df = (
    subs.merge(usage, on="customer_id", how="left")
        .merge(payments, on="customer_id", how="left")
        .merge(support, on="customer_id", how="left")
        .merge(labels, on="customer_id", how="left")
)

df.fillna(0, inplace=True)

df.to_csv("data/churn_features.csv", index=False)
conn.close()

print("Feature dataset built successfully!")
