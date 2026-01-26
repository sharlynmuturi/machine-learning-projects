import sqlite3
import random
from faker import Faker
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

fake = Faker()
random.seed(42)
np.random.seed(42)

DB_PATH = "data/saas_churn.db"
N_CUSTOMERS = 800
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Database Setup
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS usage_events;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS support_tickets;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    signup_date TEXT,
    company_size TEXT,
    industry TEXT,
    country TEXT
);

CREATE TABLE subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    plan_type TEXT,
    billing_cycle TEXT,
    start_date TEXT,
    status TEXT,
    churn_date TEXT
);

CREATE TABLE usage_events (
    event_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    event_type TEXT,
    timestamp TEXT
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    payment_date TEXT,
    status TEXT
);

CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    created_at TEXT,
    sentiment REAL
);
""")
conn.commit()


# Data Generation
industries = ["Fintech", "E-commerce", "Healthcare", "EdTech", "SaaS"]
company_sizes = ["Small", "Mid", "Enterprise"]
plans = ["Basic", "Pro", "Enterprise"]
billing_cycles = ["Monthly", "Annual"]
event_types = ["login", "feature_use", "export", "api_call"]

customer_ids = []

print("Generating customers...")
for cid in range(1, N_CUSTOMERS + 1):
    signup = fake.date_between(start_date=START_DATE, end_date=END_DATE - timedelta(days=90))
    cur.execute(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        (
            cid,
            signup.isoformat(),
            random.choice(company_sizes),
            random.choice(industries),
            fake.country()
        )
    )
    customer_ids.append(cid)

conn.commit()

print("Generating subscriptions...")
for cid in customer_ids:
    start = fake.date_between(start_date=START_DATE, end_date=END_DATE - timedelta(days=60))
    churned = random.random() < 0.25  # 25% churn
    churn_date = None

    if churned:
        churn_date = start + timedelta(days=random.randint(30, 300))

    cur.execute(
        "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            None,
            cid,
            random.choices(plans, weights=[0.5, 0.35, 0.15])[0],
            random.choices(billing_cycles, weights=[0.7, 0.3])[0],
            start.isoformat(),
            "churned" if churned else "active",
            churn_date.isoformat() if churn_date else None
        )
    )

conn.commit()

print("Generating usage events...")
for cid in tqdm(customer_ids):
    sub = pd.read_sql(
        f"SELECT start_date, churn_date FROM subscriptions WHERE customer_id={cid}",
        conn
    ).iloc[0]

    start = datetime.fromisoformat(sub["start_date"])
    end = datetime.fromisoformat(sub["churn_date"]) if sub["churn_date"] else END_DATE

    days_active = (end - start).days

    for d in range(days_active):
        date = start + timedelta(days=d)

        # Usage decay before churn
        base_events = random.randint(1, 5)
        if sub["churn_date"] and (end - date).days < 30:
            base_events = max(0, base_events - 2)

        for _ in range(base_events):
            cur.execute(
                "INSERT INTO usage_events VALUES (?, ?, ?, ?)",
                (
                    None,
                    cid,
                    random.choice(event_types),
                    date.isoformat()
                )
            )

conn.commit()

print("Generating payments...")
for cid in customer_ids:
    n_payments = random.randint(6, 18)

    for _ in range(n_payments):
        status = random.choices(
            ["success", "late", "failed"],
            weights=[0.85, 0.1, 0.05]
        )[0]

        amount = random.choice([29, 59, 99, 199])
        cur.execute(
            "INSERT INTO payments VALUES (?, ?, ?, ?, ?)",
            (
                None,
                cid,
                amount,
                fake.date_between(start_date=START_DATE, end_date=END_DATE).isoformat(),
                status
            )
        )

conn.commit()

print("Generating support tickets...")
for cid in customer_ids:
    if random.random() < 0.35:
        for _ in range(random.randint(1, 4)):
            sentiment = np.clip(np.random.normal(0, 0.6), -1, 1)
            cur.execute(
                "INSERT INTO support_tickets VALUES (?, ?, ?, ?)",
                (
                    None,
                    cid,
                    fake.date_between(start_date=START_DATE, end_date=END_DATE).isoformat(),
                    sentiment
                )
            )

conn.commit()
conn.close()

print("SaaS churn database generated successfully!")
