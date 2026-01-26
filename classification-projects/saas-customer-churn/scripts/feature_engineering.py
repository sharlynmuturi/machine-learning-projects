import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression

DB_PATH = "data/saas_churn.db"


def get_usage_features(conn, cutoff_date):
    """
    No. of events in the last 7 and 30 days
    Trend of events over the last 30 days (slope)
    Days since last event

    """
    # Fetch all events up to the cutoff date
    query = f"""
    SELECT customer_id, DATE(timestamp) as event_date
    FROM usage_events
    WHERE DATE(timestamp) <= DATE('{cutoff_date}')
    """
    df = pd.read_sql(query, conn)
    df["event_date"] = pd.to_datetime(df["event_date"])

    features = []

    # Iterate over each customer to compute features
    for cid, g in df.groupby("customer_id"):
        last_event = g["event_date"].max()
        days_since_last = (cutoff_date - last_event).days

        # Filter events in last 30 and 7 days
        last_30 = g[g["event_date"] >= cutoff_date - timedelta(days=30)]
        last_7 = g[g["event_date"] >= cutoff_date - timedelta(days=7)]

        # Compute trend of events over last 30 days
        daily_counts = (
            last_30.groupby("event_date")
            .size()
            .reset_index(name="count")
        )

        slope = 0.0
        if len(daily_counts) > 3:  # Require at least 4 days for trend
            X = np.arange(len(daily_counts)).reshape(-1, 1)
            y = daily_counts["count"].values
            slope = LinearRegression().fit(X, y).coef_[0]

        features.append({
            "customer_id": cid,
            "events_7d": len(last_7),
            "events_30d": len(last_30),
            "usage_trend_30d": slope,
            "days_since_last_event": days_since_last
        })

    return pd.DataFrame(features)


def get_payment_features(conn, cutoff_date):
    """
    No. of late payments in the last 90 days
    No. of failed payments in the last 90 days

    """
    # Fetch payments in the 90 days prior to the cutoff date
    query = f"""
    SELECT customer_id, status, DATE(payment_date) as payment_date
    FROM payments
    WHERE DATE(payment_date) >= DATE('{cutoff_date}') - 90
      AND DATE(payment_date) <= DATE('{cutoff_date}')
    """
    df = pd.read_sql(query, conn)

    # Aggregate per customer
    return (
        df.groupby("customer_id")
        .agg(
            late_payments_90d=("status", lambda x: (x == "late").sum()),
            failed_payments_90d=("status", lambda x: (x == "failed").sum())
        )
        .reset_index()
    )


def get_support_features(conn, cutoff_date):
    """
    No. of support tickets in the last 90 days
    Avg sentiment of tickets in the last 90 days

    """
    query = f"""
    SELECT customer_id, sentiment, DATE(created_at) as created_at
    FROM support_tickets
    WHERE DATE(created_at) >= DATE('{cutoff_date}') - 90
      AND DATE(created_at) <= DATE('{cutoff_date}')
    """
    df = pd.read_sql(query, conn)

    # Aggregate metrics per customer
    return (
        df.groupby("customer_id")
        .agg(
            tickets_90d=("sentiment", "count"),
            avg_sentiment_90d=("sentiment", "mean")
        )
        .reset_index()
    )


def get_churn_label(conn, cutoff_date):
    """
    Churned (1) if their churn_date falls within 30 days after the cutoff date, otherwise 0.
    """
    query = f"""
    SELECT customer_id, churn_date
    FROM subscriptions
    """
    df = pd.read_sql(query, conn)
    df["churn_date"] = pd.to_datetime(df["churn_date"])

    # Label churn within 30 days after cutoff
    df["churn_30d"] = df["churn_date"].apply(
        lambda x: 1 if pd.notnull(x) and
        cutoff_date < x <= cutoff_date + timedelta(days=30)
        else 0
    )
    return df[["customer_id", "churn_30d"]]
