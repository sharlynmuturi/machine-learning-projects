import pandas as pd
import json
from pathlib import Path
from datetime import datetime


# File paths
RAW_EVENTS = "data/raw/event_logs.csv"                 # raw event logs
METRICS_OUT = "data/processed/experiment_metrics.csv" # aggregated experiment metrics
CHECKPOINT = "data/checkpoints/last_ts.json"         # last processed timestamp
SEEN_USERS = "data/checkpoints/seen_users.csv"       # users already counted

# Ensure directories exist
Path("data/processed").mkdir(exist_ok=True)
Path("data/checkpoints").mkdir(exist_ok=True)


# Load timestamp checkpoint
if Path(CHECKPOINT).exists():
    last_ts = pd.to_datetime(json.load(open(CHECKPOINT))["last_ts"])
else:
    last_ts = None  # first run


# Load events and filter new ones
df = pd.read_csv(RAW_EVENTS)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])  # drop invalid timestamps

if last_ts is not None:
    df = df[df["timestamp"] > last_ts]  # keep only new events

if df.empty:
    print("No new data")
    exit()  # nothing to aggregate


# Load previously seen users for deduplication
if Path(SEEN_USERS).exists():
    seen_users = pd.read_csv(SEEN_USERS)
else:
    seen_users = pd.DataFrame(columns=["experiment_id", "variant", "user_id"])


# Identify NEW users only
assignments = df[df["event_type"] == "variant_assignment"][["experiment_id", "variant", "user_id"]].drop_duplicates()

# Merge with seen users, only keep users not seen before
new_users = assignments.merge(
    seen_users,
    on=["experiment_id", "variant", "user_id"],
    how="left",
    indicator=True
)
new_users = new_users[new_users["_merge"] == "left_only"][["experiment_id", "variant", "user_id"]]

# Update global seen users file
updated_seen = pd.concat([seen_users, new_users], ignore_index=True)
updated_seen.to_csv(SEEN_USERS, index=False)

# Count new users per variant
new_user_counts = new_users.groupby(["experiment_id", "variant"]).size().reset_index(name="new_users")


# Incremental clicks & impressions
responses = df[df["event_type"] == "user_response"]
clicks = (
    responses.groupby(["experiment_id", "variant"])
    .agg(
        new_impressions=("user_id", "count"),  # total responses/events
        new_clicks=("clicked", "sum")          # sum of clicks
    )
    .reset_index()
)


# Incremental latency
latency = (
    df[df["event_type"] == "model_inference"]
    .groupby(["experiment_id", "variant"])
    .agg(new_latency_sum=("latency_ms", "sum"))
    .reset_index()
)


# Merge incremental metrics
agg = (
    new_user_counts
    .merge(clicks, on=["experiment_id", "variant"], how="outer")
    .merge(latency, on=["experiment_id", "variant"], how="outer")
    .fillna(0)
)


# Handle existing cumulative metrics
if Path(METRICS_OUT).exists():
    existing = pd.read_csv(METRICS_OUT)
    if "impressions" not in existing.columns:
        existing["impressions"] = 0  # backward compatibility

    # Summarize previous totals
    cumulative = (
        existing.groupby(["experiment_id", "variant"], as_index=False)
        .agg(
            users=("users", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            latency_sum=("latency_sum", "sum")
        )
    )

    # Merge previous totals with new increments
    agg = agg.merge(cumulative, on=["experiment_id", "variant"], how="left").fillna(0)

    # Add new increments to cumulative totals
    agg["users"] += cumulative["users"].values
    agg["impressions"] += cumulative["impressions"].values
    agg["clicks"] += cumulative["clicks"].values
    agg["latency_sum"] += cumulative["latency_sum"].values

else:
    # First run — only new users
    agg["impressions"] = agg.get("new_impressions", 0)
    agg["users"] = agg["new_users"] + agg.get("users", 0)
    agg["clicks"] = agg["new_clicks"]
    agg["latency_sum"] = agg["new_latency_sum"]


# Derived metrics
agg["ctr"] = agg["clicks"] / agg["impressions"].clip(lower=1)  # click-through rate
agg["avg_latency_ms"] = agg["latency_sum"] / agg["impressions"].clip(lower=1)
agg["run_id"] = datetime.now().strftime("%Y%m%d%H%M%S")       # timestamp for this run


# Persist cumulative metrics
cols = ["experiment_id", "variant", "users", "impressions", "clicks", "ctr", "latency_sum", "avg_latency_ms", "run_id"]

if Path(METRICS_OUT).exists():
    final = pd.concat([existing, agg[cols]], ignore_index=True)
else:
    final = agg[cols]

final.to_csv(METRICS_OUT, index=False)


# Update checkpoint
json.dump({"last_ts": df["timestamp"].max().isoformat()}, open(CHECKPOINT, "w"))

print("Incremental aggregation complete")
print(agg)
