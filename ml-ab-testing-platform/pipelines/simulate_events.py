# Simulates event logs emitted by an ML-powered system running an A/B test.
# Each run represents new users arriving into the system.

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

np.random.seed(42)


# Experiment config
N_USERS = 10_000                    # Number of new users to simulate per run
EXPERIMENT_ID = "exp_model_ab_v1"   # Identifier for the experiment
CHECKPOINT = Path("data/checkpoints/last_ts.json")  # Stores timestamp of last simulation

# Determine start time for new events based on previous run
if CHECKPOINT.exists():
    last_ts = pd.to_datetime(json.load(open(CHECKPOINT))["last_ts"])
    START_TIME = last_ts + timedelta(seconds=5)  # start slightly after last event
else:
    START_TIME = datetime.now()

# Distribution of users into A/B variants
VARIANT_SPLIT = {
    "control": 0.5,
    "treatment": 0.5
}

# Model behavior per variant
MODEL_CONFIG = {
    "control": {
        "model_version": "logreg_v1",
        "ctr": 0.08,          # probability of user clicking
        "latency_mean": 35    # average model response time in ms
    },
    "treatment": {
        "model_version": "gboost_v2",
        "ctr": 0.10,
        "latency_mean": 45
    }
}

# Output paths
RAW_OUT = Path("data/raw/event_logs.csv")          # where simulated events are saved
RUN_COUNTER = Path("data/checkpoints/sim_run_id.txt")  # tracks simulation run number
RUN_COUNTER.parent.mkdir(exist_ok=True)           # ensure folder exists


# Determine run id (new users per run)
if RUN_COUNTER.exists():
    run_id = int(RUN_COUNTER.read_text())  # load previous run id
else:
    run_id = 0  # first run

RUN_COUNTER.write_text(str(run_id + 1))  # increment for next run


# Helper functions
def generate_timestamp(base_time, max_minutes=60):
    """Generate random timestamp within max_minutes of base_time"""
    return base_time + timedelta(
        minutes=np.random.randint(0, max_minutes)
    )

def assign_variant():
    """Randomly assign user to a variant according to VARIANT_SPLIT"""
    return np.random.choice(
        list(VARIANT_SPLIT.keys()),
        p=list(VARIANT_SPLIT.values())
    )


# Simulate NEW users
users = [f"user_{run_id}_{i}" for i in range(N_USERS)]  # unique user ids per run
records = []  # will store all event logs

for user_id in users:
    variant = assign_variant()
    model_info = MODEL_CONFIG[variant]

    # --- Variant assignment event (once per user) ---
    records.append({
        "event_id": str(uuid.uuid4()),  # unique event identifier
        "event_type": "variant_assignment",
        "timestamp": generate_timestamp(START_TIME),
        "user_id": user_id,
        "experiment_id": EXPERIMENT_ID,
        "variant": variant,
        "model_version": model_info["model_version"],
        "prediction_score": None,
        "latency_ms": None,
        "clicked": None
    })

    # --- Model inference event (simulated model serving a prediction) ---
    prediction_score = np.clip(np.random.normal(0.5, 0.15), 0, 1)  # simulated prediction
    latency = max(5, int(np.random.normal(model_info["latency_mean"], 8)))  # simulate latency

    records.append({
        "event_id": str(uuid.uuid4()),
        "event_type": "model_inference",
        "timestamp": generate_timestamp(START_TIME),
        "user_id": user_id,
        "experiment_id": EXPERIMENT_ID,
        "variant": variant,
        "model_version": model_info["model_version"],
        "prediction_score": prediction_score,
        "latency_ms": latency,
        "clicked": None
    })

    # --- User response event (click/no click) ---
    clicked = np.random.binomial(1, model_info["ctr"])  # simulate click based on CTR probability

    records.append({
        "event_id": str(uuid.uuid4()),
        "event_type": "user_response",
        "timestamp": generate_timestamp(START_TIME),
        "user_id": user_id,
        "experiment_id": EXPERIMENT_ID,
        "variant": variant,
        "model_version": model_info["model_version"],
        "prediction_score": None,
        "latency_ms": None,
        "clicked": clicked
    })


# Persist events
events_df = pd.DataFrame(records)

if RAW_OUT.exists():
    events_df.to_csv(RAW_OUT, mode="a", header=False, index=False)  # append to existing file
else:
    events_df.to_csv(RAW_OUT, index=False)  # first run, write new file

print(f"Simulation run {run_id} complete — {len(users)} NEW users added")
print(events_df.head())
