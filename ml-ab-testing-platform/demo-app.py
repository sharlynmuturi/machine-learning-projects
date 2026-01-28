import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
from statsmodels.stats.power import NormalIndPower


# Page setup
st.set_page_config(page_title="A/B Test Dashboard", layout="wide")
st.title("A/B Test Dashboard")


# Auto-refresh tab
refresh_interval = st.sidebar.slider("Auto Refresh Interval (seconds)", 0, 60, 10)
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh")


# Load experiment metrics
BASE_DIR = Path(__file__).parent

path = BASE_DIR / "data" / "processed" / "experiment_metrics.csv"


@st.cache_data
def load_data():
    return pd.read_csv(path)

df = load_data()
if df.empty:
    st.warning("No experiment data found. Please run the aggregation scripts.")
    st.stop()


# Select experiment
experiments = df["experiment_id"].unique()
selected_exp = st.selectbox("Select Experiment", experiments)
exp_df = df[df["experiment_id"] == selected_exp].copy()


# Ensure numeric metrics are correct
numeric_cols = ["users", "clicks", "latency_sum", "ctr", "avg_latency_ms"]
for col in numeric_cols:
    exp_df[col] = pd.to_numeric(exp_df[col], errors="coerce").fillna(0)

# Compute missing CTR if impressions exist
mask = (exp_df["ctr"] == 0) & (exp_df["impressions"] > 0)
exp_df.loc[mask, "ctr"] = exp_df.loc[mask, "clicks"] / exp_df.loc[mask, "impressions"]

# Compute missing avg latency per user
mask = (exp_df["avg_latency_ms"] == 0) & (exp_df["users"] > 0)
exp_df.loc[mask, "avg_latency_ms"] = exp_df.loc[mask, "latency_sum"] / exp_df.loc[mask, "users"]


# Assign run_id if missing
if "run_id" not in exp_df.columns:
    exp_df = exp_df.sort_values(["experiment_id", "impressions"])
    exp_df["run_id"] = exp_df.groupby("experiment_id").cumcount() // 2 + 1


# Run status logic
def run_status(run_df):
    variants = set(run_df["variant"])
    if variants == {"control", "treatment"}:
        if run_df["ctr"].notna().all():
            return "FINALIZED"
        return "RUNNING"
    return "INCOMPLETE"

status_df = exp_df.groupby("run_id")[["variant", "ctr"]].apply(run_status).reset_index(name="status")
exp_df = exp_df.merge(status_df, on="run_id", how="left")


# Decision logic with power and latency guardrails
ALPHA = 0.05        # significance level
POWER = 0.8         # desired power
MIN_LIFT = 0.02     # minimum detectable lift

def compute_decision(row, full_df):
    if row["variant"] != "treatment":
        return None

    # Get control row for the same run
    control = full_df[(full_df["run_id"] == row["run_id"]) & (full_df["variant"] == "control")]
    if control.empty or pd.isna(row["ctr"]) or pd.isna(control.iloc[0]["ctr"]):
        return None

    ctr_control = control.iloc[0]["ctr"]
    ctr_treatment_expected = ctr_control + MIN_LIFT

    # Clip probabilities to valid range
    ctr_control = np.clip(ctr_control, 0, 1)
    ctr_treatment_expected = np.clip(ctr_treatment_expected, 0, 1)

    # Compute effect size (Cohen's h)
    effect_size = 2 * (np.arcsin(np.sqrt(ctr_treatment_expected)) - np.arcsin(np.sqrt(ctr_control)))

    # Underpowered if effect size <= 0
    if effect_size <= 0:
        return "PENDING (underpowered)"

    # Power analysis: minimum sample size required
    analysis = NormalIndPower()
    min_n = analysis.solve_power(effect_size=effect_size, power=POWER, alpha=ALPHA, alternative='larger')

    # Check if actual sample size is sufficient
    n_actual = min(control.iloc[0]["impressions"], row["impressions"])
    if n_actual < min_n:
        return "PENDING (underpowered)"

    # Latency guardrail: treatment cannot be >35% slower than control & absolute minimum latency difference, ie trigger only if difference > 50ms:
    latency_control = control.iloc[0]["avg_latency_ms"]
    latency_treatment = row["avg_latency_ms"]
    if latency_treatment > latency_control * 1.35 and (latency_treatment - latency_control) > 50:
        return "PENDING (LATENCY REGRESSION)"

    # Decision based on observed CTR lift
    return "SHIP" if row["ctr"] > ctr_control else "DO NOT SHIP"


exp_df["decision"] = exp_df.apply(lambda r: compute_decision(r, exp_df), axis=1)


# Latest run summary metrics
latest_run_id = exp_df["run_id"].max()
latest_run = exp_df[exp_df["run_id"] == latest_run_id]

st.subheader("Latest Run Summary")
cols = st.columns(4)
control_latest = latest_run[latest_run["variant"] == "control"]
treatment_latest = latest_run[latest_run["variant"] == "treatment"]

if not control_latest.empty and not treatment_latest.empty:
    c = control_latest.iloc[0]
    t = treatment_latest.iloc[0]
    absolute_lift = t["ctr"] - c["ctr"]
    relative_lift = absolute_lift / c["ctr"]

    cols[0].metric("Control CTR", f"{c['ctr']*100:.2f}%")
    cols[1].metric("Treatment CTR", f"{t['ctr']*100:.2f}%")
    cols[2].metric("Absolute Lift", f"{absolute_lift*100:.2f}%", delta=f"{relative_lift*100:.2f}%")
    cols[3].metric("Decision", t["decision"] or "PENDING")
else:
    cols[0].metric("Run Status", latest_run["status"].iloc[0])
    st.warning("Latest run is not finalized yet.")


# Show all experiment runs
st.subheader("All Experiment Runs")
st.dataframe(
    exp_df[["run_id","status","variant","users","impressions","clicks","ctr","avg_latency_ms","decision"]]
    .rename(columns={
        "run_id": "Run ID",
        "status": "Run Status",
        "avg_latency_ms": "Avg Latency (ms)",
        "ctr": "CTR",
        "decision": "Decision"
    })
)


# Decisions & alerts per run
st.subheader("Decisions & Alerts")
for run_id, run_data in exp_df.groupby("run_id"):
    if run_data.empty:
        continue

    treatment = run_data[run_data["variant"] == "treatment"]
    if treatment.empty:
        continue

    decision = treatment.iloc[0]["decision"]

    # Display messages based on the exact reason
    if pd.isna(decision):
        st.info(f"Run {run_id}: Decision pending (incomplete data)")
    elif decision == "DO NOT SHIP":
        st.error(f"Run {run_id}: DO NOT SHIP (Check guardrails)")
    elif decision == "SHIP":
        st.success(f"Run {run_id}: SHIP")
    elif "PENDING" in decision:
        if "underpowered" in decision.upper():
            st.warning(f"Run {run_id}: {decision} – Increase sample size")
        elif "LATENCY" in decision.upper():
            st.warning(f"Run {run_id}: {decision} – Cannot ship due to latency regression")
        else:
            st.warning(f"Run {run_id}: {decision}")
    else:
        st.info(f"Run {run_id}: {decision}")

