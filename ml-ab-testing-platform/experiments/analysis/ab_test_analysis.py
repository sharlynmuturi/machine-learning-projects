import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm


# Experiment Config
ALPHA = 0.05
MIN_LIFT = 0.01  # 1% practical significance threshold


# Load Metrics
df = pd.read_csv(
    "data/processed/experiment_metrics.csv"
)

control = df[df["variant"] == "control"].iloc[0]
treatment = df[df["variant"] == "treatment"].iloc[0]


# Extract Values
n_control = control["users"]
n_treatment = treatment["users"]

clicks_control = control["clicks"]
clicks_treatment = treatment["clicks"]

ctr_control = control["ctr"]
ctr_treatment = treatment["ctr"]


# Lift Calculation
absolute_lift = ctr_treatment - ctr_control
relative_lift = absolute_lift / ctr_control


# Hypothesis Test (Z-test)
stat, p_value = proportions_ztest(
    count=[clicks_treatment, clicks_control],
    nobs=[n_treatment, n_control],
    alternative="larger"  # one-sided test
)


# Confidence Interval (95%)
se = np.sqrt(
    (ctr_control * (1 - ctr_control) / n_control) +
    (ctr_treatment * (1 - ctr_treatment) / n_treatment)
)

z = norm.ppf(1 - ALPHA)
ci_lower = absolute_lift - z * se
ci_upper = absolute_lift + z * se


# Decision Logic
ship = (
    (p_value < ALPHA) and
    (absolute_lift >= MIN_LIFT)
)

decision = "SHIP 🚀" if ship else "DO NOT SHIP ❌"


# Results
print("\n===== A/B TEST RESULTS =====\n")
print(f"Control CTR   : {ctr_control:.4f}")
print(f"Treatment CTR : {ctr_treatment:.4f}")
print(f"Absolute Lift : {absolute_lift:.4%}")
print(f"Relative Lift : {relative_lift:.2%}")
print(f"P-value       : {p_value:.6f}")
print(f"95% CI Lift   : [{ci_lower:.4%}, {ci_upper:.4%}]")
print(f"\nDecision      : {decision}")
