# A/B Testing and ML Experiment Dashboard

This project simulates, aggregates, analyzes and visualizes A/B testing data for ML-powered systems. It is designed to handle experiments with control and treatment variants, compute key metrics, apply statistical tests and guardrails, and also provide  **dashboards**.

---

## Core Concepts

### 1. A/B Testing
A/B testing is a controlled experiment comparing **two variants**:
- **Control**: The baseline version.
- **Treatment**: The new version being tested.

Metrics like **Click-Through Rate (CTR)**, **latency** and **conversion** are tracked for both groups to determine if the treatment improves outcomes without introducing negative side effects.

---

### 2. Users and Variant Assignment
Each experiment involves **simulated or real users**.  
- Every user is randomly assigned to **control or treatment** based on a predefined split.  
- The assignment is **unique per user** to ensure fairness.  
- Events are tracked per user in **three stages**:
  1. **Variant Assignment**: Records which variant the user was assigned to.
  2. **Model Inference**: Records the model version, prediction score, and latency.
  3. **User Response**: Records whether the user clicked (`clicked = 1`) or not (`clicked = 0`).

This separation allows us to simulate realistic ML-powered user interactions and compute metrics accurately.

---

### 3. Metrics Computation

#### 3.1 Click-Through Rate (CTR)
CTR measures user engagement:

\[
\text{CTR} = \frac{\text{clicks}}{\text{impressions}}
\]

- **Clicks**: Total number of users who clicked.  
- **Impressions**: Total number of users exposed to the variant (or total responses).


#### 3.2 Average Latency
Latency measures system performance:

\[
\text{Average Latency (ms)} = \frac{\text{Total Latency}}{\text{Number of Impressions}}
\]

- Used as a **guardrail** to prevent deploying a slower treatment version.

---

### 4. Incremental Aggregation
Instead of recalculating all metrics every time, the project supports **incremental aggregation**:

1. **New Users Detection**: Only users not previously seen are processed.
2. **Incremental Metrics**:
   - **New clicks** and **new impressions** from user responses.
   - **New latency** from inference events.
3. **Cumulative Metrics Update**: Previous metrics are combined with new ones for a **running total**:
   - Users
   - Clicks
   - Impressions
   - Latency sum
   - CTR
   - Average latency

4. **Run ID**: Each batch of new users is assigned a unique `run_id` for tracking purposes.

---

### 5. Statistical Decision Logic

#### 5.1 Minimum Detectable Lift (Guardrail)
- Define a **minimum practical lift**, e.g., 2% increase in CTR.
- Only consider deployments worth acting on if observed CTR exceeds baseline by this threshold.

#### 5.2 Power Analysis
To ensure statistically meaningful results, the dashboard calculates **required sample size per variant**:

\[
\text{Cohen's } h = 2 \left( \arcsin \sqrt{p_1} - \arcsin \sqrt{p_0} \right)
\]

Where:
- \(p_0\) = control CTR  
- \(p_1\) = expected treatment CTR  

The **statsmodels** library (`NormalIndPower`) computes the **minimum number of users** needed per variant to achieve desired power (typically 80%) at a significance level (\(\alpha = 0.05\)).

If the observed sample size is below this threshold, the decision is **marked PENDING (UNDERPOWERED)**.

#### 5.3 Latency Regression Guardrail
Deployments are blocked if treatment increases average latency by more than **25%** over control:

\[
\text{Latency Regression} = \text{latency}_{treatment} > 1.25 \times \text{latency}_{control}
\]

This prevents degrading user experience even if CTR improves.

#### 5.4 Final Decision
Each run is labeled according to its outcome and guardrail checks:

- **SHIP**:  
  The treatment meets all guardrails (power/sample size and latency), and the observed CTR lift is positive and above the minimum threshold. The feature or model is safe to deploy.

- **DO NOT SHIP (Check guardrails)**:  
  The treatment **underperformed** compared to the control (CTR lift is insufficient or negative). Even if sample size and latency guardrails are satisfied, the treatment should **not be deployed**. “Check guardrails” is a reminder to review latency and other safety constraints before making final operational decisions.

- **PENDING (UNDERPOWERED)**:  
  The sample size is insufficient to achieve the desired statistical power. The experiment may need more users before a reliable decision can be made.

- **PENDING (LATENCY REGRESSION)**:   
  The treatment violates latency guardrails (e.g., treatment is significantly slower than control). Latency indicates how fast the system responds to user actions. Even if the CTR lift is positive, the feature should not be deployed until latency issues are resolved.

---

### 6. Hypothesis Testing (Optional)
For rigorous statistical testing:
- A **one-sided Z-test for proportions** is applied:

\[
H_0: p_{treatment} \leq p_{control} \\
H_1: p_{treatment} > p_{control}
\]

Where \(p\) is the CTR.  
The **p-value** is compared with \(\alpha\) to assess statistical significance.

Confidence intervals (95%) for CTR lift are computed as:

\[
SE = \sqrt{\frac{p_{control}(1-p_{control})}{n_{control}} + \frac{p_{treatment}(1-p_{treatment})}{n_{treatment}}}
\]

\[
\text{CI} = \text{absolute lift} \pm z_{0.95} \cdot SE
\]

---

### 7. MLflow Integration
- Experiments and runs are logged in **MLflow** for traceability.  
- **Parameters logged**: alpha, min_lift, test type.  
- **Metrics logged**: CTRs, lift, p-value, average latency, number of users.  
- **Tags logged**: decision, experiment ID, guardrail violations. 

---

### 8. Dashboards

#### 8.1 Local CSV Dashboard (demo-app.py)
- Reads aggregated experiment metrics from `experiment_metrics.csv`.
- Computes missing metrics (CTR, average latency) if needed.
- Displays:
  - Latest run KPIs
  - Table of all runs
  - Decision and guardrail alerts
- Applies **power and latency guardrails** in the dashboard.

#### 8.2 MLflow Dashboard (app.py)
- Fetches runs and metrics directly from **MLflow**.
- Applies **power and latency guardrails** dynamically.
- Displays:
  - Latest valid run KPIs
  - CTR comparison chart
  - Absolute lift over time
  - Latency comparison
  - Decisions and alerts
  - Detailed run table

---

### 9. Key Libraries and Concepts Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation, aggregation, merging, cleaning |
| `numpy` | Numerical computations and probability calculations |
| `statsmodels` | Statistical power analysis and hypothesis testing |
| `mlflow` | Experiment tracking, logging parameters, metrics, and tags |
| `streamlit` | Interactive dashboard and visualization |
| `plotly.express` | Charts and graphs (CTR, lift, latency) |
| `uuid` | Unique ID generation for simulated events |
| `datetime` | Handling timestamps for events |
| `json` | Checkpointing last processed timestamp |
| `Pathlib` | File and directory management |

---

### 10. Guardrail Summary

| Guardrail | Condition | Action |
|-----------|-----------|--------|
| **Power** | Users < minimum required | Mark run `PENDING (UNDERPOWERED)` |
| **Latency Regression** | Avg latency treatment > 1.25 * control | Mark run `PENDING (LATENCY REGRESSION)` |
| **Minimum CTR Lift** | Absolute lift < threshold | Decision = `DO NOT SHIP` |
| **Data Completeness** | Missing CTR or impressions | Compute metrics or mark run `INCOMPLETE` |

---

### 11. Data Flow Overview

1. **Simulation / User Events** (`simulate_events.py`):
   - Generates variant assignments, inference events, and user responses.

2. **Incremental Aggregation** (`incremental_aggregate.py`):
   - Deduplicates users.
   - Updates cumulative metrics.
   - Saves metrics to CSV.

3. **Experiment Metrics Computation** (`compute_metrics.py`):
   - Aggregates clicks, impressions, CTR and latency per variant.

4. **Statistical Analysis** (`ab_test_analysis.py`):
   - Computes lift, confidence intervals, Z-test p-value.
   - Applies guardrails for decision.

5. **Dashboards**:
   - **CSV-based**: Loads metrics locally.
   - **MLflow-based**: Fetches runs and metrics dynamically.
   - Both dashboards visualize KPIs, charts, and apply guardrails.

---