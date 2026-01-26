# SaaS Churn Intelligence Platform

An **end-to-end, interactive SaaS churn prediction and retention intelligence platform**.  
Built using **event-level behavioral data**, **LightGBM**, **SHAP explainability**, **survival analysis** and deployed as a **Streamlit app**.  

This project demonstrates SaaS analytics, predictive modeling and actionable retention insights.

---

## Project Overview

Customer churn is one of the **most critical metrics for SaaS businesses**.  
This project builds a realistic, end-to-end churn prediction platform with:

1. **Event-level synthetic SaaS data** mimicking real product usage.  
2. **Feature engineering** from raw usage, payments and support tickets.  
3. **Predictive modeling** using LightGBM for high-quality churn predictions.  
4. **Explainable AI** via SHAP for individual and global feature importance.  
5. **Survival analysis** to estimate *time-to-churn*.  
6. **Interactive Streamlit app** for exploration, visualization and “what-if” scenarios.  

---

## Business Problem

SaaS companies want to **retain high-value customers**.  
Churn occurs when a customer cancels or becomes inactive.  

**Goals:**

- Predict **which customers are likely to churn** in the next 30 days.  
- Explain **why they are at risk**.  
- Estimate **when churn is likely to happen** (time-to-churn).  
- Provide **actionable insights** for retention campaigns.  

---

## Data Architecture

The project simulates **event-level SaaS data** stored in a **SQLite database**.  

### Tables:

1. **customers** – company info, signup date, industry, size, country  
2. **subscriptions** – plan type, billing cycle, start date, churn status/date  
3. **usage_events** – login, feature use, export, API calls with timestamps  
4. **payments** – amount, date, status (success, late, failed)  
5. **support_tickets** – ticket creation, sentiment scores  

This mirrors real-world SaaS analytics pipelines (like Stripe + Mixpanel + Zendesk).

---

## Synthetic Data Generation

To simulate realistic SaaS behavior:

- **Faker** generates company data.  
- **Randomized patterns** mimic churn behavior:  
  - High engagement → low churn risk  
  - Usage decay → precedes churn  
  - Late payments → higher churn probability  
  - Support tickets with negative sentiment → increases risk  

- Data is **time-stamped**, enabling temporal analysis.  

---

## Feature Engineering

Features are **aggregated from raw event-level data** over observation windows.  

**Observation Window:** The period we compute features.  
**Prediction Window:** The period we predict churn (e.g., next 30 days).  

### Feature Examples:

| Category | Feature | Description |
|----------|---------|-------------|
| Usage | `events_7d` | Number of events in last 7 days |
| Usage | `events_30d` | Number of events in last 30 days |
| Usage | `usage_trend_30d` | Slope of daily usage over 30 days |
| Usage | `days_since_last_event` | Days since last activity |
| Payments | `late_payments_90d` | Late payments in last 90 days |
| Payments | `failed_payments_90d` | Failed payments in last 90 days |
| Support | `tickets_90d` | Number of support tickets in last 90 days |
| Support | `avg_sentiment_90d` | Average sentiment of tickets (-1 negative, +1 positive) |
| Profile | `tenure_days` | Days since signup |
| Profile | `plan_type`, `billing_cycle` | Categorical variables |

These **dynamic, behavioral features** capture meaningful churn signals.

---

## Churn Label Logic

**Definition of churn:**

- Customer **canceled subscription** OR  
- Customer **inactive for 30 consecutive days**

**Label assignment:**

- `1` → churn occurs within the next 30 days (prediction window)  
- `0` → still active after prediction window  

This **prevents future leakage**, mimicking real prediction scenarios in SaaS.

---

## Machine Learning Models

### LightGBM Classifier

- Gradient boosting classifier for tabular data.  
- Handles **non-linearity** and **class imbalance**.  
- Evaluated using **ROC-AUC** and **precision/recall metrics**.  
- Produces **per-customer churn probabilities**.  

**Risk bands:**

- High (≥70%)  
- Medium (40–70%)  
- Low (<40%)  

These help prioritize retention efforts.

---

## Explainable AI (SHAP)

- **SHAP (SHapley Additive exPlanations)** identifies **why a customer is at risk**.  
- Provides **global feature importance** (most predictive features overall).  
- Provides **local explanations** (per customer):  

Example:

| Feature | SHAP Value | Interpretation |
|---------|------------|----------------|
| `events_30d` | -0.25 | High usage reduces risk |
| `days_since_last_event` | +0.40 | Long inactivity increases risk |
| `late_payments_90d` | +0.15 | Late payments increase risk |

These explanations are critical for **business buy-in**.

---

## Survival Analysis (Time-to-Churn)

- **Cox Proportional Hazards model** predicts the **hazard (risk) of churn over time**.  
- Allows estimation of **expected remaining lifetime** of a customer.  
- Helps prioritize interventions based on **urgency**, not just probability.  

**Output example:**

- Customer survival curve shows probability of **remaining active over the next 180 days**. 

---


## Streamlit Dashboard

The interactive dashboard has four main pages:

### 1. Overview
- Total customers, average churn risk, high-risk count  
- Risk distribution bar chart  

### 2. Customer Explorer
- Select a customer to view:  
  - Churn probability  
  - Risk band  
  - **SHAP waterfall plot**  
  - **Survival curve** (time-to-churn)

### 3. Risk Segmentation
- Filter customers by risk band (High, Medium, Low)  
- Sort and export for retention campaigns

---

