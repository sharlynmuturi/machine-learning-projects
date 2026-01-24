# Vehicle Insurance Fraud Detection System

This project implements a vehicle insurance fraud detection system that evaluates insurance claims in real time using a combination of domain-driven rules and machine-learning–based anomaly detection.

The system combines:
- **Business rules** (what insurers already know)
- **Machine learning** (to detect unknown or subtle patterns)
- **Live data storage** (MongoDB)
- **A Streamlit dashboard**

Insurance fraud has three major challenges:

1. **Fraud is rare**  
   Most claims are legitimate - datasets are highly imbalanced.

2. **Fraud patterns change**  
   Fraudsters adapt quickly, so fixed rules alone are not enough.

3. **Decisions must be explainable**  
   Insurers must justify why a claim is flagged.

This project addresses all three.

Claims evolve over time and risk is reassessed as new information arrives.

Claims in the system are treated as:
- A **claim document** (amount, type, policy, date)
- A sequence of **events** (submission, repair estimate, updates)

---

## Rule-Based Fraud Detection (Human Logic)

Rule-based fraud detection uses **insurance domain knowledge**.

Examples:
- Claim amount unusually high compared to vehicle value
- Claim submitted very soon after policy activation
- Multiple claims in a short time window
- Known high-risk repair shops

Each rule contributes points to a **rule score**.

Rule-based scoring answers:
> *“Does this claim violate known fraud heuristics?”*

---

## Machine Learning Fraud Detection (Anomaly Detection)

The machine-learning component uses **Isolation Forest**, an anomaly detection model.

Instead of predicting “fraud vs non-fraud” directly, the model:
- Learns what **normal claims look like**
- Flags **unusual or rare claims** as anomalies

ML answers:
> *“Does this claim behave unusually compared to historical data?”*

---

## Hybrid Fraud Scoring 

The system combines both:

- **Rule Score** - known, explainable risks
- **ML Score** - statistical anomalies
- **Fraud Score (Hybrid)** - final risk indicator

Fraud Score = Rule Score + ML Score

---

## Fraud Scoring

When a claim is submitted:
1. The claim is stored in the database
2. Rule-based scoring runs immediately
3. The ML model evaluates the claim
4. A final fraud score is generated
5. High-risk claims are instantly flagged

---
