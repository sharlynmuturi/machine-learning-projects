# Prophet-Based Portfolio Optimization Dashboard

This project is an **end-to-end quantitative portfolio optimization system** that combines **time-series forecasting (Facebook Prophet)** with **Modern Portfolio Theory (MPT)** and exposes the results through an **interactive Streamlit dashboard**.

An investor wants to answer three key questions:

1. **What return do I expect from each stock?**
2. **How risky are these stocks together?**
3. **How should I allocate my capital to balance risk and return optimally?**

This project answers those questions by:

* Forecasting **expected returns** using Prophet
* Estimating **risk** using historical volatility and covariance
* Optimizing portfolio weights using the **Sharpe ratio**

## End-to-End Workflow

1. User inputs stock tickers (e.g. AAPL, MSFT, GOOG)
2. Historical price data is fetched from Yahoo Finance
3. Prices are transformed for time-series forecasting
4. Prophet forecasts future prices
5. Expected returns are derived from forecasts
6. Risk is estimated from historical returns
7. Portfolio weights are optimized
8. Results are visualized in a dashboard


## Feature Engineering

### Historical Returns

Historical daily returns are computed as:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$


These returns are used **only for risk estimation**, not forecasting.

### Prophet-Ready Data

Prophet requires a specific format:

| Column | Meaning                 |
| ------ | ----------------------- |
| ds     | Date                    |
| y      | Target variable (price) |

Prices (not returns) are forecasted because:

* Prophet models **trend + seasonality** better in levels
* Returns are noisy and harder to forecast


## Time-Series Forecasting (Prophet)

### Why Prophet?

* Handles trends and seasonality
* Robust to missing data
* Widely used in production

### Model Output

Prophet produces:

* `yhat` → expected price
* `yhat_lower` → lower confidence bound
* `yhat_upper` → upper confidence bound

We use the **mean forecast price** over a horizon of *N* days to estimate expected return.

## Expected Returns (Forecast-Based)

Expected return for stock *i* is computed as:

$$
\mu_i = \mathbb{E}\left[\frac{P_{future} - P_{last}}{P_{last}}\right]
$$

Where:

* $P_{last}$ = last observed price
* $P_{future}$ = mean forecasted price

This approach reflects **forward-looking expectations**, not historical averages.

## Risk Modeling (Historical)

### Covariance Matrix

Risk is captured via the **covariance matrix** of returns:

$$
\Sigma = \text{Cov}(R)
$$

Where:

* Diagonal elements → individual asset variance
* Off-diagonal elements → co-movement between assets

Risk is **not forecasted** because:

* Volatility forecasts are unstable
* Historical covariance is more robust

This mirrors real-world quantitative practice.

## Portfolio Optimization

### Objective: Maximize Sharpe Ratio

The Sharpe ratio measures **risk-adjusted return**:

$$
\text{Sharpe}(w) = \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}
$$

Where:

* (w) = portfolio weights
* (\mu) = expected returns
* (\Sigma) = covariance matrix
* (R_f) = risk-free rate


### Why Sharpe Ratio?

It answers:

> *How much extra return am I earning per unit of risk compared to a safe investment?*

## Risk-Free Rate

The **risk-free rate** represents the return of a zero-risk investment, typically:

* Government bonds
* Treasury bills

It defines the **minimum acceptable return** for taking risk.

If a portfolio cannot beat the risk-free rate, it is not worth investing in.


## Dashboard Outputs

### Optimized Portfolio Weights

* Visualized using bar charts
* Shows capital allocation per asset

### Portfolio Metrics

* Expected annual return
* Annualized volatility
* Sharpe ratio

### Forecast Summary Table

| Ticker | Last Price | Expected Price | Predicted Return |
| ------ | ---------- | -------------- | ---------------- |

This bridges **model output** with **investor intuition**.


## Key Design Decisions (And Why)

| Decision                     | Reason                            |
| ---------------------------- | --------------------------------- |
| Forecast prices, not returns | More stable time series           |
| Use forecast mean            | Reduces noise                     |
| Historical covariance        | Avoids volatility underestimation |
| Sharpe optimization          | Industry standard                 |
| No transaction costs         | Simplicity & clarity              |

## Limitations

* No transaction costs
* No short-selling constraints
* Prophet assumes additive structure
* Yahoo Finance data may contain noise

These are deliberate trade-offs for clarity.

## Future Extensions

* Efficient frontier visualization
* Confidence interval risk bands
* Monte Carlo simulations
* Portfolio rebalancing
* Real-time interest rate integration



## Final Takeaway

This project demonstrates how **forecasting, statistics, and optimization** come together to support real-world investment decisions.
