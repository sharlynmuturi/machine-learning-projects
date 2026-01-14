import streamlit as st
import pandas as pd
from src.data_features import fetch_stock_data, prepare_for_prophet, compute_historical_returns, compute_expected_returns
from src.prophet_model import forecast_stock
from src.portfolio import compute_covariance, optimize_portfolio

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Prophet-Based Portfolio Optimization")

tickers_input = st.text_input("Enter stock tickers (comma separated)", value="AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Forecast horizon input
forecast_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=90, value=30, help="How far ahead do you want price predictions?")

# Risk-free rate input
risk_free_rate = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.2, value=0.0, step=0.001, help="What safe interest rate should the portfolio optimizer consider? eg. government bonds rates.")

if st.button("Run Optimization"):
    with st.spinner("Fetching data..."):
        data = fetch_stock_data(tickers)

    with st.spinner("Preparing data..."):
        prepared = {t: prepare_for_prophet(data[t]) for t in tickers}
        historical_returns = compute_historical_returns(data)

    with st.spinner("Forecasting..."):
        forecasts = {t: forecast_stock(prepared[t], periods=forecast_days) for t in tickers}
        mu = compute_expected_returns(forecasts)

    cov_matrix = compute_covariance(historical_returns)

    with st.spinner("Optimizing portfolio..."):
        optimal_portfolio = optimize_portfolio(mu, cov_matrix, risk_free_rate=risk_free_rate)

    st.subheader("Optimized Portfolio Weights")
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": [optimal_portfolio["weights"][t] for t in tickers]
    })
    st.bar_chart(weights_df.set_index("Ticker"))

    st.write(f"Expected Annual Return: {optimal_portfolio['expected_return']:.2%}")
    st.write(f"Annual Volatility: {optimal_portfolio['volatility']:.2%}")
    st.write(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
