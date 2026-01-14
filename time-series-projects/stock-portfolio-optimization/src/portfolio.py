import numpy as np
import pandas as pd

# Covariance / risk
def compute_covariance(returns):
    return returns.cov()


# Portfolio optimizer

def optimize_portfolio(expected_returns, cov_matrix, n_portfolios=5000, risk_free_rate=0.01):
    n_assets = len(expected_returns)
    best_sharpe = -np.inf
    best_weights = None

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    return {
        "weights": dict(zip(expected_returns.index, best_weights)),
        "expected_return": np.dot(best_weights, expected_returns),
        "volatility": np.sqrt(best_weights.T @ cov_matrix @ best_weights),
        "sharpe_ratio": best_sharpe
    }
