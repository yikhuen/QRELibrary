# src/qre_library/classical_monte_carlo.py

import math
import numpy as np
from scipy.stats import norm

def run_classical_monte_carlo(S0=50, K=55, r=0.05, sigma=0.4, T=30/365, t=30, M=1000, seed=135):
    """
    Runs a classical Monte Carlo simulation for a European call option payoff.

    Parameters
    ----------
    S0 : float
        Initial price of the underlying asset.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.
    t : int
        Number of discrete time steps in the simulation.
    M : int
        Number of Monte Carlo simulation paths.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        {
            "mc_estimate": float,
            "confidence_interval": (float, float),
            "black_scholes": float,
            "error": float,
            "S_paths": np.ndarray   # shape: (t+1, M)
        }
    """

    np.random.seed(seed)

    dt = T / t
    S = np.zeros((t + 1, M))
    S[0, :] = S0

    # Brownian increments
    Z = np.random.standard_normal((t, M))

    # Simulate the price paths
    for i in range(1, t + 1):
        S[i, :] = S[i-1, :] * np.exp((r - 0.5 * sigma**2)*dt + sigma * math.sqrt(dt) * Z[i-1, :])

    # Monte Carlo payoff
    payoffs = np.maximum(S[-1] - K, 0)
    P_call = np.exp(-r * T) * np.mean(payoffs)

    # Confidence interval
    std_error = np.std(np.exp(-r * T) * payoffs) / np.sqrt(M)
    conf_interval = (P_call - 1.96 * std_error, P_call + 1.96 * std_error)

    # Black-Scholes formula
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P_call_exact = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Estimation error
    error = abs(P_call - P_call_exact)

    return {
        "mc_estimate": P_call,
        "confidence_interval": conf_interval,
        "black_scholes": P_call_exact,
        "error": error,
        "S_paths": S
    }
