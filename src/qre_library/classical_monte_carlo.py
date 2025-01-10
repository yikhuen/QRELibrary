# src/qre_library/classical_monte_carlo.py

import math
import numpy as np
from scipy.stats import norm

import math
import numpy as np
from scipy.stats import norm

def simulate_stock_paths(S0, r, sigma, T, t, M, seed):
    """
    Simulate stock price paths using geometric Brownian motion.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity in years.
    t : int
        Number of time steps.
    M : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    S : np.ndarray
        Simulated stock price paths of shape (t+1, M).
    """
    np.random.seed(seed)
    
    dt = T / t
    S = np.zeros((t + 1, M))
    S[0, :] = S0

    # Generate random normal increments
    Z = np.random.standard_normal((t, M))

    # Evolve each path
    for i in range(1, t + 1):
        S[i, :] = S[i-1, :] * np.exp((r - 0.5 * sigma**2) * dt 
                                     + sigma * math.sqrt(dt) * Z[i-1, :])
    return S


def compute_monte_carlo_payoff(S, K, r, T):
    """
    Compute the discounted call option payoff from simulated paths.

    Parameters
    ----------
    S : np.ndarray
        Simulated stock price paths of shape (t+1, M).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity in years.

    Returns
    -------
    mc_price : float
        Monte Carlo estimate of the call option price.
    payoff_samples : np.ndarray
        Array of individual payoff samples for each path.
    """
    # Terminal payoffs
    payoffs = np.maximum(S[-1, :] - K, 0)
    # Discounted mean payoff
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    return mc_price, payoffs


def compute_confidence_interval(payoff_samples, mc_price, r, T):
    """
    Compute the 95% confidence interval for the Monte Carlo estimate.

    Parameters
    ----------
    payoff_samples : np.ndarray
        Terminal payoffs for each path.
    mc_price : float
        Monte Carlo mean estimate.
    r : float
        Risk-free rate.
    T : float
        Time to maturity in years.

    Returns
    -------
    (lower_bound, upper_bound) : tuple of floats
        95% confidence interval for the Monte Carlo estimate.
    """
    M = len(payoff_samples)
    discounted_payoffs = np.exp(-r * T) * payoff_samples
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(M)
    ci_lower = mc_price - 1.96 * std_error
    ci_upper = mc_price + 1.96 * std_error
    return (ci_lower, ci_upper)


def black_scholes_call_price(S0, K, r, sigma, T):
    """
    Compute the Black-Scholes price for a European call option.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        The Black–Scholes call option price.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def run_classical_monte_carlo(
    S0=50, K=55, r=0.05, sigma=0.4, T=30/365, t=30, M=1000, seed=135
):
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
    # 1. Simulate paths
    S_paths = simulate_stock_paths(S0, r, sigma, T, t, M, seed)

    # 2. Monte Carlo price
    mc_estimate, payoffs = compute_monte_carlo_payoff(S_paths, K, r, T)

    # 3. Confidence interval
    conf_interval = compute_confidence_interval(payoffs, mc_estimate, r, T)

    # 4. Black–Scholes
    bs_price = black_scholes_call_price(S0, K, r, sigma, T)

    # 5. Error
    error = abs(mc_estimate - bs_price)

    return {
        "mc_estimate": mc_estimate,
        "confidence_interval": conf_interval,
        "black_scholes": bs_price,
        "error": error,
        "S_paths": S_paths
    }

