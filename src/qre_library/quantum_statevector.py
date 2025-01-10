# src/qre_library/quantum_statevector.py

import numpy as np
from scipy.stats import norm

from qiskit import transpile
from qiskit_aer import StatevectorSimulator
from qiskit_finance.circuit.library import LogNormalDistribution

import numpy as np
from scipy.stats import norm

from qiskit import transpile
from qiskit_aer import StatevectorSimulator
from qiskit_finance.circuit.library import LogNormalDistribution


def compute_lognormal_parameters(S, r, sigma, T):
    """
    Compute lognormal parameters given Geometric Brownian Motion assumptions.

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.

    Returns
    -------
    mu : float
        ln-space mean parameter.
    sigma_sq : float
        ln-space variance parameter.
    mean : float
        Mean in the normal space.
    variance : float
        Variance in the normal space.
    stddev : float
        Standard deviation in the normal space.
    low : float
        Lower bound for the distribution.
    high : float
        Upper bound for the distribution.
    """
    mu = (r - 0.5*sigma**2)*T + np.log(S)
    sigma_sq = (sigma*np.sqrt(T))**2
    mean = np.exp(mu + sigma_sq / 2)
    variance = (np.exp(sigma_sq) - 1)*np.exp(2*mu + sigma_sq)
    stddev = np.sqrt(variance)

    low = max(0, mean - 3*stddev)
    high = mean + 3*stddev

    return mu, sigma_sq, mean, variance, stddev, low, high


def build_lognormal_distribution(num_qubits, mu, sigma_sq, low, high):
    """
    Build a LogNormalDistribution with specified parameters and qubit count.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used in the model.
    mu : float
        ln-space mean parameter.
    sigma_sq : float
        ln-space variance parameter.
    low : float
        Lower bound for the distribution.
    high : float
        Upper bound for the distribution.

    Returns
    -------
    LogNormalDistribution
        A Qiskit LogNormalDistribution object.
    """
    return LogNormalDistribution(
        num_qubits=num_qubits,
        mu=mu,
        sigma=sigma_sq,
        bounds=(low, high)
    )


def compute_black_scholes_call(S, K, r, sigma, T):
    """
    Compute the Black–Scholes price for a European call.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.

    Returns
    -------
    float
        The Black–Scholes call option price.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def compute_discrete_payoff(uncertainty_model, K):
    """
    Compute the payoff of a European call option discretely
    from the Qiskit distribution's built-in probabilities.

    Parameters
    ----------
    uncertainty_model : LogNormalDistribution
        The lognormal distribution circuit.
    K : float
        Strike price.

    Returns
    -------
    float
        Undiscounted expected payoff using the distribution's probabilities.
    """
    x_vals = uncertainty_model.values
    probs = uncertainty_model.probabilities
    payoff_vals = np.maximum(0, x_vals - K)
    discrete_value = np.dot(probs, payoff_vals)
    return discrete_value


def compute_statevector_payoff(uncertainty_model, K):
    """
    Run a statevector simulation and manually sum the payoff
    from the resulting amplitudes.

    Parameters
    ----------
    uncertainty_model : LogNormalDistribution
        The lognormal distribution circuit.
    K : float
        Strike price.

    Returns
    -------
    float
        The expected payoff (undiscounted) from the statevector amplitudes.
    """
    backend = StatevectorSimulator()
    circ = transpile(uncertainty_model, backend)
    result = backend.run(circ).result()
    statevec = result.get_statevector(circ)

    x_vals = uncertainty_model.values
    payoff_sum = 0.0
    
    for idx, amplitude in enumerate(statevec):
        prob = abs(amplitude)**2
        if prob < 1e-15:
            continue
        payoff_val = np.maximum(x_vals[idx] - K, 0)
        payoff_sum += payoff_val * prob

    return payoff_sum


def run_quantum_statevector(S=50, K=55, r=0.05, sigma=0.4, T=30/365, num_uncertainty_qubits=10):
    """
    Demonstrates quantum statevector simulation for pricing a European call
    with a LogNormal distribution, summing probabilities directly from the statevector.

    Returns a dict with the discrete payoff, statevector payoff,
    discounted payoffs, the Black–Scholes price, and the error.
    """

    # 1. Compute lognormal parameters
    mu, sigma_sq, mean, variance, stddev, low, high = compute_lognormal_parameters(S, r, sigma, T)

    # 2. Build lognormal distribution
    uncertainty_model = build_lognormal_distribution(
        num_qubits=num_uncertainty_qubits,
        mu=mu,
        sigma_sq=sigma_sq,
        low=low,
        high=high
    )

    # 3. Discrete payoff from the distribution’s internal probabilities
    discrete_value = compute_discrete_payoff(uncertainty_model, K)

    # 4. Black–Scholes price
    call_BS = compute_black_scholes_call(S, K, r, sigma, T)

    # 5. Statevector simulation payoff
    statevector_payoff = compute_statevector_payoff(uncertainty_model, K)

    # 6. Discount to present value
    discounted_discrete = discrete_value * np.exp(-r * T)
    discounted_statevector = statevector_payoff * np.exp(-r * T)

    # 7. Error vs. Black–Scholes
    error_vs_BS = abs(call_BS - discounted_statevector)

    return {
        "discrete_payoff": discrete_value,
        "statevector_payoff": statevector_payoff,
        "discounted_discrete_payoff": discounted_discrete,
        "discounted_statevector_payoff": discounted_statevector,
        "call_price_BS": call_BS,
        "error_vs_BS": error_vs_BS
    }

