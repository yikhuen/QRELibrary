# src/qre_library/quantum_qae.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_finance.circuit.library import LogNormalDistribution


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_finance.circuit.library import LogNormalDistribution


def compute_lognormal_parameters(S, r, sigma, T):
    """
    Compute the parameters for a lognormal distribution based on
    the geometric Brownian motion model.

    Parameters
    ----------
    S : float
        Initial stock price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.

    Returns
    -------
    mu : float
        Mean parameter for the lognormal distribution (in ln-space).
    sigma_sq : float
        Variance parameter for the lognormal distribution (in ln-space).
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
    mu = (r - 0.5 * sigma**2) * T + np.log(S)
    sigma_sq = (sigma * np.sqrt(T))**2
    mean = np.exp(mu + sigma_sq / 2)
    variance = (np.exp(sigma_sq) - 1) * np.exp(2*mu + sigma_sq)
    stddev = np.sqrt(variance)

    # Choose bounds for the distribution
    low = max(0, mean - 2*stddev)
    high = mean + 2*stddev

    return mu, sigma_sq, mean, variance, stddev, low, high


def build_log_normal_distribution(num_qubits, mu, sigma_sq, low, high):
    """
    Builds a Qiskit LogNormalDistribution circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used for the uncertainty model.
    mu : float
        Mean parameter in ln-space.
    sigma_sq : float
        Variance parameter in ln-space.
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.

    Returns
    -------
    LogNormalDistribution
        The Qiskit LogNormalDistribution object.
    """
    return LogNormalDistribution(
        num_qubits=num_qubits,
        mu=mu,
        sigma=sigma_sq,
        bounds=(low, high)
    )


def compute_black_scholes_call(S, K, r, sigma, T):
    """
    Compute Black–Scholes call option price.

    Parameters
    ----------
    S : float
        Underlying spot price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity (in years).

    Returns
    -------
    float
        Black–Scholes call option price.
    """
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def run_qae_estimation(european_call_pricing, shots, seed):
    """
    Build and run the Iterative Amplitude Estimation (QAE) problem.

    Parameters
    ----------
    european_call_pricing : EuropeanCallPricing
        The pricing application from Qiskit-Finance.
    shots : int
        Number of shots for the sampler backend.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : AmplitudeEstimationResult
        QAE estimation result from Qiskit's IterativeAmplitudeEstimation.
    """
    # Convert to EstimationProblem
    problem = european_call_pricing.to_estimation_problem()

    # Sampler
    sampler = Sampler(options={"shots": shots, "seed": seed})

    # Iterative Amplitude Estimation
    ae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=sampler)
    result = ae.estimate(problem)

    return result


def run_quantum_qae(S=50, K=55, r=0.05, sigma=0.4, T=30/365, 
                    num_uncertainty_qubits=5, shots=10000, seed=42):
    """
    Executes a QAE (Quantum Amplitude Estimation) approach for pricing
    a European call option under lognormal distribution assumptions.
    
    Returns a dict with the estimation result, confidence intervals,
    and the difference to the Black–Scholes formula.
    """

    # 1. Compute lognormal parameters
    mu, sigma_sq, _, _, _, low, high = compute_lognormal_parameters(S, r, sigma, T)

    # 2. Build the Qiskit LogNormalDistribution
    uncertainty_model = build_log_normal_distribution(
        num_qubits=num_uncertainty_qubits,
        mu=mu,
        sigma_sq=sigma_sq,
        low=low,
        high=high
    )

    # 3. Build EuropeanCallPricing application
    if (high - K) != 0:
        rescaling_factor = 1.0 / (high - K)
    else:
        rescaling_factor = 1.0

    european_call_pricing = EuropeanCallPricing(
        num_state_qubits=num_uncertainty_qubits,
        strike_price=K,
        rescaling_factor=rescaling_factor,
        bounds=(low, high),
        uncertainty_model=uncertainty_model
    )

    # 4. Compute the classical Black–Scholes value
    black_scholes_value = compute_black_scholes_call(S, K, r, sigma, T)

    # 5. Run QAE
    result = run_qae_estimation(european_call_pricing, shots, seed)

    # 6. Interpret QAE result
    estimated_payoff = european_call_pricing.interpret(result)
    discounted_estimated_value = np.exp(-r * T) * estimated_payoff

    # 7. Confidence interval (denormalized)
    conf_int = np.array(result.confidence_interval_processed)
    discounted_conf_int = [np.exp(-r * T) * c for c in conf_int]

    # 8. Compute estimation error
    estimation_error = abs(black_scholes_value - discounted_estimated_value)

    return {
        "ae_result": result,
        "estimated_payoff": estimated_payoff,
        "discounted_estimated_value": discounted_estimated_value,
        "confidence_interval": discounted_conf_int,
        "black_scholes_value": black_scholes_value,
        "estimation_error": estimation_error
    }

