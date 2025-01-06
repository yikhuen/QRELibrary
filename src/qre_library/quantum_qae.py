# src/qre_library/quantum_qae.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_finance.circuit.library import LogNormalDistribution


def run_quantum_qae(S=50, K=55, r=0.05, sigma=0.4, T=30/365, 
                    num_uncertainty_qubits=5, shots=10000, seed=42):
    """
    Executes a QAE (Quantum Amplitude Estimation) approach for pricing
    a European call option under lognormal distribution assumptions.
    
    Returns a dict with the estimation result, confidence intervals,
    and the difference to the Black–Scholes formula.
    """

    # 1. Lognormal parameters
    mu = (r - 0.5 * sigma**2)*T + np.log(S)
    sigma_sq = (sigma*np.sqrt(T))**2   # Qiskit uses variance in some versions
    mean = np.exp(mu + sigma_sq/2)
    variance = (np.exp(sigma_sq) - 1) * np.exp(2*mu + sigma_sq)
    stddev = np.sqrt(variance)
    low = max(0, mean - 2*stddev)
    high = mean + 2*stddev

    # 2. Build the Qiskit LogNormalDistribution
    uncertainty_model = LogNormalDistribution(
        num_qubits=num_uncertainty_qubits,
        mu=mu,
        sigma=sigma_sq,
        bounds=(low, high)
    )

    # 3. Use Qiskit-Finance's EuropeanCallPricing
    european_call_pricing = EuropeanCallPricing(
        num_state_qubits=num_uncertainty_qubits,
        strike_price=K,
        rescaling_factor=1.0/(high - K) if (high-K)!=0 else 1.0,
        bounds=(low, high),
        uncertainty_model=uncertainty_model
    )

    # 4. Ground truth
    d1 = (np.log(S / K) + (r + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    black_scholes_value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    # 5. Convert problem, run QAE
    sampler = Sampler(options={"shots": shots, "seed": seed})
    problem = european_call_pricing.to_estimation_problem()
    ae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=sampler)
    result = ae.estimate(problem)

    # 6. Interpret QAE result (denormalize payoff)
    estimated_payoff = european_call_pricing.interpret(result)
    discounted_estimated_value = np.exp(-r * T) * estimated_payoff

    # Confidence interval
    conf_int = np.array(result.confidence_interval_processed)
    discounted_conf_int = [np.exp(-r * T) * c for c in conf_int]

    # Estimation error
    estimation_error = abs(black_scholes_value - discounted_estimated_value)

    return {
        "ae_result": result,
        "estimated_payoff": estimated_payoff,
        "discounted_estimated_value": discounted_estimated_value,
        "confidence_interval": discounted_conf_int,
        "black_scholes_value": black_scholes_value,
        "estimation_error": estimation_error
    }
