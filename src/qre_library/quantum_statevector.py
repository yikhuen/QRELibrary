# src/qre_library/quantum_statevector.py

import numpy as np
from scipy.stats import norm

from qiskit import transpile
from qiskit_aer import StatevectorSimulator
from qiskit_finance.circuit.library import LogNormalDistribution

def run_quantum_statevector(S=50, K=55, r=0.05, sigma=0.4, T=30/365, num_uncertainty_qubits=10):
    """
    Demonstrates quantum statevector simulation for pricing a European call
    with a LogNormal distribution, summing probabilities directly from the statevector.
    """

    mu = (r - 0.5*sigma**2)*T + np.log(S)
    sigma_sq = (sigma*np.sqrt(T))**2
    mean = np.exp(mu + sigma_sq/2)
    variance = (np.exp(sigma_sq) - 1)*np.exp(2*mu + sigma_sq)
    stddev = np.sqrt(variance)

    low = max(0, mean - 3*stddev)
    high = mean + 3*stddev

    uncertainty_model = LogNormalDistribution(
        num_qubits=num_uncertainty_qubits,
        mu=mu,
        sigma=sigma_sq,
        bounds=(low, high)
    )

    x_vals = uncertainty_model.values
    probs  = uncertainty_model.probabilities

    # Discrete payoff
    payoff_vals = np.maximum(0, x_vals - K)
    discrete_value = np.dot(probs, payoff_vals)

    # Black–Scholes
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_BS = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    # Statevector simulation
    backend = StatevectorSimulator()
    circ = transpile(uncertainty_model, backend)
    result = backend.run(circ).result()
    statevec = result.get_statevector(circ)

    payoff_sum = 0.0
    for idx, amplitude in enumerate(statevec):
        prob = abs(amplitude)**2
        if prob < 1e-15:
            continue
        # mapping from basis index -> share price
        payoff_val = np.maximum(x_vals[idx] - K, 0)
        payoff_sum += payoff_val * prob

    # discount to present value
    discounted_discrete = discrete_value * np.exp(-r*T)
    discounted_statevector = payoff_sum * np.exp(-r*T)
    error_vs_BS = abs(call_BS - discounted_statevector)

    return {
        "discrete_payoff": discrete_value,
        "statevector_payoff": payoff_sum,
        "discounted_discrete_payoff": discounted_discrete,
        "discounted_statevector_payoff": discounted_statevector,
        "call_price_BS": call_BS,
        "error_vs_BS": error_vs_BS
    }
