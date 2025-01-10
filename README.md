# QRELibrary

A Python library for running classical and quantum simulations to price European call options. This library includes tools for Monte Carlo simulations, Quantum Amplitude Estimation (QAE), and Quantum Statevector techniques for financial modeling.

---

## Features

- **Classical Monte Carlo Simulation** for European call option pricing with confidence intervals and Black–Scholes comparison.
- **Quantum Amplitude Estimation (QAE)** for option pricing using Qiskit.
- **Quantum Statevector Simulation** for direct probability summation in quantum circuits.
- Easy-to-use **modular functions** and **all-in-one runners** for quick testing and exploration.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/<your-username>/QRELibrary.git
cd QRELibrary
```

### Create a Python Virtual Environment (Optional but Recommended)
```bash
python -m venv qreenv
source qreenv/bin/activate      # On macOS/Linux
qreenv\Scripts\activate         # On Windows
```
### Install the library
Install the library in editable mode to use in notebooks.
```bash
pip install -e .
```

### Configure Jupyter Notebook Kernel
```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=qreenv --display-name "Python (qreenv)"
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Importing the Library
```python
from qre_library import run_classical_monte_carlo, run_quantum_qae, run_quantum_statevector
```

---

### Classical Monte Carlo Simulation
```python
from qre_library import run_classical_monte_carlo
import matplotlib.pyplot as plt

result = run_classical_monte_carlo(S0=50, K=55, r=0.05, sigma=0.4, T=30/365, t=30, M=10000, seed=42)

print("Monte Carlo Estimate:", result["mc_estimate"])
print("Confidence Interval:", result["confidence_interval"])
print("Black-Scholes Value:", result["black_scholes"])

S_paths = result["S_paths"]
plt.figure(figsize=(10, 6))
for i in range(min(10, S_paths.shape[1])):
    plt.plot(S_paths[:, i], linewidth=0.8)
plt.title("Monte Carlo Simulated Price Paths")
plt.xlabel("Time Steps")
plt.ylabel("Asset Price")
plt.grid(True)
plt.show()
```

---

### Quantum Amplitude Estimation (QAE)
```python
from qre_library import run_quantum_qae

result = run_quantum_qae(S=50, K=55, r=0.05, sigma=0.4, T=30/365, num_uncertainty_qubits=5, shots=10000, seed=42)

print("Discounted QAE Estimate:", result["discounted_estimated_value"])
print("Confidence Interval:", result["confidence_interval"])
print("Black-Scholes Value:", result["black_scholes_value"])
```

---

### Quantum Statevector Simulation
```python
from qre_library import run_quantum_statevector

result = run_quantum_statevector(S=50, K=55, r=0.05, sigma=0.4, T=30/365, num_uncertainty_qubits=10)

print("Discounted Statevector Payoff:", result["discounted_statevector_payoff"])
print("Black-Scholes Value:", result["call_price_BS"])
```

---

## Using in Jupyter Notebooks

### Install Jupyter Notebook
```bash
pip install notebook
```

### Open a Notebook
```bash
jupyter notebook
```

### Import the Library
```python
from qre_library import run_classical_monte_carlo, run_quantum_qae, run_quantum_statevector
import matplotlib.pyplot as plt
%matplotlib inline
```

### Run Simulations
```python
from qre_library import run_classical_monte_carlo, run_quantum_qae, run_quantum_statevector
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

---

## License

MIT License
