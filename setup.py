# setup.py
from setuptools import setup, find_packages

setup(
    name="qre_library",
    version="0.1.0",
    description="A small Python library demonstrating Quantum Risk Estimation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "qiskit",
        "qiskit-finance",
    ],
    python_requires=">=3.8"
)
