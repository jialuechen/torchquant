<img decoding="async" src="quantorch-high-resolution-color-logo.png" width="50%">

# QuanTorch : Quantitative Finance Research Framework Built on Deep Learning 

[![python version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://github.com/jialuechen)
![PyPI](https://img.shields.io/pypi/v/0.0.1)
![PyPI - License](https://img.shields.io/pypi/l/quantorch)

QuanTorch is a PyTorch-based Python Library for Quantitative Finance.

## Introduction
QuanTorch is developed to empower quantitative research by providing modern deep learning computation and acceleration service. QuanTorch provides high-performance components leveraging the hardware acceleration support and automatic differentiation of PyTorch. QuanTorch supports foundational mathematical methods, mid-level methods, and specific pricing models, which is also an experimental light-weight alternative to QuantLib.

## Motivation
PyTorch provides two high-level features: 

* Tensor computing with strong acceleration via GPU
* Automatic Differentiation System

Quantorch makes use of these modern features on PyTorch library to build advanced stochastic models, high-performance pricing_models, PDE solvers and numerical methods.

## Example
* Binomial Tree Option Pircing Model
* Black-Scholes Pricing Framework
```
from quantorch.core.optionPricer import OptionPricer
from quantorch.instruments.derivatives import Option
```
* Root-Finding Algorithms
* Random Walk
* Monte Carlo Simulation
* Risk Management (e.g., Greeks Calculation, Hedging)
* Bayesian Inference
* ... (More hidden application in Quant Finance)


## Installation
```
pip install --upgrade quantorch
```

## License

MIT License
