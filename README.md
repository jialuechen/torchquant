<div align=center>
<img src="assets/quantorch-high-resolution-color-logo.png" width="50%" loc>
</div>

<div align=center>

[![python version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://github.com/jialuechen)
![PyPI](https://img.shields.io/pypi/v/0.0.1)
![PyPI - License](https://img.shields.io/pypi/l/quantorch)

# PyTorch-based Python Library for Quant Finance
</div>

## Introduction
QuanTorch is developed to empower quantitative research by providing modern deep learning computation and acceleration service. QuanTorch provides high-performance components leveraging the hardware acceleration support and automatic differentiation of PyTorch. QuanTorch supports foundational mathematical methods, mid-level methods, and specific pricing models, which is also an experimental light-weight alternative to QuantLib.

## Motivation
PyTorch provides two high-level features: 

* Tensor computing with strong acceleration via GPU
* Automatic Differentiation System

Quantorch makes use of these modern features on PyTorch library to build advanced stochastic models, high-performance pricing_models, PDE solvers and numerical methods.

## Example
* Binomial Tree Option Pircing Model
* Black-Scholes-Merton Pricing Framework
```
from quantorch.core.optionPricer import OptionPricer
from torch import Tensor

optionType='european',optionDirection='put',\
spot=Tensor([100,95]),strike=Tensor([120,80]),\
expiry=Tensor([1.0,0.75]),volatility=Tensor([0.1,0.3]),\
rate=Tensor([0.01,0.05]),dividend=Tensor([0.01,0.02]),\

pricingModel='BSM'

# here we use GPU accleration as an example
OptionPricer.price(
    optionType,optionDirection,spot,strike,expiry,volatility,rate,dividend,pricingModel,device='GPU'
    )
```
* Root-Finding Algorithms
* Random Walk
* Monte Carlo Simulation
* Risk Management (e.g., Greeks Calculation, Hedging)
* Bayesian Inference
* ... (More promising applications in quantitative finance)


## Installation
```
pip install --upgrade quantorch
```

## License

MIT License
