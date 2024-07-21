<div align=center>
<img src="assets/img/quantorch_logo.png" width="40%" loc>

# QuanTorch

</div>

[![Build Status](https://img.shields.io/github/actions/workflow/status/jialuechen/quantorch/python-package.yml)](https://github.com/jialuechen/quantorch/actions)
[![PyPI version](https://img.shields.io/pypi/v/quantorch)](https://pypi.org/project/quantorch/)
[![License](https://img.shields.io/github/license/jialuechen/quantorch)](https://github.com/jialuechen/quantorch/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/quantorch)](https://pypi.org/project/quantorch/)

QuanTorch is a comprehensive quantitative finance library built on top of PyTorch. It provides a range of tools for asset pricing, risk management, and model calibration.

## Features

- **Asset Pricing**:
  - Option pricing models including Black-Scholes-Merton, binomial tree, and Monte Carlo simulations.
  - Bond pricing models including zero-coupon, coupon, callable, putable, and convertible bonds.
  - Advanced options support including American, Bermudan, Asian, and barrier options.
  - Implied volatility calculation using "Let's be rational" algorithm.
  - Futures and currency pricing.

- **Risk Management**:
  - Greeks calculation including Malliavin calculus.
  - Scenario analysis and stress testing.
  - Market risk measures such as VaR and Expected Shortfall.
  - Credit risk models including structural and reduced form models.
  - Valuation adjustments (CVA, DVA, MVA, FVA).

- **Model Calibration**:
  - Calibration for stochastic models like Heston, Vasicek, SABR, and more.
  - Local volatility models including Dupire.

- **Machine Learning and Reinforcement Learning**:
  - Regression and classification models using PyTorch.
  - Reinforcement learning examples.

## Installation

You can install QuanTorch via pip:

```bash
pip install quantorch