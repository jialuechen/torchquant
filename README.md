<div align=center>
<img src="assets/torchquant.png" width="40%" loc>

[![PyPI version](https://badge.fury.io/py/torchquantlib.svg)](https://badge.fury.io/py/torchquantlib)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python versions](https://img.shields.io/badge/python-3.7%2B-green)
![PyTorch version](https://img.shields.io/badge/pytorch-2.4.0%2B-green)
![Downloads](https://img.shields.io/pypi/dm/torchquantlib)
[![Coverage Status](https://coveralls.io/repos/github/jialuechen/torchquant/badge.svg?branch=main)](https://coveralls.io/github/jialuechen/tfq-finance?branch=main)
[![Documentation Status](https://readthedocs.org/projects/torchquant/badge/?version=latest)](https://torchquant.readthedocs.io/en/latest/?badge=latest)
</div>

  
TorchQuant : High-Performance PyTorch Library for Quantitative Finance


TorchQuant is a comprehensive quantitative financelibrary built on top of PyTorch's automatic differentiation and GPU acceleration. It is a differentiable pricing framework with high-accuracy of numerical methods. It provides comprehensive tools for derivatives pricing, risk management, and stochastic model calibration.

## Features

- **Asset Pricing**:
  - Option pricing models including BSM, Heston, binomial tree, and Monte Carlo simulations.
  - Bond pricing models including callable, putable, and convertible bonds.
  - Advanced options support for American, Bermudan, Asian, barrier and look-back options.
  - Implied volatility calculation using "Let's be rational" algorithm.
  - Futures and currency pricing.

- **Risk Management**:
  - Greeks calculation utilizing Malliavin calculus.
  - Scenario analysis and stress testing.
  - Market risk measures such as VaR and Expected Shortfall.
  - Credit risk models including structural and reduced form models.
  - Valuation adjustments (CVA, DVA, MVA, FVA).

- **Neural Network-based Model Calibration**:
  - Calibration for stochastic models like Heston, Vasicek, SABR, and more.
  - Local volatility models including Dupire.
  - Optimal Transport for model calibration
 
- **Sequence Methods**:
  - Seq2Seq PDE solvers

## Installation

You can install torchquant via pip:

```bash
pip install -U torchquantlib
```

## Usage (check out the examples folder for more information)

### Exotic Options

#### American Option

```python
import torch
from torchquantlib.core.asset_pricing.option_pricing.american_option import american_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = american_option('call', spot, strike, expiry, volatility, rate, steps)
print(f'American Option Price: {price.item()}')
```

#### Bermudan Option

```python
import torch
from torchquantlib.core.asset_pricing.option_pricing.bermudan_option import bermudan_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100
exercise_dates = torch.tensor([30, 60, 90])

price = bermudan_option('call', spot, strike, expiry, volatility, rate, steps, exercise_dates)
print(f'Bermudan Option Price: {price.item()}')
```

#### Asian Option

```python
import torch
from torchquantlib.core.asset_pricing.option_pricing.asian_option import asian_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = asian_option('call', spot, strike, expiry, volatility, rate, steps)
print(f'Asian Option Price: {price.item()}')
```

### Greeks Calculation using Malliavin Calculus

```python
import torch
from torchquantlib.core.risk.greeks.malliavin import malliavin_greek

option_price = torch.tensor(10.0)
underlying_price = torch.tensor(100.0)
volatility = torch.tensor(0.2)
expiry = torch.tensor(1.0)

greek = malliavin_greek(option_price, underlying_price, volatility, expiry)
print(f'Malliavin Greek: {greek.item()}')
```

### Model Calibration by Optimal Transport

#### Heston Model Calibration

```python
# calibrate_heston.py

import numpy as np
import torch
from torchquantlib.calibration import model_calibrator
from torchquantlib.models.stochastic_volatility.heston import Heston

# Generate synthetic observed data using true Heston parameters
N_observed = 1000
S0 = 100.0
T = 1.0
true_params = {
    'kappa': 2.0,
    'theta': 0.04,
    'sigma_v': 0.3,
    'rho': -0.7,
    'v0': 0.04,
    'mu': 0.05
}

np.random.seed(42)
torch.manual_seed(42)
heston_true = Heston(**true_params)
S_observed = heston_true.simulate(S0=S0, T=T, N=N_observed)

# Initialize the Heston model with initial guesses
heston_model = Heston(
    kappa_init=1.0,
    theta_init=0.02,
    sigma_v_init=0.2,
    rho_init=-0.5,
    v0_init=0.02,
    mu_init=0.0
)

# Set up the calibrator
calibrator = model_calibrator(
    model=heston_model,
    observed_data=S_observed.detach().cpu().numpy(),  # Convert tensor to numpy array
    S0=S0,
    T=T,
    lr=0.01
)

# Calibrate the model
calibrator.calibrate(num_epochs=1000, steps=100, verbose=True)

# Get the calibrated parameters
calibrated_params = calibrator.get_calibrated_params()
print("Calibrated Parameters:")
for name, value in calibrated_params.items():
    print(f"{name}: {value:.6f}")
```

## Seq2Seq PDE Solver
```python
from torchquantlib.utils import Seq2SeqPDESolver
# Define model parameters
input_dim = 1      # Adjust based on your input features
hidden_dim = 64    # Number of features in the hidden state
output_dim = 1     # Adjust based on your output features
num_layers = 2     # Number of stacked LSTM layers

# Initialize the model, loss function, and optimizer
model = Seq2SeqPDESolver(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(src, trg)
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
```

## Credit Risk Management Example
```python
import torch
from torchquantlib.core.risk.credit_risk.structural_model import merton_model
from torchquantlib.core.risk.credit_risk.reduced_form_model import reduced_form_model

asset_value = torch.tensor(100.0)
debt = torch.tensor(80.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
expiry = torch.tensor(1.0)

# Merton Model
equity_value = merton_model(asset_value, debt, volatility, rate, expiry)
print(f'Equity Value (Merton Model): {equity_value.item()}')

lambda_0 = torch.tensor(0.02)
default_intensity = torch.tensor(0.05)
recovery_rate = torch.tensor(0.4)
time = torch.tensor(1.0)

# Reduced Form Model
expected_loss = reduced_form_model(lambda_0, default_intensity, recovery_rate, time)
print(f'Expected Loss (Reduced Form Model): {expected_loss.item()}')
```

## Roadmap


### Q4 2024
- Develop an interactive dashboard for visualizing risk metrics and option pricing.

### Q3 2025
- Add support for more asset classes such as commodities and real estate.
- Enhance the calibration module to support a broader range of models and optimization techniques.

## Development

To contribute to TorchQuant, clone the repository and install the required dependencies:

```bash
git clone https://github.com/jialuechen/torchquant.git
cd torchquant
pip install -r requirements.txt
```

Run the tests to ensure everything is working:

```bash
pytest
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
