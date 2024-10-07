# option_pricing_example.py

from torchquantlib.models.stochastic_volatility.heston import Heston

# Set the model parameters
params = {
    'kappa': 2.0,
    'theta': 0.04,
    'sigma_v': 0.3,
    'rho': -0.7,
    'v0': 0.04,
    'mu': 0.05
}

# Initialize the Heston model
heston_model = Heston(**params)

# Option parameters
S0 = 100.0  # Initial asset price
K = 100.0   # Strike price
T = 1.0     # Time to maturity
r = 0.05    # Risk-free interest rate
option_type = 'call'

# Price the option
option_price = heston_model.option_price(S0, K, T, r, option_type, N=100000, steps=200)

print(f"The {option_type} option price under the Heston model is: {option_price:.4f}")