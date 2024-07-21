import torch
from torch import Tensor
from scipy.optimize import brentq

def implied_volatility(option_type: str, option_style: str, market_price: Tensor, spot: Tensor, strike: Tensor, expiry: Tensor, rate: Tensor, dividend: Tensor, initial_guess: float = 0.2) -> Tensor:
    def objective_function(volatility):
        return black_scholes_merton(option_type, option_style, spot, strike, expiry, volatility, rate, dividend) - market_price

    implied_vol = brentq(objective_function, 1e-6, 10, xtol=1e-6)
    return torch.tensor(implied_vol)

# Example usage:
# iv = implied_volatility('call', 'european', market_price, spot, strike, expiry, rate, dividend)