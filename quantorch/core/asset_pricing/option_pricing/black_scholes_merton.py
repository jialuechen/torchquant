import torch
from torch import Tensor
from scipy.stats import norm

def black_scholes_merton(option_type: str, option_style: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> Tensor:
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    d2 = d1 - volatility * torch.sqrt(expiry)
    
    if option_type == 'call':
        if option_style == 'european':
            price = spot * torch.exp(-dividend * expiry) * norm.cdf(d1) - strike * torch.exp(-rate * expiry) * norm.cdf(d2)
        elif option_style == 'american':
            price = max(spot - strike, spot * torch.exp(-dividend * expiry) * norm.cdf(d1) - strike * torch.exp(-rate * expiry) * norm.cdf(d2))
    elif option_type == 'put':
        if option_style == 'european':
            price = strike * torch.exp(-rate * expiry) * norm.cdf(-d2) - spot * torch.exp(-dividend * expiry) * norm.cdf(-d1)
        elif option_style == 'american':
            price = max(strike - spot, strike * torch.exp(-rate * expiry) * norm.cdf(-d2) - spot * torch.exp(-dividend * expiry) * norm.cdf(-d1))
    
    return price