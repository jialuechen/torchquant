import torch
from torch import Tensor
import math
from scipy.stats import norm


def normal_cdf(x):
    return norm.cdf(x)

def normal_pdf(x):
    return norm.pdf(x)
    
def barrier_option(option_type: str, barrier_type: str, spot: Tensor, strike: Tensor, barrier: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

    if option_type == 'call':
        value_tree = torch.maximum(price_tree[:, steps] - strike, torch.tensor(0.0))
    elif option_type == 'put':
        value_tree = torch.maximum(strike - price_tree[:, steps], torch.tensor(0.0))

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            if barrier_type == 'up-and-out' and price_tree[j, i] >= barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'down-and-out' and price_tree[j, i] <= barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'up-and-in' and price_tree[j, i] < barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'down-and-in' and price_tree[j, i] > barrier:
                value_tree[j] = torch.tensor(0.0)

    return value_tree[0]




def chooser_option(spot, strike, expiry, volatility, rate, dividend):
    """
    Calculate the price of a chooser option.

    Parameters:
    - spot (torch.Tensor): The spot price of the underlying asset
    - strike (torch.Tensor): The strike price of the option
    - expiry (torch.Tensor): The time to expiry in years
    - volatility (torch.Tensor): The volatility of the underlying asset
    - rate (torch.Tensor): The risk-free interest rate
    - dividend (torch.Tensor): The dividend yield

    Returns:
    - torch.Tensor: The price of the chooser option
    """
    spot = spot.item()
    strike = strike.item()
    expiry = expiry.item()
    volatility = volatility.item()
    rate = rate.item()
    dividend = dividend.item()
    
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * math.sqrt(expiry))
    d2 = d1 - volatility * math.sqrt(expiry)

    call_price = spot * math.exp(-dividend * expiry) * normal_cdf(d1) - strike * math.exp(-rate * expiry) * normal_cdf(d2)
    put_price = strike * math.exp(-rate * expiry) * normal_cdf(-d2) - spot * math.exp(-dividend * expiry) * normal_cdf(-d1)
    
    return torch.tensor(call_price + put_price)

def compound_option(spot, strike1, strike2, expiry1, expiry2, volatility, rate, dividend):
    """
    Calculate the price of a compound option (an option on an option).

    Parameters:
    - spot (torch.Tensor): The spot price of the underlying asset
    - strike1 (torch.Tensor): The strike price of the underlying option
    - strike2 (torch.Tensor): The strike price of the compound option
    - expiry1 (torch.Tensor): The time to expiry of the underlying option in years
    - expiry2 (torch.Tensor): The time to expiry of the compound option in years
    - volatility (torch.Tensor): The volatility of the underlying asset
    - rate (torch.Tensor): The risk-free interest rate
    - dividend (torch.Tensor): The dividend yield

    Returns:
    - torch.Tensor: The price of the compound option
    """
    spot = spot.item()
    strike1 = strike1.item()
    strike2 = strike2.item()
    expiry1 = expiry1.item()
    expiry2 = expiry2.item()
    volatility = volatility.item()
    rate = rate.item()
    dividend = dividend.item()
    
    d1 = (math.log(spot / strike1) + (rate - dividend + 0.5 * volatility ** 2) * expiry1) / (volatility * math.sqrt(expiry1))
    d2 = d1 - volatility * math.sqrt(expiry1)
    d3 = (math.log(spot / strike2) + (rate - dividend + 0.5 * volatility ** 2) * expiry2) / (volatility * math.sqrt(expiry2))
    d4 = d3 - volatility * math.sqrt(expiry2)

    price = math.exp(-rate * expiry2) * (spot * math.exp((rate - dividend) * expiry2) * normal_cdf(d3) - strike1 * normal_cdf(d4))
    
    return torch.tensor(price)

def shout_option(spot, strike, expiry, volatility, rate, dividend):
    """
    Calculate the price of a shout option.

    Parameters:
    - spot (torch.Tensor): The spot price of the underlying asset
    - strike (torch.Tensor): The strike price of the option
    - expiry (torch.Tensor): The time to expiry in years
    - volatility (torch.Tensor): The volatility of the underlying asset
    - rate (torch.Tensor): The risk-free interest rate
    - dividend (torch.Tensor): The dividend yield

    Returns:
    - torch.Tensor: The price of the shout option
    """
    spot = spot.item()
    strike = strike.item()
    expiry = expiry.item()
    volatility = volatility.item()
    rate = rate.item()
    dividend = dividend.item()
    
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * math.sqrt(expiry))
    d2 = d1 - volatility * math.sqrt(expiry)

    call_price = spot * math.exp(-dividend * expiry) * normal_cdf(d1) - strike * math.exp(-rate * expiry) * normal_cdf(d2)
    shout_value = spot * math.exp(-dividend * expiry) * (1 - normal_cdf(d1))

    return torch.tensor(call_price + shout_value)
