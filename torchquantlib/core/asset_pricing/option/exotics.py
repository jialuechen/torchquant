import torch
from torch import Tensor
import math
from scipy.stats import norm

def barrier_option(option_type: str, barrier_type: str, spot: Tensor, strike: Tensor, barrier: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """
    Price a barrier option using a binomial tree model.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        barrier_type (str): Type of barrier - 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        barrier (Tensor): Barrier price.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the barrier option.
    """
    # Calculate parameters for the binomial model
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    # Initialize the price tree
    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

    # Initialize the option value at expiration
    if option_type == 'call':
        value_tree = torch.maximum(price_tree[:, steps] - strike, torch.tensor(0.0))
    elif option_type == 'put':
        value_tree = torch.maximum(strike - price_tree[:, steps], torch.tensor(0.0))

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            # Apply barrier conditions
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

    A chooser option gives the holder the right to choose whether the option is a call or put at some point before expiration.

    Args:
        spot (torch.Tensor): The spot price of the underlying asset
        strike (torch.Tensor): The strike price of the option
        expiry (torch.Tensor): The time to expiry in years
        volatility (torch.Tensor): The volatility of the underlying asset
        rate (torch.Tensor): The risk-free interest rate
        dividend (torch.Tensor): The dividend yield

    Returns:
        torch.Tensor: The price of the chooser option
    """
    # Convert tensor inputs to Python floats for compatibility with math functions
    spot, strike, expiry, volatility, rate, dividend = [x.item() for x in (spot, strike, expiry, volatility, rate, dividend)]
    
    # Calculate d1 and d2 parameters
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * math.sqrt(expiry))
    d2 = d1 - volatility * math.sqrt(expiry)

    # Calculate call and put prices using Black-Scholes formula
    call_price = spot * math.exp(-dividend * expiry) * norm.cdf(d1) - strike * math.exp(-rate * expiry) * norm.cdf(d2)
    put_price = strike * math.exp(-rate * expiry) * norm.cdf(-d2) - spot * math.exp(-dividend * expiry) * norm.cdf(-d1)
    
    # The chooser option price is the sum of call and put prices
    return torch.tensor(call_price + put_price)

def compound_option(spot, strike1, strike2, expiry1, expiry2, volatility, rate, dividend):
    """
    Calculate the price of a compound option (an option on an option).

    Args:
        spot (torch.Tensor): The spot price of the underlying asset
        strike1 (torch.Tensor): The strike price of the underlying option
        strike2 (torch.Tensor): The strike price of the compound option
        expiry1 (torch.Tensor): The time to expiry of the underlying option in years
        expiry2 (torch.Tensor): The time to expiry of the compound option in years
        volatility (torch.Tensor): The volatility of the underlying asset
        rate (torch.Tensor): The risk-free interest rate
        dividend (torch.Tensor): The dividend yield

    Returns:
        torch.Tensor: The price of the compound option
    """
    # Convert tensor inputs to Python floats
    spot, strike1, strike2, expiry1, expiry2, volatility, rate, dividend = [x.item() for x in (spot, strike1, strike2, expiry1, expiry2, volatility, rate, dividend)]
    
    # Calculate d parameters
    d1 = (math.log(spot / strike1) + (rate - dividend + 0.5 * volatility ** 2) * expiry1) / (volatility * math.sqrt(expiry1))
    d2 = d1 - volatility * math.sqrt(expiry1)
    d3 = (math.log(spot / strike2) + (rate - dividend + 0.5 * volatility ** 2) * expiry2) / (volatility * math.sqrt(expiry2))
    d4 = d3 - volatility * math.sqrt(expiry2)

    # Calculate the price of the compound option
    price = math.exp(-rate * expiry2) * (spot * math.exp((rate - dividend) * expiry2) * norm.cdf(d3) - strike1 * norm.cdf(d4))
    
    return torch.tensor(price)

def shout_option(spot, strike, expiry, volatility, rate, dividend):
    """
    Calculate the price of a shout option.

    A shout option allows the holder to "shout" once during the life of the option to lock in a minimum payoff.

    Args:
        spot (torch.Tensor): The spot price of the underlying asset
        strike (torch.Tensor): The strike price of the option
        expiry (torch.Tensor): The time to expiry in years
        volatility (torch.Tensor): The volatility of the underlying asset
        rate (torch.Tensor): The risk-free interest rate
        dividend (torch.Tensor): The dividend yield

    Returns:
        torch.Tensor: The price of the shout option
    """
    # Convert tensor inputs to Python floats
    spot, strike, expiry, volatility, rate, dividend = [x.item() for x in (spot, strike, expiry, volatility, rate, dividend)]
    
    # Calculate d1 and d2 parameters
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * math.sqrt(expiry))
    d2 = d1 - volatility * math.sqrt(expiry)

    # Calculate the price of a standard call option
    call_price = spot * math.exp(-dividend * expiry) * norm.cdf(d1) - strike * math.exp(-rate * expiry) * norm.cdf(d2)
    
    # Calculate the additional value from the shout feature
    shout_value = spot * math.exp(-dividend * expiry) * (1 - norm.cdf(d1))

    # The shout option price is the sum of the call price and the shout value
    return torch.tensor(call_price + shout_value)

def lookback_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """
    Price a lookback option using a binomial tree model.

    A lookback option's payoff depends on the optimal (maximum for call, minimum for put) price of the underlying asset during the option's life.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the lookback option.
    """
    # Calculate parameters for the binomial model
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    # Initialize the price tree
    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

    # Initialize and calculate the minimum/maximum price tree
    min_max_price_tree = price_tree.clone()
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            min_max_price_tree[j, i] = torch.minimum(min_max_price_tree[j, i], torch.minimum(min_max_price_tree[j, i + 1], min_max_price_tree[j + 1, i + 1]))

    # Initialize the option value at expiration
    if option_type == 'call':
        value_tree = torch.maximum(min_max_price_tree[:, steps] - strike, torch.tensor(0.0))
    elif option_type == 'put':
        value_tree = torch.maximum(strike - min_max_price_tree[:, steps], torch.tensor(0.0))

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)

    return value_tree[0]