import torch
from torch import Tensor
from scipy.stats import norm
from .black_scholes_merton import black_scholes_merton

ZERO = torch.tensor(0.0)

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
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be either 'call' or 'put'.")
    if barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']:
        raise ValueError("Barrier type must be one of 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'.")
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
        value_tree = torch.maximum(price_tree[:, steps] - strike, ZERO)
    elif option_type == 'put':
        value_tree = torch.maximum(strike - price_tree[:, steps], ZERO)

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            # Apply barrier conditions
            if barrier_type == 'up-and-out' and price_tree[j, i] >= barrier:
                value_tree[j] = ZERO
            elif barrier_type == 'down-and-out' and price_tree[j, i] < barrier:
                value_tree[j] = ZERO
            elif barrier_type == 'up-and-in' and price_tree[j, i] >= barrier:
                pass
            elif barrier_type == 'down-and-in' and price_tree[j, i] < barrier:
                pass

    return value_tree[0]

def chooser_option(spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> Tensor:
    """
    Calculate the price of a chooser option.

    A chooser option gives the holder the right to choose whether the option is a call or put at time 0.

    Args:
        spot (Tensor): The spot price of the underlying asset
        strike (Tensor): The strike price of the option
        expiry (Tensor): The time to expiry in years
        volatility (Tensor): The volatility of the underlying asset
        rate (Tensor): The risk-free interest rate
        dividend (Tensor): The dividend yield

    Returns:
        Tensor: The price of the chooser option
    """
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    call_price = black_scholes_merton(
        option_type='call',
        option_style='european',
        spot=spot,
        strike=strike,
        expiry=expiry,
        volatility=volatility,
        rate=rate,
        dividend=dividend
    )
    put_price = black_scholes_merton(
        option_type='put',
        option_style='european',
        spot=spot,
        strike=strike,
        expiry=expiry,
        volatility=volatility,
        rate=rate,
        dividend=dividend
    )
    return call_price + put_price

def compound_option(spot: Tensor, strike1: Tensor, strike2: Tensor, expiry1: Tensor, expiry2: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> Tensor:
    """
    Calculate the price of a compound option (an option on an option).

    Args:
        spot (Tensor): The spot price of the underlying asset
        strike1 (Tensor): The strike price of the underlying option
        strike2 (Tensor): The strike price of the compound option
        expiry1 (Tensor): The time to expiry of the underlying option in years
        expiry2 (Tensor): The time to expiry of the compound option in years
        volatility (Tensor): The volatility of the underlying asset
        rate (Tensor): The risk-free interest rate
        dividend (Tensor): The dividend yield

    Returns:
        Tensor: The price of the compound option
    """
    if volatility <= ZERO or expiry1 <= ZERO or expiry2 <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    
    # Calculate the price of the underlying option at expiry2
    underlying_option_price = black_scholes_merton(option_type='call', option_style='european', spot=spot, strike=strike1, expiry=expiry1, volatility=volatility, rate=rate, dividend=dividend)
    
    # Calculate the price of the compound option
    d1 = (torch.log(spot / strike2) + (rate - dividend + 0.5 * volatility ** 2) * expiry2) / (volatility * torch.sqrt(expiry2))
    d2 = d1 - volatility * torch.sqrt(expiry2)
    price = torch.exp(-rate * expiry2) * (underlying_option_price * norm.cdf(d1) - strike2 * norm.cdf(d2))
    
    return price

def shout_option(spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> Tensor:
    """
    Calculate the price of a shout option.

    A shout option allows the holder to "shout" once during the life of the option to lock in a minimum payoff.

    Args:
        spot (Tensor): The spot price of the underlying asset
        strike (Tensor): The strike price of the option
        expiry (Tensor): The time to expiry in years
        volatility (Tensor): The volatility of the underlying asset
        rate (Tensor): The risk-free interest rate
        dividend (Tensor): The dividend yield

    Returns:
        Tensor: The price of the shout option
    """
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    
    # Calculate the price of a standard call option
    call_price = black_scholes_merton(option_type='call', option_style='european', spot=spot, strike=strike, expiry=expiry, volatility=volatility, rate=rate, dividend=dividend)
    
    # Calculate the additional value from the shout feature
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    shout_value = spot * torch.exp(-dividend * expiry) * (1 - torch.tensor(norm.cdf(d1)))

    # The shout option price is the sum of the call price and the shout value
    return call_price + shout_value

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
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be either 'call' or 'put'.")
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

    # Initialize the min/max price tree
    min_max_price_tree = torch.zeros((steps + 1, steps + 1))
    min_max_price_tree[:, steps] = price_tree[:, steps]

    # Backward induction to calculate min/max price
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            if option_type == 'call':
                min_max_price_tree[j, i] = torch.maximum(price_tree[j, i], torch.maximum(min_max_price_tree[j, i + 1], min_max_price_tree[j + 1, i + 1]))
            elif option_type == 'put':
                min_max_price_tree[j, i] = torch.minimum(price_tree[j, i], torch.minimum(min_max_price_tree[j, i + 1], min_max_price_tree[j + 1, i + 1]))

    # Initialize the option value at expiration
    if option_type == 'call':
        value_tree = torch.maximum(min_max_price_tree[:, steps] - strike, ZERO)
    elif option_type == 'put':
        value_tree = torch.maximum(strike - min_max_price_tree[:, steps], ZERO)

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)

    return value_tree[0]
