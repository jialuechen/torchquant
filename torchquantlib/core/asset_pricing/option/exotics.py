import torch
from torch import Tensor
from scipy.stats import norm
from .black_scholes_merton import black_scholes_merton

ZERO = torch.tensor(0.0)

def validate_parameters(volatility: Tensor, expiry: Tensor, steps: int, option_type: str, valid_option_types: list):
    """
    Validate common parameters for option pricing functions.
    """
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    if option_type not in valid_option_types:
        raise ValueError(f"Option type must be one of {valid_option_types}.")

def initialize_binomial_tree(spot: Tensor, steps: int, u: Tensor, d: Tensor) -> Tensor:
    """
    Initialize the binomial tree for asset prices.
    """
    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)
    return price_tree

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
    validate_parameters(volatility, expiry, steps, option_type, ['call', 'put'])
    if barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']:
        raise ValueError("Barrier type must be one of 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'.")
    # Calculate parameters for the binomial model
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    # Initialize the price tree
    price_tree = initialize_binomial_tree(spot, steps, u, d)

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

def lookback_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int, strike_type: str = 'fixed') -> Tensor:
    """
    Price a lookback option using a binomial tree model.

    A lookback option's payoff depends on the optimal (maximum for call, minimum for put) price of the underlying asset during the option's life.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option (used only for fixed strike).
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.
        strike_type (str): Type of strike price - either 'fixed' or 'floating'. Defaults to 'fixed'.

    Returns:
        Tensor: The price of the lookback option.
    """
    validate_parameters(volatility, expiry, steps, option_type, ['call', 'put'])
    if strike_type not in ['fixed', 'floating']:
        raise ValueError("Strike type must be either 'fixed' or 'floating'.")

    # Calculate parameters for the binomial model
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    # Initialize the price tree
    price_tree = initialize_binomial_tree(spot, steps, u, d)

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
    if strike_type == 'fixed':
        if option_type == 'call':
            value_tree = torch.maximum(min_max_price_tree[:, steps] - strike, ZERO)
        elif option_type == 'put':
            value_tree = torch.maximum(strike - min_max_price_tree[:, steps], ZERO)
    elif strike_type == 'floating':
        if option_type == 'call':
            value_tree = torch.maximum(price_tree[:, steps] - min_max_price_tree[:, steps], ZERO)
        elif option_type == 'put':
            value_tree = torch.maximum(min_max_price_tree[:, steps] - price_tree[:, steps], ZERO)

    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)

    return value_tree[0]

def rainbow_option(spot: Tensor, weights: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, correlation: Tensor, steps: int) -> Tensor:
    """
    Price a rainbow option using a binomial tree model.

    Args:
        spot (Tensor): Spot prices of the underlying assets.
        weights (Tensor): Weights of the assets in the rainbow option.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatilities of the underlying assets.
        rate (Tensor): Risk-free interest rate.
        correlation (Tensor): Correlation matrix of the assets.
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the rainbow option.
    """
    # Validate inputs
    validate_parameters(volatility.min(), expiry, steps, 'call', ['call', 'put'])
    if correlation.shape[0] != correlation.shape[1] or correlation.shape[0] != len(spot):
        raise ValueError("Correlation matrix must be square and match the number of assets.")

    # Cholesky decomposition for correlated random numbers
    L = torch.linalg.cholesky(correlation)
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    # Initialize price tree
    price_tree = torch.zeros((steps + 1, steps + 1, len(spot)))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

    # Calculate weighted basket price
    basket_price = torch.sum(weights * price_tree, dim=2)

    # Backward induction
    value_tree = torch.maximum(basket_price[:, :, -1] - strike, ZERO)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j, i] = (p * value_tree[j, i + 1] + (1 - p) * value_tree[j + 1, i + 1]) * torch.exp(-rate * dt)

    return value_tree[0, 0]

def quanto_option(spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, fx_rate: Tensor, fx_volatility: Tensor, correlation: Tensor, steps: int) -> Tensor:
    """
    Price a quanto option using a binomial tree model.

    Args:
        spot (Tensor): Spot price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate.
        fx_rate (Tensor): Foreign exchange rate.
        fx_volatility (Tensor): Volatility of the foreign exchange rate.
        correlation (Tensor): Correlation between the asset and FX rate.
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the quanto option.
    """
    # Adjusted volatility
    adjusted_volatility = torch.sqrt(volatility**2 + fx_volatility**2 - 2 * correlation * volatility * fx_volatility)

    # Use standard binomial tree model with adjusted volatility
    return binomial_tree('call', 'european', spot, strike, expiry, adjusted_volatility, rate, steps)

def digital_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, payout: Tensor, steps: int) -> Tensor:
    """
    Price a digital option using a binomial tree model.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Spot price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate.
        payout (Tensor): Fixed payout of the option.
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the digital option.
    """
    # Validate inputs
    validate_parameters(volatility, expiry, steps, option_type, ['call', 'put'])

    # Calculate parameters for the binomial model
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    price_tree = initialize_binomial_tree(spot, steps, u, d)

    # Initialize the option value at expiration
    if option_type == 'call':
        value_tree = (price_tree[:, steps] >= strike).float() * payout
    elif option_type == 'put':
        value_tree = (price_tree[:, steps] <= strike).float() * payout

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)

    return value_tree[0]
