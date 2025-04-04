import torch
from torch import Tensor
from scipy.stats import norm
from .black_scholes_merton import black_scholes_merton

ZERO = torch.tensor(0.0)

def validate_parameters(volatility: Tensor, expiry: Tensor, num_paths: int = None):
    """
    Validate common parameters for option pricing functions.
    """
    if volatility <= ZERO or expiry <= ZERO:
        raise ValueError("Volatility and expiry must be positive.")
    if num_paths is not None and num_paths <= 0:
        raise ValueError("Number of paths must be a positive integer.")

def digital_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, payout: Tensor, num_paths: int) -> Tensor:
    """
    Price a digital option using Monte Carlo simulation.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Spot price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate.
        payout (Tensor): Fixed payout of the option.
        num_paths (int): Number of Monte Carlo simulation paths.

    Returns:
        Tensor: The price of the digital option.
    """
    validate_parameters(volatility, expiry, num_paths)

    # Simulate asset price paths at expiry
    z = torch.randn(num_paths, device=spot.device)
    asset_prices = spot * torch.exp((rate - 0.5 * volatility**2) * expiry + volatility * torch.sqrt(expiry) * z)

    # Calculate payoffs
    if option_type == 'call':
        payoffs = (asset_prices >= strike).float() * payout
    elif option_type == 'put':
        payoffs = (asset_prices <= strike).float() * payout

    # Discount payoffs to present value
    discounted_payoffs = payoffs * torch.exp(-rate * expiry)

    return discounted_payoffs.mean()

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
    validate_parameters(volatility, expiry)

    # Calculate call and put prices using Black-Scholes formula
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
    validate_parameters(volatility, expiry1)
    validate_parameters(volatility, expiry2)

    # Price of the underlying option at expiry2
    underlying_option_price = black_scholes_merton(
        option_type='call',
        option_style='european',
        spot=spot,
        strike=strike1,
        expiry=expiry1,
        volatility=volatility,
        rate=rate,
        dividend=dividend
    )

    # Compound option price
    d1 = (torch.log(underlying_option_price / strike2) + (rate + 0.5 * volatility ** 2) * expiry2) / (volatility * torch.sqrt(expiry2))
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
    validate_parameters(volatility, expiry)

    # Standard call option price
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

    # Additional value from the shout feature
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    shout_value = spot * torch.exp(-dividend * expiry) * (1 - torch.tensor(norm.cdf(d1)))

    return call_price + shout_value

def lookback_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, num_paths: int, strike_type: str = 'fixed') -> Tensor:
    """
    Price a lookback option using Monte Carlo simulation.

    A lookback option's payoff depends on the optimal (maximum for call, minimum for put) price of the underlying asset during the option's life.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option (used only for fixed strike).
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        num_paths (int): Number of Monte Carlo simulation paths.
        strike_type (str): Type of strike price - either 'fixed' or 'floating'. Defaults to 'fixed'.

    Returns:
        Tensor: The price of the lookback option.
    """
    validate_parameters(volatility, expiry, num_paths)
    if strike_type not in ['fixed', 'floating']:
        raise ValueError("Strike type must be either 'fixed' or 'floating'.")

    dt = expiry / 252  # Assume 252 trading days in a year
    num_steps = int(expiry / dt)
    z = torch.randn(num_paths, num_steps, device=spot.device)
    asset_paths = torch.zeros(num_paths, num_steps + 1, device=spot.device)
    asset_paths[:, 0] = spot

    # Simulate asset price paths
    for t in range(1, num_steps + 1):
        asset_paths[:, t] = asset_paths[:, t - 1] * torch.exp((rate - 0.5 * volatility**2) * dt + volatility * torch.sqrt(dt) * z[:, t - 1])

    if option_type == 'call':
        if strike_type == 'fixed':
            max_prices = asset_paths.max(dim=1).values
            payoffs = torch.maximum(max_prices - strike, ZERO)
        elif strike_type == 'floating':
            final_prices = asset_paths[:, -1]
            max_prices = asset_paths.max(dim=1).values
            payoffs = torch.maximum(final_prices - max_prices, ZERO)
    elif option_type == 'put':
        if strike_type == 'fixed':
            min_prices = asset_paths.min(dim=1).values
            payoffs = torch.maximum(strike - min_prices, ZERO)
        elif strike_type == 'floating':
            final_prices = asset_paths[:, -1]
            min_prices = asset_paths.min(dim=1).values
            payoffs = torch.maximum(min_prices - final_prices, ZERO)

    # Discount payoffs to present value
    discounted_payoffs = payoffs * torch.exp(-rate * expiry)

    return discounted_payoffs.mean()