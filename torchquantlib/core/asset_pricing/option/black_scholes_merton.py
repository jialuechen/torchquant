import torch
from torch import Tensor
from scipy.stats import norm

def black_scholes_merton(option_type: str, option_style: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> Tensor:
    """
    Price an option using the Black-Scholes-Merton model.

    This function implements the Black-Scholes-Merton model to price European and American options on dividend-paying stocks.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        option_style (str): Style of option - either 'european' or 'american'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        dividend (Tensor): Continuous dividend yield of the underlying asset.

    Returns:
        Tensor: The price of the option.

    Note:
        - This implementation uses the Black-Scholes-Merton formula for European options.
        - For American options, it uses a simple approximation that may not be accurate in all cases.
        - For more accurate pricing of American options, especially puts on dividend-paying stocks,
          numerical methods like binomial trees or finite difference methods are recommended.
    """
    # Calculate d1 and d2 parameters
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    d2 = d1 - volatility * torch.sqrt(expiry)
    
    if option_type == 'call':
        if option_style == 'european':
            # European call option price
            price = spot * torch.exp(-dividend * expiry) * norm.cdf(d1) - strike * torch.exp(-rate * expiry) * norm.cdf(d2)
        elif option_style == 'american':
            # American call option price (approximation)
            price = max(spot - strike, spot * torch.exp(-dividend * expiry) * norm.cdf(d1) - strike * torch.exp(-rate * expiry) * norm.cdf(d2))
    elif option_type == 'put':
        if option_style == 'european':
            # European put option price
            price = strike * torch.exp(-rate * expiry) * norm.cdf(-d2) - spot * torch.exp(-dividend * expiry) * norm.cdf(-d1)
        elif option_style == 'american':
            # American put option price (approximation)
            price = max(strike - spot, strike * torch.exp(-rate * expiry) * norm.cdf(-d2) - spot * torch.exp(-dividend * expiry) * norm.cdf(-d1))
    
    return price