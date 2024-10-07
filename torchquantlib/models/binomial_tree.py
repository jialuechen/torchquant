import torch
from torch import Tensor

def binomial_tree(option_type: str, option_style: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """
    Price an option using the binomial tree model.

    This function implements a binomial tree model for pricing European and American options.
    It constructs a tree of possible asset prices and then works backwards to determine the option price.

    Args:
        option_type (str): Type of option, either 'call' or 'put'.
        option_style (str): Style of option, either 'european' or 'american'.
        spot (Tensor): Current spot price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate.
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The calculated option price.
    """
    # Calculate time step and up/down factors
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (torch.exp(rate * dt) - d) / (u - d)  # Risk-neutral probability

    # Construct the price tree
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
            # Calculate the expected option value
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            
            # For American options, check for early exercise
            if option_style == 'american':
                if option_type == 'call':
                    value_tree[j] = torch.maximum(value_tree[j], price_tree[j, i] - strike)
                elif option_type == 'put':
                    value_tree[j] = torch.maximum(value_tree[j], strike - price_tree[j, i])

    # Return the option price (root of the tree)
    return value_tree[0]