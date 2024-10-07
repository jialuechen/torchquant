import torch
from torch import Tensor

def american_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """
    Price an American option using the binomial tree model.

    This function implements the Cox-Ross-Rubinstein binomial tree model to price
    American call or put options, which can be exercised at any time up to expiration.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the American option.

    Note:
        This implementation uses a backward induction approach to account for the
        possibility of early exercise at each node of the tree.
    """
    # Calculate parameters for the binomial model
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (torch.exp(rate * dt) - d) / (u - d)  # Risk-neutral probability
    
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
            # Calculate the expected option value
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            # Check for early exercise
            if option_type == 'call':
                value_tree[j] = torch.maximum(value_tree[j], price_tree[j, i] - strike)
            elif option_type == 'put':
                value_tree[j] = torch.maximum(value_tree[j], strike - price_tree[j, i])
    
    # The option price is the value at the root of the tree
    return value_tree[0]