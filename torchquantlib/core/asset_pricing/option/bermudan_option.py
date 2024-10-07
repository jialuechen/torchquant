import torch
from torch import Tensor

def bermudan_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int, exercise_dates: Tensor) -> Tensor:
    """
    Price a Bermudan option using a binomial tree model.

    This function implements a binomial tree approach to price Bermudan options,
    which are options that can be exercised on specific dates before expiration.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.
        exercise_dates (Tensor): Tensor of indices representing the steps at which the option can be exercised.

    Returns:
        Tensor: The price of the Bermudan option.

    Note:
        This implementation combines features of European and American options.
        It allows for early exercise, but only on specified dates.
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
            # Check for early exercise if current step is an exercise date
            if i in exercise_dates:
                if option_type == 'call':
                    value_tree[j] = torch.maximum(value_tree[j], price_tree[j, i] - strike)
                elif option_type == 'put':
                    value_tree[j] = torch.maximum(value_tree[j], strike - price_tree[j, i])
    
    # The option price is the value at the root of the tree
    return value_tree[0]