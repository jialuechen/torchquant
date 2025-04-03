import torch
from torch import Tensor

def asian_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    """
    Price an arithmetic average Asian option using a modified binomial tree model.

    This function implements a binomial tree approach to price Asian options,
    which are path-dependent options where the payoff depends on the average
    price of the underlying asset over a specified period.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.

    Returns:
        Tensor: The price of the arithmetic average Asian option.

    Note:
        This implementation uses a simplified approach for Asian options within
        the binomial tree framework. It approximates the average at each node
        using only the current and next time step prices. For more accurate
        pricing of Asian options, more sophisticated methods like Monte Carlo
        simulation are often preferred.
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

    # Backward induction through the tree (vectorized)
    for i in range(steps - 1, -1, -1):
        # Compute the average price of adjacent nodes
        avg_prices = (price_tree[:i+1, i+1] + price_tree[1:i+2, i+1]) / 2
        # Update the expected option value using risk-neutral probability p and discount factor
        value_tree[:i+1] = (p * value_tree[:i+1] + (1 - p) * value_tree[1:i+2]) * torch.exp(-rate * dt)
        # For call/put options, compare with early exercise payoff
        if option_type == 'call':
            value_tree[:i+1] = torch.maximum(value_tree[:i+1], avg_prices - strike)
        elif option_type == 'put':
            value_tree[:i+1] = torch.maximum(value_tree[:i+1], strike - avg_prices)

    # The option price is the value at the root of the tree
    return value_tree[0]