"""
Asian Option Pricing Module
This module provides functionality for pricing Asian options using a binomial tree model.
"""
import torch
from torch import Tensor
from .utils import calculate_binomial_tree_params, backward_induction

def asian_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int, average_type: str = "arithmetic") -> Tensor:
    """
    Price an Asian option using a modified binomial tree model.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        steps (int): Number of time steps in the binomial tree.
        average_type (str): Type of average - 'arithmetic' or 'geometric'.

    Returns:
        Tensor: The price of the Asian option.
    """
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    
    # Initialize tensors for prices and averages
    price_tree = torch.zeros((steps + 1, steps + 1), device=spot.device)
    avg_tree = torch.zeros_like(price_tree)
    price_tree[:, 0] = spot  # Set initial spot price for all paths
    avg_tree[:, 0] = spot

    # Build the tree dynamically using vectorized operations
    for i in range(1, steps + 1):
        up_mask = torch.arange(i + 1, device=spot.device) > 0
        down_mask = ~up_mask

        # Calculate up and down prices
        price_tree[:i + 1, i] = torch.where(
            up_mask,
            price_tree[:i, i - 1] * u,
            price_tree[:i, i - 1] * d
        )

        # Update average tree based on the selected average type
        if average_type == "arithmetic":
            avg_tree[:i + 1, i] = (avg_tree[:i, i - 1] * (i - 1) + price_tree[:i + 1, i]) / i
        elif average_type == "geometric":
            avg_tree[:i + 1, i] = torch.exp((torch.log(avg_tree[:i, i - 1]) * (i - 1) + torch.log(price_tree[:i + 1, i])) / i)
        else:
            raise ValueError(f"Unsupported average_type: {average_type}")

    # Perform backward induction using the average tree
    return backward_induction(option_type, avg_tree, strike, rate, dt, p, steps)