"""
Asian Option Pricing Module
This module provides functionality for pricing Asian options using a binomial tree model.
"""
import torch
from torch import Tensor
from .utils import calculate_binomial_tree_params, backward_induction

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
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)
    return backward_induction(option_type, price_tree, strike, rate, dt, p, steps)