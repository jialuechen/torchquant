"""
Bermudan Option Pricing Module
This module provides functionality for pricing Bermudan options using a binomial tree model.
"""
import torch
from torch import Tensor
from .utils import calculate_binomial_tree_params, backward_induction

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
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    
    # Construct the price tree using tensor operations
    indices = torch.arange(steps + 1, dtype=torch.float32)
    i, j = torch.meshgrid(indices, indices, indexing='ij')
    mask = j <= i  # Only keep valid positions in the tree
    price_tree = spot * (u ** (i - j)) * (d ** j) * mask

    # Perform backward induction
    return backward_induction(option_type, price_tree, strike, rate, dt, p, steps, exercise_dates)