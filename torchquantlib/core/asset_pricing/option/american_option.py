import torch
from torch import Tensor
from .utils import calculate_binomial_tree_params, backward_induction

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
    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)
    return backward_induction(option_type, price_tree, strike, rate, dt, p, steps)