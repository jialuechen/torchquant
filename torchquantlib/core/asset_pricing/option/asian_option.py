"""
Asian Option Pricing Module
This module provides functionality for pricing Asian options using Monte Carlo simulation.
"""
import torch
from torch import Tensor

def asian_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, num_paths: int, num_steps: int, average_type: str = "arithmetic") -> Tensor:
    """
    Price an Asian option using Monte Carlo simulation.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        num_paths (int): Number of Monte Carlo simulation paths.
        num_steps (int): Number of time steps in the simulation.
        average_type (str): Type of average - 'arithmetic' or 'geometric'. Defaults to 'arithmetic'.

    Returns:
        Tensor: The price of the Asian option.
    """
    if average_type not in ["arithmetic", "geometric"]:
        raise ValueError(f"Unsupported average_type: {average_type}. Must be 'arithmetic' or 'geometric'.")

    # Time step size
    dt = expiry / num_steps

    # Simulate asset price paths
    z = torch.randn(num_paths, num_steps, device=spot.device)  # Standard normal random variables
    price_paths = torch.zeros(num_paths, num_steps + 1, device=spot.device)
    price_paths[:, 0] = spot

    for t in range(1, num_steps + 1):
        price_paths[:, t] = price_paths[:, t - 1] * torch.exp(
            (rate - 0.5 * volatility**2) * dt + volatility * torch.sqrt(dt) * z[:, t - 1]
        )

    # Calculate average prices
    if average_type == "arithmetic":
        average_prices = price_paths[:, 1:].mean(dim=1)  # Exclude the initial price
    elif average_type == "geometric":
        average_prices = torch.exp(torch.log(price_paths[:, 1:]).mean(dim=1))  # Exclude the initial price

    # Calculate payoffs
    if option_type == "call":
        payoffs = torch.maximum(average_prices - strike, torch.tensor(0.0, device=spot.device))
    elif option_type == "put":
        payoffs = torch.maximum(strike - average_prices, torch.tensor(0.0, device=spot.device))
    else:
        raise ValueError(f"Unsupported option_type: {option_type}. Must be 'call' or 'put'.")

    # Discount payoffs to present value
    discounted_payoffs = payoffs * torch.exp(-rate * expiry)

    # Return the mean of the discounted payoffs as the option price
    return discounted_payoffs.mean()