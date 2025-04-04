import torch
from torch import Tensor

def american_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, num_paths: int, num_steps: int) -> Tensor:
    """
    Price an American option using Monte Carlo simulation with the Longstaff-Schwartz method.

    Args:
        option_type (str): Type of option - either 'call' or 'put'.
        spot (Tensor): Current price of the underlying asset.
        strike (Tensor): Strike price of the option.
        expiry (Tensor): Time to expiration in years.
        volatility (Tensor): Volatility of the underlying asset.
        rate (Tensor): Risk-free interest rate (annualized).
        num_paths (int): Number of Monte Carlo simulation paths.
        num_steps (int): Number of time steps in the simulation.

    Returns:
        Tensor: The price of the American option.
    """
    if option_type not in ["call", "put"]:
        raise ValueError(f"Unsupported option_type: {option_type}. Must be 'call' or 'put'.")

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

    # Calculate payoffs at maturity
    if option_type == "call":
        payoffs = torch.maximum(price_paths[:, -1] - strike, torch.tensor(0.0, device=spot.device))
    elif option_type == "put":
        payoffs = torch.maximum(strike - price_paths[:, -1], torch.tensor(0.0, device=spot.device))

    # Discount payoffs back to present value
    discounted_payoffs = payoffs * torch.exp(-rate * expiry)

    # Backward induction using Longstaff-Schwartz method
    for t in range(num_steps - 1, 0, -1):
        in_the_money = (price_paths[:, t] > strike) if option_type == "call" else (price_paths[:, t] < strike)
        if in_the_money.any():
            # Select paths that are in the money
            x = price_paths[in_the_money, t]
            y = discounted_payoffs[in_the_money] * torch.exp(rate * dt * (num_steps - t))

            # Fit regression model (polynomial basis)
            A = torch.stack([torch.ones_like(x), x, x**2], dim=1)  # Basis functions: 1, x, x^2
            coeffs = torch.linalg.lstsq(A, y).solution

            # Estimate continuation value
            continuation_value = (A @ coeffs).squeeze()

            # Update discounted payoffs
            exercise_value = (x - strike) if option_type == "call" else (strike - x)
            discounted_payoffs[in_the_money] = torch.where(
                exercise_value > continuation_value,
                exercise_value,
                discounted_payoffs[in_the_money] * torch.exp(rate * dt)
            )

    # Return the mean of the discounted payoffs as the option price
    return discounted_payoffs.mean()