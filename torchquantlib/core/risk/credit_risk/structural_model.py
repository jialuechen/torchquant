import torch
from torch import Tensor

def merton_model(asset_value: Tensor, debt: Tensor, volatility: Tensor, rate: Tensor, expiry: Tensor) -> Tensor:
    """
    Implement the Merton structural model for credit risk.

    This function calculates the equity value of a firm using Merton's model,
    which treats the equity as a call option on the firm's assets.

    Args:
        asset_value (Tensor): Current market value of the firm's assets
        debt (Tensor): Face value of the firm's debt (assumed to be a zero-coupon bond)
        volatility (Tensor): Volatility of the firm's asset returns
        rate (Tensor): Risk-free interest rate
        expiry (Tensor): Time to maturity of the debt / Time horizon for the model

    Returns:
        Tensor: Equity value of the firm

    Note:
        - This model assumes that the firm's debt is a single zero-coupon bond.
        - The model uses the Black-Scholes-Merton formula, treating equity as a call option.
        - Default occurs if the asset value falls below the face value of debt at maturity.
    """
    # Calculate d1 and d2 parameters (similar to Black-Scholes model)
    d1 = (torch.log(asset_value / debt) + (rate + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    d2 = d1 - volatility * torch.sqrt(expiry)

    # Calculate the equity value using the Black-Scholes-Merton formula
    # Equity is treated as a call option on the firm's assets
    equity_value = asset_value * torch.distributions.Normal(0, 1).cdf(d1) - \
                   debt * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(d2)

    return equity_value