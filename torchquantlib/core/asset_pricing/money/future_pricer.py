import torch
from torch import Tensor

def future_pricer(spot: Tensor, domestic_rate: Tensor, foreign_rate: Tensor, expiry: Tensor) -> Tensor:
    """
    Calculate the theoretical price of a futures contract.

    This function computes the fair price of a futures contract using the
    cost-of-carry model. It assumes continuous compounding and no dividends
    or storage costs.

    Args:
        spot (Tensor): The current spot price of the underlying asset.
        domestic_rate (Tensor): The domestic risk-free interest rate (as a decimal).
        foreign_rate (Tensor): The foreign risk-free interest rate (as a decimal).
        expiry (Tensor): The time to expiration of the futures contract (in years).

    Returns:
        Tensor: The theoretical futures price.

    Formula:
        Futures Price = Spot Price * e^(domestic_rate - foreign_rate * time to expiry)

    Note:
        - This model assumes perfect markets with no transaction costs or taxes.
        - For commodities or dividend-paying stocks, additional factors would 
          need to be considered in the pricing model.
    """
    return spot * torch.exp((domestic_rate - foreign_rate) * expiry)