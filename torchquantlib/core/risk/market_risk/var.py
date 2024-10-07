import torch
from torch import Tensor

def calculate_var(returns: Tensor, confidence_level: float) -> Tensor:
    """
    Calculate Value at Risk (VaR) using the historical simulation method.

    Value at Risk is a measure of the potential loss in value of a risky asset or portfolio
    over a defined period for a given confidence interval.

    Args:
        returns (Tensor): A tensor of historical returns for the asset or portfolio.
        confidence_level (float): The confidence level for VaR calculation, typically 0.95 or 0.99.

    Returns:
        Tensor: The calculated Value at Risk.

    Note:
        - This function assumes that the input returns are properly preprocessed and represent
          a relevant historical period for the asset or portfolio.
        - The calculated VaR represents the loss that is expected to be exceeded only
          (1 - confidence_level) * 100% of the time.
        - A higher confidence level results in a more conservative (higher) VaR estimate.
    """
    # Sort the returns in ascending order
    sorted_returns = torch.sort(returns)[0]

    # Calculate the index corresponding to the VaR quantile
    index = int((1 - confidence_level) * len(sorted_returns))

    # The VaR is the negative of the return at the calculated index
    # We use the negative because VaR represents a loss
    var = -sorted_returns[index]

    return var