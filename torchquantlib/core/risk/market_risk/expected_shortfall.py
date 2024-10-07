from torch import Tensor
from var import calculate_var

def calculate_es(returns: Tensor, confidence_level: float) -> Tensor:
    """
    Calculate Expected Shortfall (ES), also known as Conditional Value at Risk (CVaR).

    Expected Shortfall measures the expected loss given that the loss is greater than
    the Value at Risk (VaR). It provides a more conservative risk measure than VaR
    by considering the tail risk beyond the VaR threshold.

    Args:
        returns (Tensor): A tensor of historical returns for the asset or portfolio.
        confidence_level (float): The confidence level for ES calculation, typically 0.95 or 0.99.

    Returns:
        Tensor: The calculated Expected Shortfall.

    Note:
        - This function first calculates VaR using the calculate_var function.
        - ES is always greater than or equal to VaR for the same confidence level.
        - ES gives a better understanding of the tail risk compared to VaR.
        - A higher confidence level results in a more conservative (higher) ES estimate.
    """
    # Calculate Value at Risk (VaR) for the given confidence level
    var = calculate_var(returns, confidence_level)

    # Calculate the mean of returns that are less than or equal to the negative VaR
    # This represents the average loss in the worst (1 - confidence_level) * 100% of cases
    shortfall = returns[returns <= -var].mean()

    # Return the negative of the shortfall as ES represents a loss
    return -shortfall