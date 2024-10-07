import torch
from torch import Tensor

def bootstrap_yield_curve(cash_flows: Tensor, prices: Tensor) -> Tensor:
    """
    Construct a yield curve using the bootstrap method.

    The bootstrap method iteratively calculates yields for each maturity,
    using the previously calculated yields for shorter maturities.

    Args:
        cash_flows (Tensor): A 2D tensor where each row represents the cash flows for an instrument.
        prices (Tensor): A 1D tensor of market prices for each instrument.

    Returns:
        Tensor: A 1D tensor of yields corresponding to each maturity.

    Note:
        This method assumes that the cash flows and prices are sorted by maturity.
        It also assumes a simple structure where each instrument has a single cash flow at maturity.
    """
    n = len(prices)
    yields = torch.zeros(n)
    for i in range(n):
        sum_cfs = torch.sum(cash_flows[:i + 1])
        # Calculate yield to maturity assuming a single cash flow at maturity
        yields[i] = (sum_cfs / prices[i]) ** (1 / (i + 1)) - 1
    return yields

def nelson_siegel_yield_curve(tau: Tensor, beta0: Tensor, beta1: Tensor, beta2: Tensor) -> Tensor:
    """
    Construct a yield curve using the Nelson-Siegel model.

    The Nelson-Siegel model is a parametric method for modeling the yield curve,
    using four parameters to generate a smooth and flexible curve shape.

    Args:
        tau (Tensor): A 1D tensor of time to maturities.
        beta0 (Tensor): Long-term interest rate level parameter.
        beta1 (Tensor): Short-term interest rate parameter.
        beta2 (Tensor): Medium-term interest rate parameter.

    Returns:
        Tensor: A 1D tensor of yields corresponding to each maturity in tau.

    Note:
        This implementation uses a simplified version of the Nelson-Siegel model
        where the decay parameter lambda is assumed to be 1.
    """
    n = len(tau)
    yields = torch.zeros(n)
    for i in range(n):
        t = tau[i]
        # Nelson-Siegel yield curve formula
        yields[i] = beta0 + (beta1 + beta2) * (1 - torch.exp(-t)) / t - beta2 * torch.exp(-t)
    return yields