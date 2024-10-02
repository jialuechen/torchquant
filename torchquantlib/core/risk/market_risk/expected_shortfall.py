import torch
from torch import Tensor

def calculate_es(returns: Tensor, confidence_level: float) -> Tensor:
    var = calculate_var(returns, confidence_level)
    shortfall = returns[returns <= -var].mean()
    return -shortfall