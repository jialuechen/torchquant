import torch
from torch import Tensor

def calculate_var(returns: Tensor, confidence_level: float) -> Tensor:
    sorted_returns = torch.sort(returns)[0]
    index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    return var