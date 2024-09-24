import torch
from torch import Tensor

def calculate_mva(exposure: Tensor, funding_cost: Tensor, maturity: Tensor) -> Tensor:
    mva = exposure * funding_cost * maturity
    return mva