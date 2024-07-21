import torch
from torch import Tensor

def calculate_fva(exposure: Tensor, funding_spread: Tensor, maturity: Tensor) -> Tensor:
    fva = exposure * funding_spread * maturity
    return fva