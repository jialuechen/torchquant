import torch
from torch import Tensor

def future_pricer(spot: Tensor, rate: Tensor, expiry: Tensor) -> Tensor:
    return spot * torch.exp(rate * expiry)