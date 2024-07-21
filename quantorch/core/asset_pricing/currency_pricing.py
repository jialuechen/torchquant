import torch
from torch import Tensor

def futures_price(spot: Tensor, rate: Tensor, expiry: Tensor) -> Tensor:
    return spot * torch.exp(rate * expiry)