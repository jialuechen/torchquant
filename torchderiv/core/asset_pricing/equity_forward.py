import torch
from torch import Tensor

def equity_forward(spot: Tensor, rate: Tensor, dividend_yield: Tensor, expiry: Tensor) -> Tensor:
    return spot * torch.exp((rate - dividend_yield) * expiry)