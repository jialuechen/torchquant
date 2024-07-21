import torch
from torch import Tensor

def malliavin_greek(option_price: Tensor, underlying_price: Tensor, volatility: Tensor, expiry: Tensor) -> Tensor:
    greek = option_price * underlying_price * volatility * torch.sqrt(expiry)
    return greek