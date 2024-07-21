import torch
from torch import Tensor

def merton_model(asset_value: Tensor, debt: Tensor, volatility: Tensor, rate: Tensor, expiry: Tensor) -> Tensor:
    d1 = (torch.log(asset_value / debt) + (rate + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    d2 = d1 - volatility * torch.sqrt(expiry)
    equity_value = asset_value * torch.distributions.Normal(0, 1).cdf(d1) - debt * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(d2)
    return equity_value