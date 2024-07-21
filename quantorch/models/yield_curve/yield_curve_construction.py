import torch
from torch import Tensor

def bootstrap_yield_curve(cash_flows: Tensor, prices: Tensor) -> Tensor:
    n = len(prices)
    yields = torch.zeros(n)
    for i in range(n):
        sum_cfs = torch.sum(cash_flows[:i + 1])
        yields[i] = (sum_cfs / prices[i]) ** (1 / (i + 1)) - 1
    return yields

def nelson_siegel_yield_curve(tau: Tensor, beta0: Tensor, beta1: Tensor, beta2: Tensor) -> Tensor:
    n = len(tau)
    yields = torch.zeros(n)
    for i in range(n):
        t = tau[i]
        yields[i] = beta0 + (beta1 + beta2) * (1 - torch.exp(-t)) / t - beta2 * torch.exp(-t)
    return yields