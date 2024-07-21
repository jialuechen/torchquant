import torch
from torch import Tensor

def random_walk(spot: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, num_steps: int) -> Tensor:
    dt = expiry / num_steps
    path = torch.zeros(num_steps)
    path[0] = spot

    for t in range(1, num_steps):
        z = torch.randn(1)
        path[t] = path[t-1] * torch.exp((rate - 0.5 * volatility**2) * dt + volatility * torch.sqrt(dt) * z)

    return path

def bsm_greeks(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, dividend: Tensor) -> dict:
    d1 = (torch.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * expiry) / (volatility * torch.sqrt(expiry))
    d2 = d1 - volatility * torch.sqrt(expiry)

    delta = torch.exp(-dividend * expiry) * torch.distributions.Normal(0, 1).cdf(d1)
    gamma = torch.exp(-dividend * expiry) * torch.distributions.Normal(0, 1).pdf(d1) / (spot * volatility * torch.sqrt(expiry))
    theta = (-spot * torch.exp(-dividend * expiry) * torch.distributions.Normal(0, 1).pdf(d1) * volatility / (2 * torch.sqrt(expiry))
            - rate * strike * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(d2))
    vega = spot * torch.exp(-dividend * expiry) * torch.sqrt(expiry) * torch.distributions.Normal(0, 1).pdf(d1)
    rho = strike * expiry * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(d2)

    if option_type == 'put':
        delta = delta - 1
        theta = theta - rate * spot * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(-d1)
        rho = -strike * expiry * torch.exp(-rate * expiry) * torch.distributions.Normal(0, 1).cdf(-d2)

    greeks = {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    return greeks