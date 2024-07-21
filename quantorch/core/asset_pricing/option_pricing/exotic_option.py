import torch
from torch import Tensor

def barrier_option(option_type: str, barrier_type: str, spot: Tensor, strike: Tensor, barrier: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))
    d = 1 / u
    p = (torch.exp(rate * dt) - d) / (u - d)

    price_tree = torch.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

    if option_type == 'call':
        value_tree = torch.maximum(price_tree[:, steps] - strike, torch.tensor(0.0))
    elif option_type == 'put':
        value_tree = torch.maximum(strike - price_tree[:, steps], torch.tensor(0.0))

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j] = (p * value_tree[j] + (1 - p) * value_tree[j + 1]) * torch.exp(-rate * dt)
            if barrier_type == 'up-and-out' and price_tree[j, i] >= barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'down-and-out' and price_tree[j, i] <= barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'up-and-in' and price_tree[j, i] < barrier:
                value_tree[j] = torch.tensor(0.0)
            elif barrier_type == 'down-and-in' and price_tree[j, i] > barrier:
                value_tree[j] = torch.tensor(0.0)

    return value_tree[0]