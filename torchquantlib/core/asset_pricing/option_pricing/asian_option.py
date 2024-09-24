import torch
from torch import Tensor

def asian_option(option_type: str, spot: Tensor, strike: Tensor, expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int) -> Tensor:
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
            average_price = torch.mean(price_tree[j:j + 2, i + 1])
            if option_type == 'call':
                value_tree[j] = torch.maximum(value_tree[j], average_price - strike)
            elif option_type == 'put':
                value_tree[j] = torch.maximum(value_tree[j], strike - average_price)

    return value_tree[0]