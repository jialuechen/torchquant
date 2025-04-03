import torch
from torch import Tensor

def calculate_binomial_tree_params(expiry: Tensor, volatility: Tensor, rate: Tensor, steps: int):
    dt = expiry / steps
    u = torch.exp(volatility * torch.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (torch.exp(rate * dt) - d) / (u - d)  # Risk-neutral probability
    return dt, u, d, p

def backward_induction(option_type: str, price_tree: Tensor, strike: Tensor, rate: Tensor, dt: Tensor, p: Tensor, steps: int, exercise_dates: Tensor = None):
    value_tree = torch.zeros_like(price_tree)
    if option_type == 'call':
        value_tree[:, steps] = torch.maximum(price_tree[:, steps] - strike, torch.tensor(0.0))
    elif option_type == 'put':
        value_tree[:, steps] = torch.maximum(strike - price_tree[:, steps], torch.tensor(0.0))

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            value_tree[j, i] = (p * value_tree[j, i + 1] + (1 - p) * value_tree[j + 1, i + 1]) * torch.exp(-rate * dt)
            if exercise_dates is not None and i in exercise_dates:
                if option_type == 'call':
                    value_tree[j, i] = torch.maximum(value_tree[j, i], price_tree[j, i] - strike)
                elif option_type == 'put':
                    value_tree[j, i] = torch.maximum(value_tree[j, i], strike - price_tree[j, i])
    return value_tree[0, 0]
