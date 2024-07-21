import torch
from torch import Tensor

def stress_test(portfolio_value: Tensor, stress_scenarios: Tensor) -> Tensor:
    stressed_values = portfolio_value * (1 + stress_scenarios)
    return stressed_values