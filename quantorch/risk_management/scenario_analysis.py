import torch
from torch import Tensor

def scenario_analysis(portfolio_value: Tensor, scenarios: Tensor) -> Tensor:
    scenario_values = portfolio_value * scenarios
    return scenario_values