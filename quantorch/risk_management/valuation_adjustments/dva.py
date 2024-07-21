import torch
from torch import Tensor

def calculate_dva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    dva = exposure * (1 - recovery_rate) * default_prob
    return dva