import torch
from torch import Tensor

def calculate_cva(exposure: Tensor, default_prob: Tensor, recovery_rate: Tensor) -> Tensor:
    cva = exposure * (1 - recovery_rate) * default_prob
    return cva