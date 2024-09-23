import torch
from torch import Tensor

def fixed_income_forward(face_value: Tensor, rate: Tensor, time_to_maturity: Tensor, forward_rate: Tensor) -> Tensor:
    return face_value * torch.exp((forward_rate - rate) * time_to_maturity)