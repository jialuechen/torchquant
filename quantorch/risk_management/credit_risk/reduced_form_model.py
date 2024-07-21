import torch
from torch import Tensor

def reduced_form_model(lambda_0: Tensor, default_intensity: Tensor, recovery_rate: Tensor, time: Tensor) -> Tensor:
    survival_prob = torch.exp(-default_intensity * time)
    expected_loss = (1 - recovery_rate) * (1 - survival_prob)
    return expected_loss