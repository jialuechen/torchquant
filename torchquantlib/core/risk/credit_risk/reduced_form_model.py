import torch
from torch import Tensor

def reduced_form_model(lambda_0: Tensor, default_intensity: Tensor, recovery_rate: Tensor, time: Tensor) -> Tensor:
    """
    Implement a reduced-form model for credit risk.

    This function calculates the expected loss of a credit instrument using a simple
    reduced-form model. It assumes a constant hazard rate (default intensity) and
    a constant recovery rate.

    Args:
        lambda_0 (Tensor): Initial default intensity (not used in this simple model)
        default_intensity (Tensor): Constant hazard rate or default intensity
        recovery_rate (Tensor): Expected recovery rate in case of default (between 0 and 1)
        time (Tensor): Time horizon for the calculation

    Returns:
        Tensor: Expected loss over the given time horizon

    Note:
        - This is a simplified model. More complex models might incorporate
          time-varying default intensities or stochastic recovery rates.
        - The lambda_0 parameter is included for potential future extensions
          but is not used in the current implementation.
    """
    # Calculate the survival probability
    survival_prob = torch.exp(-default_intensity * time)

    # Calculate the expected loss
    # Expected loss = Loss given default * Probability of default
    # where Loss given default = (1 - recovery_rate)
    # and Probability of default = (1 - survival_prob)
    expected_loss = (1 - recovery_rate) * (1 - survival_prob)

    return expected_loss