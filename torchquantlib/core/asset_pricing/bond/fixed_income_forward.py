import torch
from torch import Tensor

def fixed_income_forward(face_value: Tensor, rate: Tensor, time_to_maturity: Tensor, forward_rate: Tensor) -> Tensor:
    """
    Calculate the forward price of a fixed income security.

    This function computes the forward price of a fixed income security using the
    continuous compounding formula. It's based on the principle that the forward
    price should reflect the difference between the current interest rate and the
    forward rate over the time to maturity.

    Args:
        face_value (Tensor): The face value (or notional amount) of the fixed income security.
        rate (Tensor): The current interest rate (as a decimal).
        time_to_maturity (Tensor): The time to maturity of the forward contract (in years).
        forward_rate (Tensor): The forward interest rate (as a decimal).

    Returns:
        Tensor: The calculated forward price of the fixed income security.

    Formula:
        Forward Price = Face Value * e^((Forward Rate - Current Rate) * Time to Maturity)

    Note:
        This function assumes continuous compounding. For discrete compounding,
        a different formula would be required.
    """
    return face_value * torch.exp((forward_rate - rate) * time_to_maturity)