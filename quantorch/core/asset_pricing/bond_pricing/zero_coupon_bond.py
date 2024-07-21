import torch
from torch import Tensor

def zero_coupon_bond(face_value: float, rate: float, maturity: float) -> float:
    return face_value / (1 + rate)**maturity