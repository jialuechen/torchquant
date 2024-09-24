import torch
from torch import Tensor

def stochastic_rate_bond(face_value: float, coupon_rate: float, rate: Tensor, periods: int) -> Tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate[t-1]) ** t
    bond_price += face_value / (1 + rate[-1]) ** periods
    return bond_price