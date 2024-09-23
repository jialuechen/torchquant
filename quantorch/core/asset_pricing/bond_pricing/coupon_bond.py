import torch
from torch import Tensor

def coupon_bond(face_value: float, coupon_rate: float, rate: float, periods: int) -> float:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price