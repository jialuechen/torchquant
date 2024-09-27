from torch import tensor

def zero_coupon_bond(face_value: tensor, rate: tensor, maturity: tensor) -> tensor:
    return face_value / (1 + rate)**maturity

def coupon_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor) -> tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def callable_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, call_price: tensor, call_period: tensor) -> tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        if t == call_period:
            bond_price += min(face_value + coupon * t, call_price) / (1 + rate) ** t
        else:
            bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def putable_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, put_price: tensor, put_period:tensor) -> tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        if t == put_period:
            bond_price += max(face_value + coupon * t, put_price) / (1 + rate) ** t
        else:
            bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    return bond_price

def convertible_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor, conversion_ratio: tensor, conversion_price: tensor) -> tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate) ** t
    bond_price += face_value / (1 + rate) ** periods
    conversion_value = conversion_ratio * conversion_price
    return min(bond_price, conversion_value)

def stochastic_rate_bond(face_value: tensor, coupon_rate: tensor, rate: tensor, periods: tensor) -> tensor:
    coupon = face_value * coupon_rate
    bond_price = 0.0
    for t in range(1, periods + 1):
        bond_price += coupon / (1 + rate[t-1]) ** t
    bond_price += face_value / (1 + rate[-1]) ** periods
    return bond_price