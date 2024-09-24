import torch
from torchquantlib.core.asset_pricing.bond_pricing.convertible_bond import convertible_bond

face_value = 1000.0
coupon_rate = 0.06
rate = 0.05
periods = 10
conversion_ratio = 1.1
conversion_price = 100.0

price = convertible_bond(face_value, coupon_rate, rate, periods, conversion_ratio, conversion_price)
print(f'Convertible Bond Price: {price}')