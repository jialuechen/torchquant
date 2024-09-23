import torch
from torchderiv.core.asset_pricing.bond_pricing.putable_bond import putable_bond

face_value = 1000.0
coupon_rate = 0.06
rate = 0.05
periods = 10
put_price = 950.0
put_period = 5

price = putable_bond(face_value, coupon_rate, rate, periods, put_price, put_period)
print(f'Putable Bond Price: {price}')