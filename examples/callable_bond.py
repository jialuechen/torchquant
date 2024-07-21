import torch
from quantorch.core.asset_pricing.bond_pricing.callable_bond import callable_bond

face_value = 1000.0
coupon_rate = 0.06
rate = 0.05
periods = 10
call_price = 1050.0
call_period = 5

price = callable_bond(face_value, coupon_rate, rate, periods, call_price, call_period)
print(f'Callable Bond Price: {price}')