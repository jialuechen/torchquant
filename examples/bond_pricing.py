import torch
from torchquant.core.asset_pricing.bond_pricing.zero_coupon_bond import zero_coupon_bond
from torchquant.core.asset_pricing.bond_pricing.coupon_bond import coupon_bond
from torchquant.core.asset_pricing.bond_pricing.stochastic_rate_bond import stochastic_rate_bond

face_value = 1000.0
rate = 0.05
maturity = 5
coupon_rate = 0.06
periods = 10

# Zero Coupon Bond
zcb_price = zero_coupon_bond(face_value, rate, maturity)
print(f'Zero Coupon Bond Price: {zcb_price}')

# Coupon Bond
cb_price = coupon_bond(face_value, coupon_rate, rate, periods)
print(f'Coupon Bond Price: {cb_price}')

# Stochastic Rate Bond
rate_tensor = torch.tensor([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095])
srb_price = stochastic_rate_bond(face_value, coupon_rate, rate_tensor, periods)
print(f'Stochastic Rate Bond Price: {srb_price}')