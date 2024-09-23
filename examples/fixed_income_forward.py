import torch
from torchderiv.core.asset_pricing.fixed_income_forward import fixed_income_forward

face_value = torch.tensor(1000.0)
rate = torch.tensor(0.05)
time_to_maturity = torch.tensor(1.0)
forward_rate = torch.tensor(0.06)

price = fixed_income_forward(face_value, rate, time_to_maturity, forward_rate)
print(f'Fixed Income Forward Price: {price.item()}')