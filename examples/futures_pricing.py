import torch
from torchderiv.core.asset_pricing.futures_pricing import futures_price

spot = torch.tensor(100.0)
rate = torch.tensor(0.05)
expiry = torch.tensor(1.0)

price = futures_price(spot, rate, expiry)
print(f'Futures Price: {price.item()}')