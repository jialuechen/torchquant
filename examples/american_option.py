import torch
from torchderiv.core.asset_pricing.option_pricing.american_option import american_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = american_option('call', spot, strike, expiry, volatility, rate, steps)
print(f'Option Price: {price.item()}')