import torch
from torchquantlib.core.asset_pricing.option_pricing.binomial_tree import binomial_tree

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = binomial_tree('call', 'european', spot, strike, expiry, volatility, rate, steps)
print(f'Option Price: {price.item()}')