import torch
from torchquantlib.core.asset_pricing.option.exotics import barrier_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
barrier = torch.tensor(110.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = barrier_option('call', 'up-and-out', spot, strike, barrier, expiry, volatility, rate, steps)
print(f'Option Price: {price.item()}')