import torch
from quantorch.core.asset_pricing.option_pricing.advanced_exotic_options import lookback_option

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
steps = 100

price = lookback_option('call', spot, strike, expiry, volatility, rate, steps)
print(f'Lookback Option Price: {price.item()}')