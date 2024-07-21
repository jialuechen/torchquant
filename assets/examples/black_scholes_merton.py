import torch
from quantorch.core.asset_pricing.option_pricing.black_scholes_merton import black_scholes_merton

spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
dividend = torch.tensor(0.02)

price = black_scholes_merton('call', 'european', spot, strike, expiry, volatility, rate, dividend)
print(f'Option Price: {price.item()}')