import torch
from torchquantlib.core.asset_pricing.letsberational import implied_volatility

market_price = torch.tensor(10.0)
spot = torch.tensor(100.0)
strike = torch.tensor(105.0)
expiry = torch.tensor(1.0)
rate = torch.tensor(0.05)
dividend = torch.tensor(0.02)

iv = implied_volatility('call', 'european', market_price, spot, strike, expiry, rate, dividend)
print(f'Implied Volatility: {iv.item()}')