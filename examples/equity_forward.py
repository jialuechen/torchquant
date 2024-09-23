import torch
from torchquant.core.asset_pricing.equity_forward import equity_forward

spot = torch.tensor(100.0)
rate = torch.tensor(0.05)
dividend_yield = torch.tensor(0.02)
expiry = torch.tensor(1.0)

price = equity_forward(spot, rate, dividend_yield, expiry)
print(f'Equity Forward Price: {price.item()}')