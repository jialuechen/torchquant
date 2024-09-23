import torch
from torchquant.core.asset_pricing.currency_pricing import currency_price

domestic_rate = torch.tensor(0.05)
foreign_rate = torch.tensor(0.03)
spot = torch.tensor(1.0)
expiry = torch.tensor(1.0)

price = currency_price(domestic_rate, foreign_rate, spot, expiry)
print(f'Currency Price: {price.item()}')