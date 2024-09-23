import torch
from torchquant.calibration.heston_calibration import calibrate_heston

market_prices = torch.tensor([10.0, 12.0, 14.0, 16.0])
strikes = torch.tensor([100.0, 105.0, 110.0, 115.0])
expiries = torch.tensor([1.0, 1.0, 1.0, 1.0])
spot = torch.tensor(100.0)
rate = torch.tensor(0.05)

params = calibrate_heston(market_prices, strikes, expiries, spot, rate)
print(f'Calibrated Heston Parameters: {params}')