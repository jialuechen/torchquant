import torch
from torchquantlib.models.local_volatility.dupire_local_volatility import DupireLocalVol

spots = torch.linspace(80, 120, 10)
strikes = torch.linspace(80, 120, 10)
expiries = torch.linspace(0.1, 2.0, 10)
implied_vols = torch.rand(10, 10) * 0.3

model = DupireLocalVol(spots, strikes, expiries, implied_vols)
local_vol = model.local_vol(100, 100, 1.0)
print(f'Local Volatility: {local_vol}')