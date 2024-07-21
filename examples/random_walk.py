import torch
from quantorch.utils.math_tools import random_walk

spot = torch.tensor(100.0)
expiry = torch.tensor(1.0)
volatility = torch.tensor(0.2)
rate = torch.tensor(0.05)
num_steps = 365

walk = random_walk(spot, expiry, volatility, rate, num_steps)
print(walk)