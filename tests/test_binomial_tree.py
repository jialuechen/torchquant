import torch
from torchquantlib.models.monte_carlo import MonteCarlo

def test_monte_carlo():
    spot = torch.tensor(100.0)
    strike = torch.tensor(105.0)
    expiry = torch.tensor(1.0)
    volatility = torch.tensor(0.2)
    rate = torch.tensor(0.05)
    num_paths = 100000
    num_steps = 365

    mc = MonteCarlo(spot, strike, expiry, volatility, rate, num_paths, num_steps)
    price = mc.simulate()
    assert torch.isclose(price, torch.tensor(10.4506), atol=1e-2)