import torch
from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton

def test_black_scholes_merton():
    spot = torch.tensor(100.0)
    strike = torch.tensor(105.0)
    expiry = torch.tensor(1.0)
    volatility = torch.tensor(0.2)
    rate = torch.tensor(0.05)
    dividend = torch.tensor(0.02)

    price = black_scholes_merton('call', 'european', spot, strike, expiry, volatility, rate, dividend)
    assert torch.isclose(price, torch.tensor(5.9436), atol=1e-4)