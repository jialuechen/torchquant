import torch
from torchquantlib.risk.greeks.malliavin import malliavin_greek

def test_malliavin_greek():
    option_price = torch.tensor(10.0)
    underlying_price = torch.tensor(100.0)
    volatility = torch.tensor(0.2)
    expiry = torch.tensor(1.0)

    greek = malliavin_greek(option_price, underlying_price, volatility, expiry)
    assert torch.isclose(greek, torch.tensor(2.0), atol=1e-1)