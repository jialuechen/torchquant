import pytest
import torch
from torchquantlib.core.asset_pricing.money.future_pricer import future_pricer

@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor([100.0, 150.0, 200.0]),
        'rate': torch.tensor([0.05, 0.06, 0.07]),
        'expiry': torch.tensor([1.0, 2.0, 3.0])
    }

def test_future_pricer(setup_tensors):
    result = future_pricer(**setup_tensors)
    expected = setup_tensors['spot'] * torch.exp(setup_tensors['rate'] * setup_tensors['expiry'])
    assert torch.allclose(result, expected, rtol=1e-5)

def test_future_pricer_zero_rate(setup_tensors):
    setup_tensors['rate'] = torch.tensor([0.0, 0.0, 0.0])
    result = future_pricer(**setup_tensors)
    assert torch.allclose(result, setup_tensors['spot'], rtol=1e-5)

def test_future_pricer_zero_expiry(setup_tensors):
    setup_tensors['expiry'] = torch.tensor([0.0, 0.0, 0.0])
    result = future_pricer(**setup_tensors)
    assert torch.allclose(result, setup_tensors['spot'], rtol=1e-5)

def test_future_pricer_negative_rate(setup_tensors):
    setup_tensors['rate'] = torch.tensor([-0.01, -0.02, -0.03])
    result = future_pricer(**setup_tensors)
    expected = setup_tensors['spot'] * torch.exp(setup_tensors['rate'] * setup_tensors['expiry'])
    assert torch.allclose(result, expected, rtol=1e-5)

def test_future_pricer_large_expiry(setup_tensors):
    setup_tensors['expiry'] = torch.tensor([10.0, 20.0, 30.0])
    result = future_pricer(**setup_tensors)
    expected = setup_tensors['spot'] * torch.exp(setup_tensors['rate'] * setup_tensors['expiry'])
    assert torch.allclose(result, expected, rtol=1e-5)

def test_future_pricer_scalar_inputs():
    spot = torch.tensor(100.0)
    rate = torch.tensor(0.05)
    expiry = torch.tensor(1.0)
    result = future_pricer(spot, rate, expiry)
    expected = spot * torch.exp(rate * expiry)
    assert torch.isclose(result, expected, rtol=1e-5)
