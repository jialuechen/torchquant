import pytest
import torch
from torchquantlib.core.risk.credit_risk.structural_model import merton_model

@pytest.fixture
def setup_tensors():
    return {
        'asset_value': torch.tensor(100.0),
        'debt': torch.tensor(80.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'expiry': torch.tensor(1.0)
    }

def test_merton_model_basic(setup_tensors):
    equity_value = merton_model(**setup_tensors)
    assert torch.is_tensor(equity_value)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_zero_debt(setup_tensors):
    setup_tensors['debt'] = torch.tensor(0.0)
    equity_value = merton_model(**setup_tensors)
    assert torch.isclose(equity_value, setup_tensors['asset_value'], rtol=1e-4)

def test_merton_model_high_debt(setup_tensors):
    setup_tensors['debt'] = torch.tensor(200.0)
    equity_value = merton_model(**setup_tensors)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    equity_value = merton_model(**setup_tensors)
    assert equity_value >= 0
    assert equity_value <= setup_tensors['asset_value']

def test_merton_model_high_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(1.0)
    equity_value = merton_model(**setup_tensors)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_zero_rate(setup_tensors):
    setup_tensors['rate'] = torch.tensor(0.0)
    equity_value = merton_model(**setup_tensors)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_negative_rate(setup_tensors):
    setup_tensors['rate'] = torch.tensor(-0.01)
    equity_value = merton_model(**setup_tensors)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_zero_expiry(setup_tensors):
    setup_tensors['expiry'] = torch.tensor(0.0)
    equity_value = merton_model(**setup_tensors)
    assert torch.isclose(equity_value, torch.max(setup_tensors['asset_value'] - setup_tensors['debt'], torch.tensor(0.0)), rtol=1e-4)

def test_merton_model_long_expiry(setup_tensors):
    setup_tensors['expiry'] = torch.tensor(10.0)
    equity_value = merton_model(**setup_tensors)
    assert equity_value > 0
    assert equity_value < setup_tensors['asset_value']

def test_merton_model_batch_input():
    asset_value = torch.tensor([100.0, 120.0, 80.0])
    debt = torch.tensor([80.0, 90.0, 70.0])
    volatility = torch.tensor([0.2, 0.3, 0.1])
    rate = torch.tensor([0.05, 0.06, 0.04])
    expiry = torch.tensor([1.0, 2.0, 0.5])
    
    equity_value = merton_model(asset_value, debt, volatility, rate, expiry)
    assert equity_value.shape == (3,)
    assert torch.all(equity_value > 0)
    assert torch.all(equity_value < asset_value)
