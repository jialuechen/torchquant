import pytest
import torch
from torchquantlib.core.asset_pricing.option.american_option import american_option

@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'steps': 100
    }

def test_american_call_option(setup_tensors):
    price = american_option('call', **setup_tensors)
    assert price > 0
    assert price >= setup_tensors['spot'] - setup_tensors['strike']

def test_american_put_option(setup_tensors):
    price = american_option('put', **setup_tensors)
    assert price > 0
    assert price >= setup_tensors['strike'] - setup_tensors['spot']

def test_american_call_vs_put(setup_tensors):
    call_price = american_option('call', **setup_tensors)
    put_price = american_option('put', **setup_tensors)
    assert call_price != put_price

def test_american_option_increase_volatility(setup_tensors):
    low_vol_price = american_option('call', **setup_tensors)
    setup_tensors['volatility'] *= 2
    high_vol_price = american_option('call', **setup_tensors)
    assert high_vol_price > low_vol_price

def test_american_option_increase_expiry(setup_tensors):
    short_expiry_price = american_option('call', **setup_tensors)
    setup_tensors['expiry'] *= 2
    long_expiry_price = american_option('call', **setup_tensors)
    assert long_expiry_price > short_expiry_price

def test_american_option_increase_steps(setup_tensors):
    low_steps_price = american_option('call', **setup_tensors)
    setup_tensors['steps'] *= 2
    high_steps_price = american_option('call', **setup_tensors)
    assert torch.isclose(low_steps_price, high_steps_price, rtol=1e-2)

def test_american_option_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    with pytest.raises(RuntimeError):
        american_option('call', **setup_tensors)

def test_american_option_invalid_type():
    with pytest.raises(ValueError):
        american_option('invalid', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                        expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                        rate=torch.tensor(0.05), steps=100)
