import pytest
import torch
from torchquantlib.core.asset_pricing.option.asian_option import asian_option

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

def test_asian_call_option(setup_tensors):
    price = asian_option('call', **setup_tensors)
    assert price > 0
    assert price <= setup_tensors['spot']  # Asian call should be cheaper than European call

def test_asian_put_option(setup_tensors):
    price = asian_option('put', **setup_tensors)
    assert price > 0
    assert price <= setup_tensors['strike']  # Asian put should be cheaper than European put

def test_asian_call_vs_put(setup_tensors):
    call_price = asian_option('call', **setup_tensors)
    put_price = asian_option('put', **setup_tensors)
    assert call_price != put_price

def test_asian_option_increase_volatility(setup_tensors):
    low_vol_price = asian_option('call', **setup_tensors)
    setup_tensors['volatility'] *= 2
    high_vol_price = asian_option('call', **setup_tensors)
    assert high_vol_price > low_vol_price

def test_asian_option_increase_expiry(setup_tensors):
    short_expiry_price = asian_option('call', **setup_tensors)
    setup_tensors['expiry'] *= 2
    long_expiry_price = asian_option('call', **setup_tensors)
    assert long_expiry_price != short_expiry_price

def test_asian_option_increase_steps(setup_tensors):
    low_steps_price = asian_option('call', **setup_tensors)
    setup_tensors['steps'] *= 2
    high_steps_price = asian_option('call', **setup_tensors)
    assert torch.isclose(low_steps_price, high_steps_price, rtol=1e-2)

def test_asian_option_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    price = asian_option('call', **setup_tensors)
    assert price >= 0

def test_asian_option_invalid_type():
    with pytest.raises(ValueError):
        asian_option('invalid', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                     expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                     rate=torch.tensor(0.05), steps=100)

def test_asian_option_at_the_money(setup_tensors):
    call_price = asian_option('call', **setup_tensors)
    put_price = asian_option('put', **setup_tensors)
    assert torch.isclose(call_price, put_price, rtol=1e-2)
