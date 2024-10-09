import pytest
import torch
from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton

@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'dividend': torch.tensor(0.02)
    }

def test_european_call(setup_tensors):
    price = black_scholes_merton('call', 'european', **setup_tensors)
    assert price > 0
    assert price < setup_tensors['spot']

def test_european_put(setup_tensors):
    price = black_scholes_merton('put', 'european', **setup_tensors)
    assert price > 0
    assert price < setup_tensors['strike']

def test_american_call(setup_tensors):
    price = black_scholes_merton('call', 'american', **setup_tensors)
    assert price > 0
    assert price >= black_scholes_merton('call', 'european', **setup_tensors)

def test_american_put(setup_tensors):
    price = black_scholes_merton('put', 'american', **setup_tensors)
    assert price > 0
    assert price >= black_scholes_merton('put', 'european', **setup_tensors)

def test_put_call_parity(setup_tensors):
    call_price = black_scholes_merton('call', 'european', **setup_tensors)
    put_price = black_scholes_merton('put', 'european', **setup_tensors)
    S = setup_tensors['spot']
    K = setup_tensors['strike']
    r = setup_tensors['rate']
    q = setup_tensors['dividend']
    T = setup_tensors['expiry']
    assert torch.isclose(call_price - put_price, S * torch.exp(-q * T) - K * torch.exp(-r * T), rtol=1e-4)

def test_increase_volatility(setup_tensors):
    low_vol_price = black_scholes_merton('call', 'european', **setup_tensors)
    setup_tensors['volatility'] *= 2
    high_vol_price = black_scholes_merton('call', 'european', **setup_tensors)
    assert high_vol_price > low_vol_price

def test_increase_expiry(setup_tensors):
    short_expiry_price = black_scholes_merton('call', 'european', **setup_tensors)
    setup_tensors['expiry'] *= 2
    long_expiry_price = black_scholes_merton('call', 'european', **setup_tensors)
    assert long_expiry_price > short_expiry_price

def test_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    price = black_scholes_merton('call', 'european', **setup_tensors)
    assert price >= 0

def test_invalid_option_type():
    with pytest.raises(ValueError):
        black_scholes_merton('invalid', 'european', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                             expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                             rate=torch.tensor(0.05), dividend=torch.tensor(0.02))

def test_invalid_option_style():
    with pytest.raises(ValueError):
        black_scholes_merton('call', 'invalid', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                             expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                             rate=torch.tensor(0.05), dividend=torch.tensor(0.02))
