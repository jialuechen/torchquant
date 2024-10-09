import pytest
import torch
from torchquantlib.models.binomial_tree import binomial_tree

@pytest.fixture
def option_params():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'steps': 100
    }

def test_european_call(option_params):
    price = binomial_tree('call', 'european', **option_params)
    assert isinstance(price, torch.Tensor)
    assert price.item() > 0

def test_european_put(option_params):
    price = binomial_tree('put', 'european', **option_params)
    assert isinstance(price, torch.Tensor)
    assert price.item() > 0

def test_american_call(option_params):
    price = binomial_tree('call', 'american', **option_params)
    assert isinstance(price, torch.Tensor)
    assert price.item() > 0

def test_american_put(option_params):
    price = binomial_tree('put', 'american', **option_params)
    assert isinstance(price, torch.Tensor)
    assert price.item() > 0

def test_put_call_parity(option_params):
    call_price = binomial_tree('call', 'european', **option_params)
    put_price = binomial_tree('put', 'european', **option_params)
    S = option_params['spot']
    K = option_params['strike']
    r = option_params['rate']
    T = option_params['expiry']
    expected_diff = S - K * torch.exp(-r * T)
    actual_diff = call_price - put_price
    assert torch.isclose(actual_diff, expected_diff, rtol=1e-3)

def test_american_vs_european_call(option_params):
    american_price = binomial_tree('call', 'american', **option_params)
    european_price = binomial_tree('call', 'european', **option_params)
    assert american_price >= european_price

def test_american_vs_european_put(option_params):
    american_price = binomial_tree('put', 'american', **option_params)
    european_price = binomial_tree('put', 'european', **option_params)
    assert american_price >= european_price

def test_increase_volatility(option_params):
    low_vol_price = binomial_tree('call', 'european', **option_params)
    option_params['volatility'] *= 2
    high_vol_price = binomial_tree('call', 'european', **option_params)
    assert high_vol_price > low_vol_price

def test_increase_expiry(option_params):
    short_expiry_price = binomial_tree('call', 'european', **option_params)
    option_params['expiry'] *= 2
    long_expiry_price = binomial_tree('call', 'european', **option_params)
    assert long_expiry_price > short_expiry_price

def test_zero_volatility(option_params):
    option_params['volatility'] = torch.tensor(0.0)
    price = binomial_tree('call', 'european', **option_params)
    assert price >= 0

def test_zero_expiry(option_params):
    option_params['expiry'] = torch.tensor(0.0)
    price = binomial_tree('call', 'european', **option_params)
    assert price >= 0

def test_invalid_option_type(option_params):
    with pytest.raises(ValueError):
        binomial_tree('invalid', 'european', **option_params)

def test_invalid_option_style(option_params):
    with pytest.raises(ValueError):
        binomial_tree('call', 'invalid', **option_params)

def test_batch_input():
    batch_params = {
        'spot': torch.tensor([100.0, 110.0, 90.0]),
        'strike': torch.tensor([100.0, 100.0, 100.0]),
        'expiry': torch.tensor([1.0, 2.0, 0.5]),
        'volatility': torch.tensor([0.2, 0.3, 0.1]),
        'rate': torch.tensor([0.05, 0.06, 0.04]),
        'steps': 100
    }
    prices = binomial_tree('call', 'european', **batch_params)
    assert prices.shape == (3,)

def test_convergence():
    params = {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'steps': 100
    }
    price_100 = binomial_tree('call', 'european', **params)
    params['steps'] = 1000
    price_1000 = binomial_tree('call', 'european', **params)
    assert torch.isclose(price_100, price_1000, rtol=1e-2)
