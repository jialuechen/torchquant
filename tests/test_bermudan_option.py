import pytest
import torch
from torchquantlib.core.asset_pricing.option.bermudan_option import bermudan_option

@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'steps': 100,
        'exercise_dates': torch.tensor([25, 50, 75])  # Example exercise dates
    }

def test_bermudan_call_option(setup_tensors):
    price = bermudan_option('call', **setup_tensors)
    assert price > 0
    assert price >= setup_tensors['spot'] - setup_tensors['strike']

def test_bermudan_put_option(setup_tensors):
    price = bermudan_option('put', **setup_tensors)
    assert price > 0
    assert price >= setup_tensors['strike'] - setup_tensors['spot']

def test_bermudan_call_vs_put(setup_tensors):
    call_price = bermudan_option('call', **setup_tensors)
    put_price = bermudan_option('put', **setup_tensors)
    assert call_price != put_price

def test_bermudan_option_increase_volatility(setup_tensors):
    low_vol_price = bermudan_option('call', **setup_tensors)
    setup_tensors['volatility'] *= 2
    high_vol_price = bermudan_option('call', **setup_tensors)
    assert high_vol_price > low_vol_price

def test_bermudan_option_increase_expiry(setup_tensors):
    short_expiry_price = bermudan_option('call', **setup_tensors)
    setup_tensors['expiry'] *= 2
    setup_tensors['exercise_dates'] *= 2
    long_expiry_price = bermudan_option('call', **setup_tensors)
    assert long_expiry_price > short_expiry_price

def test_bermudan_option_increase_steps(setup_tensors):
    low_steps_price = bermudan_option('call', **setup_tensors)
    setup_tensors['steps'] *= 2
    setup_tensors['exercise_dates'] *= 2
    high_steps_price = bermudan_option('call', **setup_tensors)
    assert torch.isclose(low_steps_price, high_steps_price, rtol=1e-2)

def test_bermudan_option_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    with pytest.raises(RuntimeError):
        bermudan_option('call', **setup_tensors)

def test_bermudan_option_invalid_type():
    with pytest.raises(ValueError):
        bermudan_option('invalid', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                        expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                        rate=torch.tensor(0.05), steps=100, exercise_dates=torch.tensor([25, 50, 75]))

def test_bermudan_option_no_exercise_dates(setup_tensors):
    setup_tensors['exercise_dates'] = torch.tensor([])
    price = bermudan_option('call', **setup_tensors)
    assert price > 0  # Should behave like a European option

def test_bermudan_option_all_exercise_dates(setup_tensors):
    setup_tensors['exercise_dates'] = torch.arange(1, setup_tensors['steps'])
    price = bermudan_option('call', **setup_tensors)
    assert price > 0  # Should behave like an American option
