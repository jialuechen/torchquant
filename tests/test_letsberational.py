import pytest
import torch
from torchquantlib.core.asset_pricing.letsberational import implied_volatility, normal_cdf, normal_pdf

@pytest.fixture
def setup_tensors():
    return {
        'option_price': torch.tensor(10.0),
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'rate': torch.tensor(0.05),
        'is_call': True
    }

def test_implied_volatility_call(setup_tensors):
    iv = implied_volatility(**setup_tensors)
    assert iv > 0
    assert iv < 1  # Implied volatility is typically less than 100%

def test_implied_volatility_put(setup_tensors):
    setup_tensors['is_call'] = False
    iv = implied_volatility(**setup_tensors)
    assert iv > 0
    assert iv < 1

def test_implied_volatility_atm(setup_tensors):
    iv_call = implied_volatility(**setup_tensors)
    setup_tensors['is_call'] = False
    iv_put = implied_volatility(**setup_tensors)
    assert torch.isclose(iv_call, iv_put, rtol=1e-4)

def test_implied_volatility_itm_call(setup_tensors):
    setup_tensors['strike'] = torch.tensor(90.0)
    iv = implied_volatility(**setup_tensors)
    assert iv > 0

def test_implied_volatility_otm_call(setup_tensors):
    setup_tensors['strike'] = torch.tensor(110.0)
    iv = implied_volatility(**setup_tensors)
    assert iv > 0

def test_implied_volatility_zero_price(setup_tensors):
    setup_tensors['option_price'] = torch.tensor(0.0)
    with pytest.raises(RuntimeError):
        implied_volatility(**setup_tensors)

def test_implied_volatility_high_price(setup_tensors):
    setup_tensors['option_price'] = torch.tensor(100.0)
    with pytest.raises(RuntimeError):
        implied_volatility(**setup_tensors)

def test_implied_volatility_convergence(setup_tensors):
    iv = implied_volatility(**setup_tensors)
    # Use the calculated IV to price the option and compare with the original price
    d1 = (torch.log(setup_tensors['spot'] / setup_tensors['strike']) + 
          (setup_tensors['rate'] + 0.5 * iv**2) * setup_tensors['expiry']) / (iv * torch.sqrt(setup_tensors['expiry']))
    d2 = d1 - iv * torch.sqrt(setup_tensors['expiry'])
    
    if setup_tensors['is_call']:
        calculated_price = (setup_tensors['spot'] * normal_cdf(d1) - 
                            setup_tensors['strike'] * torch.exp(-setup_tensors['rate'] * setup_tensors['expiry']) * normal_cdf(d2))
    else:
        calculated_price = (setup_tensors['strike'] * torch.exp(-setup_tensors['rate'] * setup_tensors['expiry']) * normal_cdf(-d2) - 
                            setup_tensors['spot'] * normal_cdf(-d1))
    
    assert torch.isclose(calculated_price, setup_tensors['option_price'], rtol=1e-4)

def test_normal_cdf():
    assert torch.isclose(normal_cdf(torch.tensor(0.0)), torch.tensor(0.5), rtol=1e-4)
    assert normal_cdf(torch.tensor(-float('inf'))) == 0.0
    assert normal_cdf(torch.tensor(float('inf'))) == 1.0

def test_normal_pdf():
    assert torch.isclose(normal_pdf(torch.tensor(0.0)), torch.tensor(0.3989), rtol=1e-4)
    assert normal_pdf(torch.tensor(-float('inf'))) == 0.0
    assert normal_pdf(torch.tensor(float('inf'))) == 0.0
