import pytest
import torch
from torchquantlib.core.asset_pricing.bond.fixed_income_forward import fixed_income_forward

@pytest.fixture
def setup_tensors():
    return {
        'face_value': torch.tensor(1000.0),
        'rate': torch.tensor(0.05),
        'time_to_maturity': torch.tensor(2.0),
        'forward_rate': torch.tensor(0.06)
    }

def test_fixed_income_forward_positive_rates(setup_tensors):
    result = fixed_income_forward(**setup_tensors)
    expected = setup_tensors['face_value'] * torch.exp((setup_tensors['forward_rate'] - setup_tensors['rate']) * setup_tensors['time_to_maturity'])
    assert torch.isclose(result, expected, rtol=1e-5)

def test_fixed_income_forward_zero_rates(setup_tensors):
    setup_tensors['rate'] = torch.tensor(0.0)
    setup_tensors['forward_rate'] = torch.tensor(0.0)
    result = fixed_income_forward(**setup_tensors)
    assert torch.isclose(result, setup_tensors['face_value'], rtol=1e-5)

def test_fixed_income_forward_negative_rates(setup_tensors):
    setup_tensors['rate'] = torch.tensor(-0.01)
    setup_tensors['forward_rate'] = torch.tensor(-0.02)
    result = fixed_income_forward(**setup_tensors)
    expected = setup_tensors['face_value'] * torch.exp((setup_tensors['forward_rate'] - setup_tensors['rate']) * setup_tensors['time_to_maturity'])
    assert torch.isclose(result, expected, rtol=1e-5)

def test_fixed_income_forward_zero_time_to_maturity(setup_tensors):
    setup_tensors['time_to_maturity'] = torch.tensor(0.0)
    result = fixed_income_forward(**setup_tensors)
    assert torch.isclose(result, setup_tensors['face_value'], rtol=1e-5)

def test_fixed_income_forward_large_time_to_maturity(setup_tensors):
    setup_tensors['time_to_maturity'] = torch.tensor(100.0)
    result = fixed_income_forward(**setup_tensors)
    expected = setup_tensors['face_value'] * torch.exp((setup_tensors['forward_rate'] - setup_tensors['rate']) * setup_tensors['time_to_maturity'])
    assert torch.isclose(result, expected, rtol=1e-5)

def test_fixed_income_forward_batch_input():
    face_value = torch.tensor([1000.0, 2000.0, 3000.0])
    rate = torch.tensor([0.05, 0.06, 0.07])
    time_to_maturity = torch.tensor([2.0, 3.0, 4.0])
    forward_rate = torch.tensor([0.06, 0.07, 0.08])
    
    result = fixed_income_forward(face_value, rate, time_to_maturity, forward_rate)
    expected = face_value * torch.exp((forward_rate - rate) * time_to_maturity)
    assert torch.allclose(result, expected, rtol=1e-5)
