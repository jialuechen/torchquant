import pytest
import torch
from torchquantlib.core.risk.valuation_adjustment.valuation_adjustment import (
    calculate_cva, calculate_dva, calculate_fva, calculate_mva
)

@pytest.fixture
def sample_data():
    return {
        'exposure': torch.tensor(1000.0),
        'default_prob': torch.tensor(0.05),
        'recovery_rate': torch.tensor(0.4),
        'funding_spread': torch.tensor(0.02),
        'funding_cost': torch.tensor(0.03),
        'maturity': torch.tensor(5.0)
    }

def test_calculate_cva(sample_data):
    cva = calculate_cva(sample_data['exposure'], sample_data['default_prob'], sample_data['recovery_rate'])
    assert torch.is_tensor(cva)
    expected_cva = 1000.0 * (1 - 0.4) * 0.05
    assert torch.isclose(cva, torch.tensor(expected_cva))

def test_calculate_dva(sample_data):
    dva = calculate_dva(sample_data['exposure'], sample_data['default_prob'], sample_data['recovery_rate'])
    assert torch.is_tensor(dva)
    expected_dva = 1000.0 * (1 - 0.4) * 0.05
    assert torch.isclose(dva, torch.tensor(expected_dva))

def test_calculate_fva(sample_data):
    fva = calculate_fva(sample_data['exposure'], sample_data['funding_spread'], sample_data['maturity'])
    assert torch.is_tensor(fva)
    expected_fva = 1000.0 * 0.02 * 5.0
    assert torch.isclose(fva, torch.tensor(expected_fva))

def test_calculate_mva(sample_data):
    mva = calculate_mva(sample_data['exposure'], sample_data['funding_cost'], sample_data['maturity'])
    assert torch.is_tensor(mva)
    expected_mva = 1000.0 * 0.03 * 5.0
    assert torch.isclose(mva, torch.tensor(expected_mva))

def test_zero_exposure():
    zero_exposure = torch.tensor(0.0)
    assert calculate_cva(zero_exposure, torch.tensor(0.05), torch.tensor(0.4)) == 0.0
    assert calculate_dva(zero_exposure, torch.tensor(0.05), torch.tensor(0.4)) == 0.0
    assert calculate_fva(zero_exposure, torch.tensor(0.02), torch.tensor(5.0)) == 0.0
    assert calculate_mva(zero_exposure, torch.tensor(0.03), torch.tensor(5.0)) == 0.0

def test_full_recovery():
    full_recovery = torch.tensor(1.0)
    assert calculate_cva(torch.tensor(1000.0), torch.tensor(0.05), full_recovery) == 0.0
    assert calculate_dva(torch.tensor(1000.0), torch.tensor(0.05), full_recovery) == 0.0

def test_zero_default_prob():
    zero_default = torch.tensor(0.0)
    assert calculate_cva(torch.tensor(1000.0), zero_default, torch.tensor(0.4)) == 0.0
    assert calculate_dva(torch.tensor(1000.0), zero_default, torch.tensor(0.4)) == 0.0

def test_zero_maturity():
    zero_maturity = torch.tensor(0.0)
    assert calculate_fva(torch.tensor(1000.0), torch.tensor(0.02), zero_maturity) == 0.0
    assert calculate_mva(torch.tensor(1000.0), torch.tensor(0.03), zero_maturity) == 0.0

def test_batch_input():
    exposure = torch.tensor([1000.0, 2000.0, 3000.0])
    default_prob = torch.tensor([0.05, 0.06, 0.07])
    recovery_rate = torch.tensor([0.4, 0.5, 0.6])
    funding_spread = torch.tensor([0.02, 0.03, 0.04])
    funding_cost = torch.tensor([0.03, 0.04, 0.05])
    maturity = torch.tensor([5.0, 6.0, 7.0])

    cva = calculate_cva(exposure, default_prob, recovery_rate)
    dva = calculate_dva(exposure, default_prob, recovery_rate)
    fva = calculate_fva(exposure, funding_spread, maturity)
    mva = calculate_mva(exposure, funding_cost, maturity)

    assert cva.shape == (3,)
    assert dva.shape == (3,)
    assert fva.shape == (3,)
    assert mva.shape == (3,)

def test_negative_values():
    with pytest.raises(RuntimeError):
        calculate_cva(torch.tensor(-1000.0), torch.tensor(0.05), torch.tensor(0.4))
    with pytest.raises(RuntimeError):
        calculate_dva(torch.tensor(-1000.0), torch.tensor(0.05), torch.tensor(0.4))
    with pytest.raises(RuntimeError):
        calculate_fva(torch.tensor(1000.0), torch.tensor(-0.02), torch.tensor(5.0))
    with pytest.raises(RuntimeError):
        calculate_mva(torch.tensor(1000.0), torch.tensor(0.03), torch.tensor(-5.0))
