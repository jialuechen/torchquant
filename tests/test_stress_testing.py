import pytest
import torch
from torchquantlib.core.risk.market_risk.stress_testing import stress_test

@pytest.fixture
def sample_portfolio():
    return torch.tensor(1000.0)

@pytest.fixture
def sample_stress_scenarios():
    return torch.tensor([-0.3, -0.2, -0.1, 0.0, 0.1])

def test_stress_test_basic(sample_portfolio, sample_stress_scenarios):
    result = stress_test(sample_portfolio, sample_stress_scenarios)
    assert torch.is_tensor(result)
    assert result.shape == sample_stress_scenarios.shape
    expected = sample_portfolio * (1 + sample_stress_scenarios)
    assert torch.allclose(result, expected)

def test_stress_test_single_scenario(sample_portfolio):
    single_scenario = torch.tensor(-0.3)
    result = stress_test(sample_portfolio, single_scenario)
    assert torch.is_tensor(result)
    assert result.item() == pytest.approx(700.0)

def test_stress_test_extreme_scenarios(sample_portfolio):
    extreme_scenarios = torch.tensor([-1.0, 1.0])
    result = stress_test(sample_portfolio, extreme_scenarios)
    assert torch.allclose(result, torch.tensor([0.0, 2000.0]))

def test_stress_test_positive_scenario(sample_portfolio):
    positive_scenario = torch.tensor([0.5])
    result = stress_test(sample_portfolio, positive_scenario)
    assert result.item() == pytest.approx(1500.0)

def test_stress_test_2d_scenarios(sample_portfolio):
    scenarios_2d = torch.tensor([[-0.3, -0.2, -0.1], [-0.1, 0.0, 0.1]])
    result = stress_test(sample_portfolio, scenarios_2d)
    assert result.shape == scenarios_2d.shape
    expected = sample_portfolio * (1 + scenarios_2d)
    assert torch.allclose(result, expected)

def test_stress_test_zero_portfolio():
    zero_portfolio = torch.tensor(0.0)
    scenarios = torch.tensor([-0.3, 0.0, 0.3])
    result = stress_test(zero_portfolio, scenarios)
    assert torch.allclose(result, torch.zeros_like(scenarios))

def test_stress_test_empty_scenarios(sample_portfolio):
    empty_scenarios = torch.tensor([])
    result = stress_test(sample_portfolio, empty_scenarios)
    assert result.numel() == 0

def test_stress_test_large_portfolio():
    large_portfolio = torch.tensor(1e9)  # 1 billion
    scenarios = torch.tensor([-0.01, 0.0, 0.01])
    result = stress_test(large_portfolio, scenarios)
    expected = torch.tensor([0.99e9, 1e9, 1.01e9])
    assert torch.allclose(result, expected)

def test_stress_test_type_error():
    with pytest.raises(TypeError):
        stress_test(1000, torch.tensor([-0.1, 0.1]))  # portfolio_value should be a tensor

def test_stress_test_device_consistency(sample_stress_scenarios):
    if torch.cuda.is_available():
        portfolio_gpu = torch.tensor(1000.0, device='cuda')
        scenarios_gpu = sample_stress_scenarios.to('cuda')
        result = stress_test(portfolio_gpu, scenarios_gpu)
        assert result.device == portfolio_gpu.device

def test_stress_test_extreme_positive_scenario(sample_portfolio):
    extreme_positive = torch.tensor([10.0])  # 1000% increase
    result = stress_test(sample_portfolio, extreme_positive)
    assert result.item() == pytest.approx(11000.0)