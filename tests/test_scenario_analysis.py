import pytest
import torch
from torchquantlib.core.risk.market_risk.scenario_analysis import scenario_analysis

@pytest.fixture
def sample_portfolio():
    return torch.tensor(1000.0)

@pytest.fixture
def sample_scenarios():
    return torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2])

def test_scenario_analysis_basic(sample_portfolio, sample_scenarios):
    result = scenario_analysis(sample_portfolio, sample_scenarios)
    assert torch.is_tensor(result)
    assert result.shape == sample_scenarios.shape
    assert torch.allclose(result, sample_portfolio * sample_scenarios)

def test_scenario_analysis_single_scenario(sample_portfolio):
    single_scenario = torch.tensor(1.1)
    result = scenario_analysis(sample_portfolio, single_scenario)
    assert torch.is_tensor(result)
    assert result.item() == pytest.approx(1100.0)

def test_scenario_analysis_extreme_scenarios(sample_portfolio):
    extreme_scenarios = torch.tensor([0.0, 2.0])
    result = scenario_analysis(sample_portfolio, extreme_scenarios)
    assert torch.allclose(result, torch.tensor([0.0, 2000.0]))

def test_scenario_analysis_negative_scenario(sample_portfolio):
    negative_scenario = torch.tensor([-0.5])
    result = scenario_analysis(sample_portfolio, negative_scenario)
    assert result.item() == pytest.approx(-500.0)

def test_scenario_analysis_2d_scenarios(sample_portfolio):
    scenarios_2d = torch.tensor([[0.9, 1.0, 1.1], [0.8, 1.0, 1.2]])
    result = scenario_analysis(sample_portfolio, scenarios_2d)
    assert result.shape == scenarios_2d.shape
    assert torch.allclose(result, sample_portfolio * scenarios_2d)

def test_scenario_analysis_zero_portfolio():
    zero_portfolio = torch.tensor(0.0)
    scenarios = torch.tensor([0.8, 1.0, 1.2])
    result = scenario_analysis(zero_portfolio, scenarios)
    assert torch.allclose(result, torch.zeros_like(scenarios))

def test_scenario_analysis_empty_scenarios(sample_portfolio):
    empty_scenarios = torch.tensor([])
    result = scenario_analysis(sample_portfolio, empty_scenarios)
    assert result.numel() == 0

def test_scenario_analysis_large_portfolio():
    large_portfolio = torch.tensor(1e9)  # 1 billion
    scenarios = torch.tensor([0.99, 1.0, 1.01])
    result = scenario_analysis(large_portfolio, scenarios)
    expected = torch.tensor([0.99e9, 1e9, 1.01e9])
    assert torch.allclose(result, expected)

def test_scenario_analysis_type_error():
    with pytest.raises(TypeError):
        scenario_analysis(1000, torch.tensor([1.0, 1.1]))  # portfolio_value should be a tensor

def test_scenario_analysis_device_consistency(sample_scenarios):
    if torch.cuda.is_available():
        portfolio_gpu = torch.tensor(1000.0, device='cuda')
        scenarios_gpu = sample_scenarios.to('cuda')
        result = scenario_analysis(portfolio_gpu, scenarios_gpu)
        assert result.device == portfolio_gpu.device