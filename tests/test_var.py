import pytest
import torch
from torchquantlib.core.risk.market_risk.var import calculate_var

@pytest.fixture
def sample_returns():
    return torch.tensor([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

def test_calculate_var_basic(sample_returns):
    var = calculate_var(sample_returns, 0.95)
    assert isinstance(var, torch.Tensor)
    assert var.item() > 0

def test_calculate_var_higher_confidence(sample_returns):
    var_95 = calculate_var(sample_returns, 0.95)
    var_99 = calculate_var(sample_returns, 0.99)
    assert var_99 > var_95

def test_calculate_var_extreme_confidence(sample_returns):
    var_low = calculate_var(sample_returns, 0.01)
    var_high = calculate_var(sample_returns, 0.99)
    assert var_low < var_high

def test_calculate_var_all_positive_returns():
    positive_returns = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    var = calculate_var(positive_returns, 0.95)
    assert var == 0

def test_calculate_var_all_negative_returns():
    negative_returns = torch.tensor([-0.05, -0.04, -0.03, -0.02, -0.01])
    var = calculate_var(negative_returns, 0.95)
    assert var > 0

def test_calculate_var_single_value():
    single_value = torch.tensor([0.1])
    var = calculate_var(single_value, 0.95)
    assert var == 0

def test_calculate_var_empty_tensor():
    with pytest.raises(RuntimeError):
        calculate_var(torch.tensor([]), 0.95)

def test_calculate_var_invalid_confidence():
    with pytest.raises(IndexError):
        calculate_var(torch.tensor([0.1, 0.2, 0.3]), 1.5)

def test_calculate_var_2d_tensor():
    returns_2d = torch.tensor([[-0.05, -0.03, -0.01, 0.01, 0.03],
                               [-0.04, -0.02, 0.00, 0.02, 0.04]])
    var = calculate_var(returns_2d, 0.95)
    assert var.shape == (2,)

def test_calculate_var_large_dataset():
    large_returns = torch.randn(10000)
    var = calculate_var(large_returns, 0.99)
    assert 2 < var.item() < 3  # Approximately 2.33 for standard normal distribution

def test_calculate_var_device_consistency():
    if torch.cuda.is_available():
        returns_gpu = torch.tensor([-0.05, -0.03, -0.01, 0.01, 0.03], device='cuda')
        var = calculate_var(returns_gpu, 0.95)
        assert var.device == returns_gpu.device

def test_calculate_var_gradient():
    returns = torch.tensor([-0.05, -0.03, -0.01, 0.01, 0.03], requires_grad=True)
    var = calculate_var(returns, 0.95)
    assert not var.requires_grad  # VaR calculation should not propagate gradients
