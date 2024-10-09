import pytest
import torch
from torchquantlib.core.risk.market_risk.expected_shortfall import calculate_es
from torchquantlib.core.risk.market_risk.var import calculate_var

@pytest.fixture
def sample_returns():
    return torch.tensor([-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

def test_calculate_es_basic(sample_returns):
    es = calculate_es(sample_returns, 0.95)
    assert isinstance(es, torch.Tensor)
    assert es.item() > 0

def test_calculate_es_higher_confidence(sample_returns):
    es_95 = calculate_es(sample_returns, 0.95)
    es_99 = calculate_es(sample_returns, 0.99)
    assert es_99 > es_95

def test_calculate_es_vs_var(sample_returns):
    confidence_level = 0.95
    es = calculate_es(sample_returns, confidence_level)
    var = calculate_var(sample_returns, confidence_level)
    assert es >= var

def test_calculate_es_extreme_confidence(sample_returns):
    es_low = calculate_es(sample_returns, 0.01)
    es_high = calculate_es(sample_returns, 0.99)
    assert es_low < es_high

def test_calculate_es_all_positive_returns():
    positive_returns = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
    es = calculate_es(positive_returns, 0.95)
    assert es == 0

def test_calculate_es_all_negative_returns():
    negative_returns = torch.tensor([-0.05, -0.04, -0.03, -0.02, -0.01])
    es = calculate_es(negative_returns, 0.95)
    assert es > 0

def test_calculate_es_single_value():
    single_value = torch.tensor([0.1])
    es = calculate_es(single_value, 0.95)
    assert es == 0

def test_calculate_es_empty_tensor():
    with pytest.raises(RuntimeError):
        calculate_es(torch.tensor([]), 0.95)

def test_calculate_es_invalid_confidence():
    with pytest.raises(ValueError):
        calculate_es(torch.tensor([0.1, 0.2, 0.3]), 1.5)

def test_calculate_es_2d_tensor():
    returns_2d = torch.tensor([[-0.05, -0.03, -0.01, 0.01, 0.03],
                               [-0.04, -0.02, 0.00, 0.02, 0.04]])
    es = calculate_es(returns_2d, 0.95)
    assert es.shape == (2,)
