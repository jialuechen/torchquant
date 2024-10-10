import pytest
import torch
from torchquantlib.utils.yield_curve_construction import bootstrap_yield_curve, nelson_siegel_yield_curve

@pytest.fixture
def sample_cash_flows():
    return torch.tensor([
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100]
    ], dtype=torch.float32)

@pytest.fixture
def sample_prices():
    return torch.tensor([95, 90, 85], dtype=torch.float32)

def test_bootstrap_yield_curve(sample_cash_flows, sample_prices):
    yields = bootstrap_yield_curve(sample_cash_flows, sample_prices)
    assert isinstance(yields, torch.Tensor)
    assert yields.shape == (3,)
    assert torch.all(yields >= 0)
    assert torch.all(yields.diff() >= 0)  # Yields should be non-decreasing

def test_bootstrap_yield_curve_single_instrument():
    cash_flows = torch.tensor([[100]], dtype=torch.float32)
    prices = torch.tensor([95], dtype=torch.float32)
    yields = bootstrap_yield_curve(cash_flows, prices)
    assert yields.shape == (1,)
    assert yields.item() == pytest.approx(0.0526, rel=1e-3)

def test_bootstrap_yield_curve_zero_price():
    with pytest.raises(RuntimeError):
        bootstrap_yield_curve(torch.tensor([[100]]), torch.tensor([0.0]))

def test_bootstrap_yield_curve_negative_price():
    with pytest.raises(RuntimeError):
        bootstrap_yield_curve(torch.tensor([[100]]), torch.tensor([-95.0]))

def test_nelson_siegel_yield_curve():
    tau = torch.tensor([1, 2, 3, 5, 10], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
    assert isinstance(yields, torch.Tensor)
    assert yields.shape == (5,)
    assert torch.all(yields >= 0)

def test_nelson_siegel_yield_curve_single_maturity():
    tau = torch.tensor([5], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
    assert yields.shape == (1,)

def test_nelson_siegel_yield_curve_long_term():
    tau = torch.tensor([1000], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
    assert yields.item() == pytest.approx(beta0.item(), rel=1e-3)

def test_nelson_siegel_yield_curve_zero_maturity():
    tau = torch.tensor([0], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    with pytest.raises(RuntimeError):
        nelson_siegel_yield_curve(tau, beta0, beta1, beta2)

def test_nelson_siegel_yield_curve_negative_maturity():
    tau = torch.tensor([-1], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    with pytest.raises(RuntimeError):
        nelson_siegel_yield_curve(tau, beta0, beta1, beta2)

def test_nelson_siegel_yield_curve_shape():
    tau = torch.tensor([1, 2, 3, 5, 10], dtype=torch.float32)
    beta0 = torch.tensor(0.03)
    beta1 = torch.tensor(0.02)
    beta2 = torch.tensor(-0.01)
    yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
    assert yields[0] > yields[-1]  # Short-term rate should be higher than long-term rate

def test_bootstrap_yield_curve_batch():
    cash_flows = torch.tensor([
        [[100, 0], [0, 100]],
        [[100, 0], [0, 100]]
    ], dtype=torch.float32)
    prices = torch.tensor([[95, 90], [97, 92]], dtype=torch.float32)
    yields = bootstrap_yield_curve(cash_flows, prices)
    assert yields.shape == (2, 2)

def test_nelson_siegel_yield_curve_batch():
    tau = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    beta0 = torch.tensor([0.03, 0.04])
    beta1 = torch.tensor([0.02, 0.01])
    beta2 = torch.tensor([-0.01, -0.02])
    yields = nelson_siegel_yield_curve(tau, beta0, beta1, beta2)
    assert yields.shape == (2, 3)
