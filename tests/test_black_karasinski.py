import pytest
import torch
from torchquantlib.models.interest_rate.black_karasinski import BlackKarasinski

@pytest.fixture
def bk_model():
    return BlackKarasinski()

def test_initialization(bk_model):
    assert isinstance(bk_model.params['a'], torch.Tensor)
    assert isinstance(bk_model.params['sigma'], torch.Tensor)
    assert isinstance(bk_model.params['r0'], torch.Tensor)
    assert bk_model.params['a'].requires_grad
    assert bk_model.params['sigma'].requires_grad
    assert bk_model.params['r0'].requires_grad

def test_simulate_shape(bk_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_positive_rates(bk_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_different_paths(bk_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = bk_model.simulate(S0, T, N, steps)
    result2 = bk_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(bk_model):
    bk_model.params['a'].data.fill_(-1)
    bk_model.params['sigma'].data.fill_(-1)
    bk_model._apply_constraints()
    assert bk_model.params['a'].item() >= 1e-6
    assert bk_model.params['sigma'].item() >= 1e-6

def test_simulate_with_zero_volatility():
    bk_model = BlackKarasinski(sigma_init=0)
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_with_high_mean_reversion():
    bk_model = BlackKarasinski(a_init=10)
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_with_long_time_horizon():
    bk_model = BlackKarasinski()
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = bk_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_batch_consistency():
    bk_model = BlackKarasinski()
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = bk_model.simulate(S0, T, N1, steps)
    result2 = bk_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    bk_model = BlackKarasinski()
    bk_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow():
    bk_model = BlackKarasinski()
    S0, T, N, steps = 100, 1, 1000, 252
    result = bk_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert bk_model.params['a'].grad is not None
    assert bk_model.params['sigma'].grad is not None
    assert bk_model.params['r0'].grad is not None
