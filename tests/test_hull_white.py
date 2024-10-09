import pytest
import torch
from torchquantlib.models.interest_rate.hull_white import HullWhiteModel

@pytest.fixture
def hw_model():
    return HullWhiteModel(a_init=0.1, sigma_init=0.01, r0_init=0.03)

def test_initialization(hw_model):
    assert isinstance(hw_model.params['a'], torch.Tensor)
    assert isinstance(hw_model.params['sigma'], torch.Tensor)
    assert isinstance(hw_model.params['r0'], torch.Tensor)
    assert all(param.requires_grad for param in hw_model.params.values())

def test_simulate_shape(hw_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_different_paths(hw_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = hw_model.simulate(S0, T, N, steps)
    result2 = hw_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(hw_model):
    hw_model.params['a'].data.fill_(-1)
    hw_model.params['sigma'].data.fill_(-1)
    hw_model._apply_constraints()
    assert hw_model.params['a'].item() >= 1e-6
    assert hw_model.params['sigma'].item() >= 1e-6

def test_simulate_long_time_horizon(hw_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = hw_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_batch_consistency(hw_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = hw_model.simulate(S0, T, N1, steps)
    result2 = hw_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    hw_model = HullWhiteModel()
    hw_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(hw_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert all(param.grad is not None for param in hw_model.params.values())

def test_mean_reversion(hw_model):
    S0, T, N, steps = 100, 10, 10000, 252*10
    result = hw_model.simulate(S0, T, N, steps)
    mean_rate = result.mean()
    assert torch.abs(mean_rate) < 0.1  # Rates should revert towards 0 (assuming θ(t) = 0)

def test_zero_volatility():
    hw_model = HullWhiteModel(sigma_init=0)
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    assert torch.allclose(result, result[0])  # All paths should be identical

def test_high_mean_reversion():
    hw_model = HullWhiteModel(a_init=10)
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    assert torch.std(result) < 0.1  # Rates should be close to 0 (assuming θ(t) = 0)

def test_negative_rates():
    hw_model = HullWhiteModel(r0_init=-0.01)  # Start with negative rate
    S0, T, N, steps = 100, 1, 1000, 252
    result = hw_model.simulate(S0, T, N, steps)
    assert torch.any(result < 0)  # Hull-White model allows for negative rates

def test_parameter_sensitivity():
    hw_model1 = HullWhiteModel(a_init=0.1, sigma_init=0.01)
    hw_model2 = HullWhiteModel(a_init=0.2, sigma_init=0.02)
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = hw_model1.simulate(S0, T, N, steps)
    result2 = hw_model2.simulate(S0, T, N, steps)
    assert torch.std(result2) > torch.std(result1)  # Higher volatility should lead to more dispersion
