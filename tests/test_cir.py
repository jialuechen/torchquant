import pytest
import torch
from torchquantlib.models.interest_rate.cir import CIR

@pytest.fixture
def cir_model():
    return CIR(kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03)

def test_initialization(cir_model):
    assert isinstance(cir_model.params['kappa'], torch.Tensor)
    assert isinstance(cir_model.params['theta'], torch.Tensor)
    assert isinstance(cir_model.params['sigma'], torch.Tensor)
    assert isinstance(cir_model.params['r0'], torch.Tensor)
    assert all(param.requires_grad for param in cir_model.params.values())

def test_simulate_shape(cir_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_positive_rates(cir_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)

def test_simulate_different_paths(cir_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = cir_model.simulate(S0, T, N, steps)
    result2 = cir_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(cir_model):
    cir_model.params['kappa'].data.fill_(-1)
    cir_model.params['theta'].data.fill_(-1)
    cir_model.params['sigma'].data.fill_(-1)
    cir_model._apply_constraints()
    assert cir_model.params['kappa'].item() >= 1e-6
    assert cir_model.params['theta'].item() >= 1e-6
    assert cir_model.params['sigma'].item() >= 1e-6

def test_simulate_long_time_horizon(cir_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = cir_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)

def test_simulate_batch_consistency(cir_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = cir_model.simulate(S0, T, N1, steps)
    result2 = cir_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    cir_model = CIR()
    cir_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(cir_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert all(param.grad is not None for param in cir_model.params.values())

def test_mean_reversion(cir_model):
    S0, T, N, steps = 100, 10, 10000, 252*10
    result = cir_model.simulate(S0, T, N, steps)
    mean_rate = result.mean()
    assert torch.isclose(mean_rate, cir_model.params['theta'], rtol=0.1)

def test_zero_volatility():
    cir_model = CIR(sigma_init=0)
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)
    assert torch.allclose(result, result[0])  # All paths should be identical

def test_high_mean_reversion():
    cir_model = CIR(kappa_init=10)
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)
    assert torch.std(result) < 0.1 * cir_model.params['theta']  # Rates should be close to theta

def test_feller_condition():
    cir_model = CIR(kappa_init=0.1, theta_init=0.05, sigma_init=0.1)
    feller = 2 * cir_model.params['kappa'] * cir_model.params['theta'] - cir_model.params['sigma']**2
    assert feller > 0  # Feller condition should be satisfied

def test_negative_rates_handling():
    cir_model = CIR(r0_init=-0.01)  # Start with negative rate
    S0, T, N, steps = 100, 1, 1000, 252
    result = cir_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)  # All rates should be non-negative
