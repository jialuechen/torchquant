import pytest
import torch
from torchquantlib.models.interest_rate.vasicek import Vasicek

@pytest.fixture
def vasicek_model():
    return Vasicek(kappa_init=0.1, theta_init=0.05, sigma_init=0.01, r0_init=0.03)

def test_initialization(vasicek_model):
    assert isinstance(vasicek_model.params['kappa'], torch.Tensor)
    assert isinstance(vasicek_model.params['theta'], torch.Tensor)
    assert isinstance(vasicek_model.params['sigma'], torch.Tensor)
    assert isinstance(vasicek_model.params['r0'], torch.Tensor)
    assert all(param.requires_grad for param in vasicek_model.params.values())

def test_simulate_shape(vasicek_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_different_paths(vasicek_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = vasicek_model.simulate(S0, T, N, steps)
    result2 = vasicek_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(vasicek_model):
    vasicek_model.params['kappa'].data.fill_(-1)
    vasicek_model.params['sigma'].data.fill_(-1)
    vasicek_model._apply_constraints()
    assert vasicek_model.params['kappa'].item() >= 1e-6
    assert vasicek_model.params['sigma'].item() >= 1e-6

def test_simulate_long_time_horizon(vasicek_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = vasicek_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_batch_consistency(vasicek_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = vasicek_model.simulate(S0, T, N1, steps)
    result2 = vasicek_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    vasicek_model = Vasicek()
    vasicek_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(vasicek_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert all(param.grad is not None for param in vasicek_model.params.values())

def test_mean_reversion(vasicek_model):
    S0, T, N, steps = 100, 10, 10000, 252*10
    result = vasicek_model.simulate(S0, T, N, steps)
    mean_rate = result.mean()
    assert torch.isclose(mean_rate, vasicek_model.params['theta'], rtol=0.1)

def test_zero_volatility():
    vasicek_model = Vasicek(sigma_init=0)
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    assert torch.allclose(result, result[0])  # All paths should be identical

def test_high_mean_reversion():
    vasicek_model = Vasicek(kappa_init=10)
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    assert torch.std(result) < 0.1 * vasicek_model.params['theta']  # Rates should be close to theta

def test_negative_rates():
    vasicek_model = Vasicek(r0_init=-0.01, theta_init=-0.02)  # Start with negative rate and negative long-term mean
    S0, T, N, steps = 100, 1, 1000, 252
    result = vasicek_model.simulate(S0, T, N, steps)
    assert torch.any(result < 0)  # Vasicek model allows for negative rates

def test_parameter_sensitivity():
    vasicek_model1 = Vasicek(kappa_init=0.1, sigma_init=0.01)
    vasicek_model2 = Vasicek(kappa_init=0.2, sigma_init=0.02)
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = vasicek_model1.simulate(S0, T, N, steps)
    result2 = vasicek_model2.simulate(S0, T, N, steps)
    assert torch.std(result2) > torch.std(result1)  # Higher volatility should lead to more dispersion

def test_long_term_distribution():
    vasicek_model = Vasicek(kappa_init=0.5, theta_init=0.05, sigma_init=0.02)
    S0, T, N, steps = 100, 50, 10000, 252*50
    result = vasicek_model.simulate(S0, T, N, steps)
    theoretical_mean = vasicek_model.params['theta']
    theoretical_var = (vasicek_model.params['sigma']**2) / (2 * vasicek_model.params['kappa'])
    assert torch.isclose(result.mean(), theoretical_mean, rtol=0.1)
    assert torch.isclose(result.var(), theoretical_var, rtol=0.1)
