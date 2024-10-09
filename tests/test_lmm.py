import pytest
import torch
import numpy as np
from torchquantlib.models.interest_rate.lmm import LMM

@pytest.fixture
def lmm_model():
    forward_rates_init = [0.01, 0.015, 0.02, 0.025, 0.03]
    volatilities_init = [0.2, 0.18, 0.16, 0.14, 0.12]
    correlations_init = np.array([
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.9, 1.0, 0.9, 0.8, 0.7],
        [0.8, 0.9, 1.0, 0.9, 0.8],
        [0.7, 0.8, 0.9, 1.0, 0.9],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ])
    return LMM(forward_rates_init, volatilities_init, correlations_init)

def test_initialization(lmm_model):
    assert isinstance(lmm_model.forward_rates, torch.Tensor)
    assert isinstance(lmm_model.volatilities, torch.Tensor)
    assert isinstance(lmm_model.correlation_matrix, torch.Tensor)
    assert lmm_model.forward_rates.requires_grad
    assert lmm_model.volatilities.requires_grad
    assert not lmm_model.correlation_matrix.requires_grad

def test_simulate_shape(lmm_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    assert result.shape == (N, lmm_model.num_rates)

def test_simulate_positive_rates(lmm_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    assert torch.all(result >= 0)

def test_simulate_different_paths(lmm_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = lmm_model.simulate(S0, T, N, steps)
    result2 = lmm_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(lmm_model):
    lmm_model.volatilities.data.fill_(-1)
    lmm_model._apply_constraints()
    assert torch.all(lmm_model.volatilities >= 1e-6)

def test_simulate_long_time_horizon(lmm_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = lmm_model.simulate(S0, T, N, steps)
    assert result.shape == (N, lmm_model.num_rates)
    assert torch.all(result >= 0)

def test_simulate_batch_consistency(lmm_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = lmm_model.simulate(S0, T, N1, steps)
    result2 = lmm_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    forward_rates_init = [0.01, 0.015, 0.02, 0.025, 0.03]
    volatilities_init = [0.2, 0.18, 0.16, 0.14, 0.12]
    correlations_init = np.array([
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.9, 1.0, 0.9, 0.8, 0.7],
        [0.8, 0.9, 1.0, 0.9, 0.8],
        [0.7, 0.8, 0.9, 1.0, 0.9],
        [0.6, 0.7, 0.8, 0.9, 1.0]
    ])
    lmm_model = LMM(forward_rates_init, volatilities_init, correlations_init)
    lmm_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(lmm_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert lmm_model.forward_rates.grad is not None
    assert lmm_model.volatilities.grad is not None

def test_correlation_effect():
    forward_rates_init = [0.01, 0.015]
    volatilities_init = [0.2, 0.18]
    
    # High correlation
    correlations_high = np.array([[1.0, 0.9], [0.9, 1.0]])
    lmm_high_corr = LMM(forward_rates_init, volatilities_init, correlations_high)
    
    # Low correlation
    correlations_low = np.array([[1.0, 0.1], [0.1, 1.0]])
    lmm_low_corr = LMM(forward_rates_init, volatilities_init, correlations_low)
    
    S0, T, N, steps = 100, 1, 10000, 252
    result_high = lmm_high_corr.simulate(S0, T, N, steps)
    result_low = lmm_low_corr.simulate(S0, T, N, steps)
    
    corr_high = torch.corrcoef(result_high.T)[0, 1]
    corr_low = torch.corrcoef(result_low.T)[0, 1]
    
    assert corr_high > corr_low

def test_volatility_effect(lmm_model):
    S0, T, N, steps = 100, 1, 10000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    std_devs = torch.std(result, dim=0)
    assert torch.all(std_devs[:-1] > std_devs[1:])  # Earlier rates should have higher volatility

def test_negative_rates_handling(lmm_model):
    lmm_model.forward_rates.data.fill_(-0.01)  # Set negative initial rates
    S0, T, N, steps = 100, 1, 1000, 252
    result = lmm_model.simulate(S0, T, N, steps)
    assert torch.any(result < 0)  # LMM can produce negative rates
