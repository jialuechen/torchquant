import pytest
import torch
from torchquantlib.models.local_volatility.dupire_local_volatility import DupireLocalVol

@pytest.fixture
def constant_vol_func():
    return lambda S, t: torch.full_like(S, 0.2)

@pytest.fixture
def smile_vol_func():
    return lambda S, t: 0.2 + 0.1 * torch.abs(torch.log(S / 100))

@pytest.fixture
def dupire_model(constant_vol_func):
    return DupireLocalVol(constant_vol_func)

def test_initialization(dupire_model):
    assert hasattr(dupire_model, 'local_vol_func')
    assert callable(dupire_model.local_vol_func)

def test_simulate_shape(dupire_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_positive_prices(dupire_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_different_paths(dupire_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = dupire_model.simulate(S0, T, N, steps)
    result2 = dupire_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_simulate_constant_vol(dupire_model):
    S0, T, N, steps = 100, 1, 10000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    log_returns = torch.log(result / S0)
    mean = log_returns.mean()
    std = log_returns.std()
    assert torch.isclose(mean, torch.tensor(0.0), atol=1e-2)
    assert torch.isclose(std, torch.tensor(0.2), atol=1e-2)

def test_simulate_smile_vol(smile_vol_func):
    dupire_model = DupireLocalVol(smile_vol_func)
    S0, T, N, steps = 100, 1, 10000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_long_time_horizon(dupire_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_batch_consistency(dupire_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = dupire_model.simulate(S0, T, N1, steps)
    result2 = dupire_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility(constant_vol_func):
    dupire_model = DupireLocalVol(constant_vol_func)
    dupire_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_time_dependent_vol():
    time_dep_vol_func = lambda S, t: 0.2 + 0.1 * t
    dupire_model = DupireLocalVol(time_dep_vol_func)
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_zero_volatility():
    zero_vol_func = lambda S, t: torch.zeros_like(S)
    dupire_model = DupireLocalVol(zero_vol_func)
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.allclose(result, torch.full_like(result, S0))

def test_high_volatility():
    high_vol_func = lambda S, t: torch.full_like(S, 2.0)
    dupire_model = DupireLocalVol(high_vol_func)
    S0, T, N, steps = 100, 1, 1000, 252
    result = dupire_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)
    assert result.std() > S0  # High volatility should lead to high dispersion

def test_negative_volatility():
    neg_vol_func = lambda S, t: torch.full_like(S, -0.2)
    dupire_model = DupireLocalVol(neg_vol_func)
    S0, T, N, steps = 100, 1, 1000, 252
    with pytest.raises(RuntimeError):  # Expecting runtime error due to NaN values
        dupire_model.simulate(S0, T, N, steps)
