import pytest
import torch
from torchquantlib.models.stochastic_volatility.sabr import SABR

@pytest.fixture
def sabr_model():
    return SABR(alpha_init=0.2, beta_init=0.5, rho_init=0.0, nu_init=0.3, F0=100.0)

def test_initialization(sabr_model):
    assert isinstance(sabr_model.params['alpha'], torch.Tensor)
    assert isinstance(sabr_model.params['beta'], torch.Tensor)
    assert isinstance(sabr_model.params['rho'], torch.Tensor)
    assert isinstance(sabr_model.params['nu'], torch.Tensor)
    assert sabr_model.params['alpha'].requires_grad
    assert sabr_model.params['beta'].requires_grad
    assert sabr_model.params['rho'].requires_grad
    assert sabr_model.params['nu'].requires_grad

def test_simulate_shape(sabr_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = sabr_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_positive_rates(sabr_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = sabr_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_different_paths(sabr_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = sabr_model.simulate(S0, T, N, steps)
    result2 = sabr_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(sabr_model):
    sabr_model.params['alpha'].data.fill_(-1)
    sabr_model.params['beta'].data.fill_(2)
    sabr_model.params['rho'].data.fill_(2)
    sabr_model.params['nu'].data.fill_(-1)
    sabr_model._apply_constraints()
    assert sabr_model.params['alpha'].item() >= 1e-6
    assert 0 <= sabr_model.params['beta'].item() <= 1
    assert -0.999 <= sabr_model.params['rho'].item() <= 0.999
    assert sabr_model.params['nu'].item() >= 1e-6

def test_option_price_call(sabr_model):
    price = sabr_model.option_price(K=100, T=1, r=0.05, option_type='call', N=10000, steps=100)
    assert isinstance(price, float)
    assert price > 0

def test_option_price_put(sabr_model):
    price = sabr_model.option_price(K=100, T=1, r=0.05, option_type='put', N=10000, steps=100)
    assert isinstance(price, float)
    assert price > 0

def test_option_price_invalid_type(sabr_model):
    with pytest.raises(ValueError):
        sabr_model.option_price(K=100, T=1, r=0.05, option_type='invalid', N=10000, steps=100)

def test_simulate_long_time_horizon(sabr_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = sabr_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_batch_consistency(sabr_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = sabr_model.simulate(S0, T, N1, steps)
    result2 = sabr_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    sabr_model = SABR()
    sabr_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = sabr_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(sabr_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = sabr_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert sabr_model.params['alpha'].grad is not None
    assert sabr_model.params['beta'].grad is not None
    assert sabr_model.params['rho'].grad is not None
    assert sabr_model.params['nu'].grad is not None

def test_option_price_sensitivity(sabr_model):
    price1 = sabr_model.option_price(K=100, T=1, r=0.05, option_type='call', N=10000, steps=100)
    price2 = sabr_model.option_price(K=110, T=1, r=0.05, option_type='call', N=10000, steps=100)
    assert price1 > price2  # Option price should decrease as strike increases

def test_option_price_put_call_parity(sabr_model):
    K, T, r = 100, 1, 0.05
    call_price = sabr_model.option_price(K, T, r, option_type='call', N=10000, steps=100)
    put_price = sabr_model.option_price(K, T, r, option_type='put', N=10000, steps=100)
    F0 = sabr_model.F0.item()
    parity = call_price - put_price - (F0 - K * torch.exp(-r * T).item())
    assert abs(parity) < 1e-2  # Allow for some Monte Carlo error
