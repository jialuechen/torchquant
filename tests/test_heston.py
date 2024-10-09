import pytest
import torch
from torchquantlib.models.stochastic_volatility.heston import Heston

@pytest.fixture
def heston_model():
    return Heston(kappa_init=1.0, theta_init=0.04, sigma_v_init=0.5, rho_init=-0.5, v0_init=0.04, mu_init=0.0)

def test_initialization(heston_model):
    assert isinstance(heston_model.params['kappa'], torch.Tensor)
    assert isinstance(heston_model.params['theta'], torch.Tensor)
    assert isinstance(heston_model.params['sigma_v'], torch.Tensor)
    assert isinstance(heston_model.params['rho'], torch.Tensor)
    assert isinstance(heston_model.params['v0'], torch.Tensor)
    assert isinstance(heston_model.params['mu'], torch.Tensor)
    assert all(param.requires_grad for param in heston_model.params.values())

def test_simulate_shape(heston_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    assert result.shape == (N,)

def test_simulate_positive_prices(heston_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_different_paths(heston_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result1 = heston_model.simulate(S0, T, N, steps)
    result2 = heston_model.simulate(S0, T, N, steps)
    assert not torch.allclose(result1, result2)

def test_apply_constraints(heston_model):
    heston_model.params['theta'].data.fill_(-1)
    heston_model.params['sigma_v'].data.fill_(-1)
    heston_model.params['v0'].data.fill_(-1)
    heston_model.params['rho'].data.fill_(2)
    heston_model._apply_constraints()
    assert heston_model.params['theta'].item() >= 1e-6
    assert heston_model.params['sigma_v'].item() >= 1e-6
    assert heston_model.params['v0'].item() >= 1e-6
    assert -0.999 <= heston_model.params['rho'].item() <= 0.999

def test_option_price_call(heston_model):
    price = heston_model.option_price(S0=100, K=100, T=1, r=0.05, option_type='call', N=10000, steps=100)
    assert isinstance(price, float)
    assert price > 0

def test_option_price_put(heston_model):
    price = heston_model.option_price(S0=100, K=100, T=1, r=0.05, option_type='put', N=10000, steps=100)
    assert isinstance(price, float)
    assert price > 0

def test_option_price_invalid_type(heston_model):
    with pytest.raises(ValueError):
        heston_model.option_price(S0=100, K=100, T=1, r=0.05, option_type='invalid', N=10000, steps=100)

def test_simulate_long_time_horizon(heston_model):
    S0, T, N, steps = 100, 30, 1000, 252*30
    result = heston_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_simulate_batch_consistency(heston_model):
    S0, T, N1, N2, steps = 100, 1, 1000, 2000, 252
    result1 = heston_model.simulate(S0, T, N1, steps)
    result2 = heston_model.simulate(S0, T, N2, steps)
    assert result1.shape[0] == N1
    assert result2.shape[0] == N2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    heston_model = Heston()
    heston_model.to('cuda')
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    assert result.device.type == 'cuda'

def test_gradient_flow(heston_model):
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    loss = result.mean()
    loss.backward()
    assert all(param.grad is not None for param in heston_model.params.values())

def test_option_price_sensitivity(heston_model):
    price1 = heston_model.option_price(S0=100, K=100, T=1, r=0.05, option_type='call', N=10000, steps=100)
    price2 = heston_model.option_price(S0=100, K=110, T=1, r=0.05, option_type='call', N=10000, steps=100)
    assert price1 > price2  # Option price should decrease as strike increases

def test_option_price_put_call_parity(heston_model):
    S0, K, T, r = 100, 100, 1, 0.05
    call_price = heston_model.option_price(S0, K, T, r, option_type='call', N=10000, steps=100)
    put_price = heston_model.option_price(S0, K, T, r, option_type='put', N=10000, steps=100)
    parity = call_price - put_price - (S0 - K * torch.exp(-r * T).item())
    assert abs(parity) < 1e-2  # Allow for some Monte Carlo error

def test_zero_volatility(heston_model):
    heston_model.params['sigma_v'].data.fill_(0)
    heston_model.params['v0'].data.fill_(0)
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)

def test_high_mean_reversion(heston_model):
    heston_model.params['kappa'].data.fill_(10)
    S0, T, N, steps = 100, 1, 1000, 252
    result = heston_model.simulate(S0, T, N, steps)
    assert torch.all(result > 0)
