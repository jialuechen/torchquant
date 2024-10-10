import pytest
import torch
from torchquantlib.core.risk.greeks.greeks import MalliavinGreeks

@pytest.fixture
def malliavin_greeks():
    return MalliavinGreeks()

@pytest.fixture
def option_params():
    return {
        'S0': torch.tensor(100.0),
        'K': torch.tensor(100.0),
        'T': torch.tensor(1.0),
        'r': torch.tensor(0.05),
        'sigma': torch.tensor(0.2),
        'num_paths': 100000,
        'seed': 42
    }

def test_european_option_greeks(malliavin_greeks, option_params):
    greeks = malliavin_greeks.european_option_greeks(**option_params)
    assert isinstance(greeks, dict)
    assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'}
    for greek in greeks.values():
        assert isinstance(greek, float)

def test_digital_option_greeks(malliavin_greeks, option_params):
    greeks = malliavin_greeks.digital_option_greeks(**option_params, Q=10.0)
    assert isinstance(greeks, dict)
    assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega'}
    for greek in greeks.values():
        assert isinstance(greek, float)

def test_barrier_option_greeks(malliavin_greeks, option_params):
    greeks = malliavin_greeks.barrier_option_greeks(**option_params, H=torch.tensor(110.0))
    assert isinstance(greeks, dict)
    assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega'}
    for greek in greeks.values():
        assert isinstance(greek, float)

def test_lookback_option_delta(malliavin_greeks, option_params):
    delta = malliavin_greeks.lookback_option_delta(**option_params)
    assert isinstance(delta, dict)
    assert set(delta.keys()) == {'Delta'}
    assert isinstance(delta['Delta'], float)

def test_basket_option_greeks(malliavin_greeks):
    S0 = torch.tensor([100.0, 110.0])
    sigma = torch.tensor([0.2, 0.25])
    rho = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    weights = torch.tensor([0.6, 0.4])
    greeks = malliavin_greeks.basket_option_greeks(S0, torch.tensor(100.0), torch.tensor(1.0),
                                                   torch.tensor(0.05), sigma, rho, weights)
    assert isinstance(greeks, dict)
    assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega'}
    for greek in greeks.values():
        assert isinstance(greek, list)
        assert len(greek) == len(S0)

def test_asian_option_greeks(malliavin_greeks, option_params):
    greeks = malliavin_greeks.asian_option_greeks(**option_params)
    assert isinstance(greeks, dict)
    assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega'}
    for greek in greeks.values():
        assert isinstance(greek, float)

def test_device_selection():
    cpu_greeks = MalliavinGreeks(device=torch.device('cpu'))
    assert cpu_greeks.device == torch.device('cpu')

    if torch.cuda.is_available():
        gpu_greeks = MalliavinGreeks(device=torch.device('cuda'))
        assert gpu_greeks.device == torch.device('cuda')

def test_invalid_barrier_option_type(malliavin_greeks, option_params):
    with pytest.raises(ValueError):
        malliavin_greeks.barrier_option_greeks(**option_params, H=torch.tensor(110.0), option_type='invalid')
