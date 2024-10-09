import pytest
import torch
from torchquantlib.models.stochastic_model import StochasticModel

class DummyModel(StochasticModel):
    def simulate(self, S0, T, N, **kwargs):
        return torch.ones(N) * S0

    def _apply_constraints(self):
        self.params['dummy'].data.clamp_(min=0)

@pytest.fixture
def dummy_model():
    params = {'dummy': torch.tensor(1.0, requires_grad=True)}
    return DummyModel(params)

def test_initialization():
    params = {'param1': torch.tensor(1.0, requires_grad=True),
              'param2': torch.tensor(2.0, requires_grad=True)}
    model = StochasticModel(params)
    assert isinstance(model.params, dict)
    assert 'param1' in model.params
    assert 'param2' in model.params
    assert model.params['param1'].requires_grad
    assert model.params['param2'].requires_grad

def test_device_assignment():
    params = {'param': torch.tensor(1.0, requires_grad=True)}
    model = StochasticModel(params)
    expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert model.device == expected_device
    assert model.params['param'].device == expected_device

def test_simulate_not_implemented():
    params = {'param': torch.tensor(1.0, requires_grad=True)}
    model = StochasticModel(params)
    with pytest.raises(NotImplementedError):
        model.simulate(100, 1, 1000)

def test_apply_constraints_base():
    params = {'param': torch.tensor(1.0, requires_grad=True)}
    model = StochasticModel(params)
    # Should not raise an error
    model._apply_constraints()

def test_dummy_model_simulate(dummy_model):
    result = dummy_model.simulate(100, 1, 1000)
    assert result.shape == (1000,)
    assert torch.all(result == 100)

def test_dummy_model_apply_constraints(dummy_model):
    dummy_model.params['dummy'].data.fill_(-1)
    dummy_model._apply_constraints()
    assert dummy_model.params['dummy'].item() == 0

def test_parameter_gradient(dummy_model):
    result = dummy_model.simulate(100, 1, 1000)
    loss = result.mean()
    loss.backward()
    assert dummy_model.params['dummy'].grad is not None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    params = {'param': torch.tensor(1.0, requires_grad=True)}
    model = StochasticModel(params)
    assert model.device.type == 'cuda'
    assert model.params['param'].device.type == 'cuda'

def test_multiple_parameters():
    params = {'param1': torch.tensor(1.0, requires_grad=True),
              'param2': torch.tensor(2.0, requires_grad=True),
              'param3': torch.tensor(3.0, requires_grad=True)}
    model = StochasticModel(params)
    assert len(model.params) == 3
    for param in model.params.values():
        assert param.requires_grad

def test_non_tensor_parameter():
    with pytest.raises(AttributeError):
        StochasticModel({'param': 1.0})  # Should be a tensor
