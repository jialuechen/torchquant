import pytest
import torch
import numpy as np
from torchquantlib.calibration.model_calibrator import ModelCalibrator

class MockModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.params = {
            'mu': torch.tensor(0.05, requires_grad=True),
            'sigma': torch.tensor(0.2, requires_grad=True)
        }

    def simulate(self, S0, T, N, steps):
        return torch.randn(N, steps) * self.params['sigma'] + self.params['mu']

    def _apply_constraints(self):
        self.params['sigma'].data.clamp_(min=0.01)

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def observed_data():
    return np.random.randn(1000, 100) * 0.15 + 0.03

def test_model_calibrator_initialization(mock_model, observed_data):
    calibrator = ModelCalibrator(mock_model, observed_data)
    assert isinstance(calibrator, ModelCalibrator)
    assert calibrator.model == mock_model
    assert torch.is_tensor(calibrator.observed_data)
    assert calibrator.observed_data.shape == (1000, 100)

def test_model_calibrator_calibration(mock_model, observed_data):
    calibrator = ModelCalibrator(mock_model, observed_data)
    initial_params = calibrator.get_calibrated_params()
    
    calibrator.calibrate(num_epochs=10, verbose=False)
    
    calibrated_params = calibrator.get_calibrated_params()
    assert calibrated_params != initial_params
    assert 'mu' in calibrated_params
    assert 'sigma' in calibrated_params

def test_model_calibrator_get_calibrated_params(mock_model, observed_data):
    calibrator = ModelCalibrator(mock_model, observed_data)
    params = calibrator.get_calibrated_params()
    assert isinstance(params, dict)
    assert 'mu' in params
    assert 'sigma' in params
    assert isinstance(params['mu'], float)
    assert isinstance(params['sigma'], float)

def test_model_calibrator_with_custom_optimizer(mock_model, observed_data):
    calibrator = ModelCalibrator(mock_model, observed_data, optimizer_cls=torch.optim.SGD, lr=0.1)
    assert isinstance(calibrator.optimizer, torch.optim.SGD)
    assert calibrator.optimizer.param_groups[0]['lr'] == 0.1

def test_model_calibrator_with_custom_loss(mock_model, observed_data):
    calibrator = ModelCalibrator(mock_model, observed_data, loss_type="sinkhorn", p=1, blur=0.1)
    assert calibrator.loss_fn.p == 1
    assert calibrator.loss_fn.blur == 0.1
