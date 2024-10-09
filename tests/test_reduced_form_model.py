import pytest
import torch
from torchquantlib.core.risk.credit_risk.reduced_form_model import reduced_form_model

@pytest.fixture
def setup_tensors():
    return {
        'lambda_0': torch.tensor(0.02),
        'default_intensity': torch.tensor(0.03),
        'recovery_rate': torch.tensor(0.4),
        'time': torch.tensor(5.0)
    }

def test_reduced_form_model_basic(setup_tensors):
    expected_loss = reduced_form_model(**setup_tensors)
    assert 0 <= expected_loss <= 1
    assert torch.is_tensor(expected_loss)

def test_reduced_form_model_zero_default_intensity(setup_tensors):
    setup_tensors['default_intensity'] = torch.tensor(0.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss == 0

def test_reduced_form_model_full_recovery(setup_tensors):
    setup_tensors['recovery_rate'] = torch.tensor(1.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss == 0

def test_reduced_form_model_no_recovery(setup_tensors):
    setup_tensors['recovery_rate'] = torch.tensor(0.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss > 0
    assert expected_loss <= 1

def test_reduced_form_model_zero_time(setup_tensors):
    setup_tensors['time'] = torch.tensor(0.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss == 0

def test_reduced_form_model_long_time(setup_tensors):
    setup_tensors['time'] = torch.tensor(100.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss > 0
    assert expected_loss <= 1

def test_reduced_form_model_high_default_intensity(setup_tensors):
    setup_tensors['default_intensity'] = torch.tensor(1.0)
    expected_loss = reduced_form_model(**setup_tensors)
    assert expected_loss > 0
    assert expected_loss <= 1

def test_reduced_form_model_batch_input():
    lambda_0 = torch.tensor([0.02, 0.03, 0.04])
    default_intensity = torch.tensor([0.03, 0.04, 0.05])
    recovery_rate = torch.tensor([0.4, 0.5, 0.6])
    time = torch.tensor([5.0, 6.0, 7.0])
    
    expected_loss = reduced_form_model(lambda_0, default_intensity, recovery_rate, time)
    assert expected_loss.shape == (3,)
    assert torch.all((0 <= expected_loss) & (expected_loss <= 1))

def test_reduced_form_model_lambda_0_not_used(setup_tensors):
    result1 = reduced_form_model(**setup_tensors)
    setup_tensors['lambda_0'] = torch.tensor(0.05)  # Change lambda_0
    result2 = reduced_form_model(**setup_tensors)
    assert torch.isclose(result1, result2)
