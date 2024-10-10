Reduced Form Model
==================

.. currentmodule:: torchquantlib.core.risk.credit_risk.reduced_form_model

This module implements a simple reduced-form model for credit risk calculation.

Functions
---------

.. autofunction:: reduced_form_model

Detailed Description
--------------------

The `reduced_form_model` function implements a basic reduced-form model for credit risk assessment. This model is used to calculate the expected loss of a credit instrument over a given time horizon.

Key features of this implementation:

1. Assumes a constant hazard rate (default intensity)
2. Uses a constant recovery rate
3. Calculates survival probability based on the default intensity and time
4. Computes expected loss as a function of recovery rate and survival probability

Mathematical Background
^^^^^^^^^^^^^^^^^^^^^^^

The model uses the following key equations:

1. Survival Probability: :math:`P(survival) = e^{-\lambda t}`
   Where :math:`\lambda` is the default intensity and :math:`t` is the time horizon.

2. Expected Loss: :math:`E(Loss) = (1 - R) * (1 - P(survival))`
   Where :math:`R` is the recovery rate.

Usage Example
^^^^^^^^^^^^^

Here's a basic example of how to use the `reduced_form_model` function:

.. code-block:: python

    import torch
    from torchquantlib.core.risk.credit_risk.reduced_form_model import reduced_form_model

    # Set up parameters
    lambda_0 = torch.tensor(0.05)
    default_intensity = torch.tensor(0.03)
    recovery_rate = torch.tensor(0.4)
    time = torch.tensor(5.0)

    # Calculate expected loss
    expected_loss = reduced_form_model(lambda_0, default_intensity, recovery_rate, time)
    print(f"Expected Loss: {expected_loss.item():.4f}")

Note
^^^^

The current implementation does not use the `lambda_0` parameter. This parameter is included for potential future extensions of the model, such as implementing time-varying default intensities.