Examples
========

This section provides examples of how to use TorchQuant for various quantitative finance tasks.

Model Calibration
-----------------

Here's an example of calibrating a model using TorchQuant:

.. code-block:: python

   # calibrate_heston.py

    import numpy as np
    import torch
    from torchquantlib.calibration.model_calibrator import ModelCalibrator
    from torchquantlib.models.stochastic_volatility.heston import Heston

    # Generate synthetic observed data using true Heston parameters
    N_observed = 1000
    S0 = 100.0
    T = 1.0
    true_params = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma_v': 0.3,
        'rho': -0.7,
        'v0': 0.04,
        'mu': 0.05
    }

    np.random.seed(42)
    torch.manual_seed(42)
    heston_true = Heston(**true_params)
    S_observed = heston_true.simulate(S0=S0, T=T, N=N_observed)

    # Initialize the Heston model with initial guesses
    heston_model = Heston(
        kappa_init=1.0,
        theta_init=0.02,
        sigma_v_init=0.2,
        rho_init=-0.5,
        v0_init=0.02,
        mu_init=0.0
    )

    # Set up the calibrator
    calibrator = ModelCalibrator(
        model=heston_model,
        observed_data=S_observed.detach().cpu().numpy(),  # Convert tensor to numpy array
        S0=S0,
        T=T,
        lr=0.01
    )

    # Calibrate the model
    calibrator.calibrate(num_epochs=1000, steps=100, verbose=True)

    # Get the calibrated parameters
    calibrated_params = calibrator.get_calibrated_params()
    print("Calibrated Parameters:")
    for name, value in calibrated_params.items():
        print(f"{name}: {value:.6f}")

More examples covering different aspects of TorchQuant will be added in future updates.