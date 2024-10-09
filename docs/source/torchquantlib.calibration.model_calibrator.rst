ModelCalibrator
===============

.. autoclass:: torchquantlib.calibration.model_calibrator.ModelCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Class Description
-----------------

The ``ModelCalibrator`` class is designed to calibrate a stochastic model using the Sinkhorn divergence. It utilizes the geomloss library for calculating the loss.

Parameters
----------

- ``model``: The stochastic model to be calibrated.
- ``observed_data``: The observed market data used for calibration.
- ``S0`` (optional): Initial value. Default is None.
- ``T`` (float): Time horizon. Default is 1.0.
- ``loss_type`` (str): Type of loss function. Default is "sinkhorn".
- ``p`` (int): Power parameter for the loss function. Default is 2.
- ``blur`` (float): Blur parameter for the loss function. Default is 0.05.
- ``optimizer_cls``: Optimizer class. Default is ``torch.optim.Adam``.
- ``lr`` (float): Learning rate for the optimizer. Default is 0.01.

Methods
-------

__init__(self, model, observed_data, S0=None, T=1.0, loss_type="sinkhorn", p=2, blur=0.05, optimizer_cls=optim.Adam, lr=0.01)
    Initialize the ModelCalibrator with the given parameters.

calibrate(self, num_epochs=1000, batch_size=None, steps=100, verbose=True)
    Perform the calibration process.

get_calibrated_params(self)
    Retrieve the calibrated parameters after calibration.


